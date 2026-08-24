"""
Module providing a wrapper class for Jina-clip-v2 model to extract hidden states.

This module defines a simple interface for using Jina-clip-v2's text encoder model
to generate hidden states from text inputs, which can then be cached and used by
components like the JinaToSDXLAdapter.
"""

import os
import torch
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple
from library.transformers_compat import (
    JINA_CLIP_TRUST_REMOTE_CODE,
    ensure_legacy_clip_symbols,
    jina_clip_load_context,
    load_jina_clip_tokenizer,
    pretrained_dtype_kwargs,
    repair_jina_clip_nonpersistent_buffers,
)

ensure_legacy_clip_symbols()

from transformers import AutoImageProcessor, AutoModel
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


def restore_text_sequence_shape(
    state: torch.Tensor,
    attention_mask: torch.Tensor,
    label: str = "Jina hidden state",
) -> torch.Tensor:
    """Restore flash-attention's unpadded ``[valid_tokens, D]`` layer output."""
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{label} must be a Tensor, got {type(state)}.")
    while state.dim() > 3 and state.shape[0] == 1:
        state = state.squeeze(0)

    batch, sequence = attention_mask.shape
    if state.dim() == 3:
        if tuple(state.shape[:2]) != (batch, sequence):
            raise RuntimeError(
                f"{label} has sequence shape {tuple(state.shape[:2])}, expected {(batch, sequence)}."
            )
        return state
    if state.dim() != 2:
        raise RuntimeError(
            f"{label} must be [B, L, D] or flattened [tokens, D], got {tuple(state.shape)}."
        )

    if state.shape[0] == batch * sequence:
        return state.reshape(batch, sequence, state.shape[-1])

    valid_mask = attention_mask.to(device=state.device, dtype=torch.bool)
    valid_tokens = int(valid_mask.sum().item())
    if state.shape[0] != valid_tokens:
        raise RuntimeError(
            f"{label} contains {state.shape[0]} flattened tokens, but the attention mask has "
            f"{valid_tokens} valid tokens ({batch}x{sequence} padded shape)."
        )
    restored = state.new_zeros(batch, sequence, state.shape[-1])
    restored[valid_mask] = state
    return restored


def assemble_selected_text_states(
    captured_states: List[torch.Tensor],
    final_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_numbers: Tuple[int, ...],
    final_layer_number: int,
) -> torch.Tensor:
    """Restore and stack selected layer hooks as ``[B, layers, L, D]``."""
    if len(captured_states) != len(layer_numbers):
        raise RuntimeError(
            f"Captured {len(captured_states)} states for {len(layer_numbers)} requested layers."
        )
    selected = torch.stack(
        [
            restore_text_sequence_shape(
                state,
                attention_mask,
                f"Jina text layer h{layer_number}",
            )
            for layer_number, state in zip(layer_numbers, captured_states)
        ],
        dim=1,
    )
    if layer_numbers[-1] == final_layer_number:
        final_hidden_state = restore_text_sequence_shape(
            final_hidden_state,
            attention_mask,
            "Jina final text state",
        )
        selected = torch.cat(
            [selected[:, :-1], final_hidden_state.unsqueeze(1)],
            dim=1,
        )
    return selected

class TextDataset(Dataset):
    """Simple dataset to efficiently load text files from a directory."""
    def __init__(self, directory: str, files: Optional[List[Path]] = None):
        if files is None:
            self.files = sorted(Path(directory).glob("*.txt"))
        else:
            self.files = list(files)
        
    def __len__(self) -> int:
        return len(self.files)
        
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        file_path = self.files[idx]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # Handle empty files to prevent tokenization/pooling crashes
        if not text:
            text = " "
        return str(file_path), text

def custom_collate(batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Collate function for the TextDataset to return lists of files and texts."""
    files = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return files, texts


class JinaStates:
    """
    Extracts hidden states from jina-clip-v2 dynamically using PyTorch hooks.
    This safely bypasses Jina's custom wrappers without needing hardcoded model paths.
    """

    def __init__(self,
                 model_id: str,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16,
                 max_length: int = 512,
                 custom_train_jina: bool = False,
                 init_add_artist_tag=False,
                 keep_vision_model: bool = False,
                 image_only: bool = False,
                 num_hidden_state_layers: int = 1,
                 hidden_state_layer_indices: Optional[Tuple[int, ...]] = None,
                 revision: Optional[str] = None,
                 code_revision: Optional[str] = None,
                 local_files_only: bool = False):
        
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.custom_train_jina = custom_train_jina
        self.keep_vision_model = keep_vision_model
        self.image_only = image_only
        self.revision = revision
        self.code_revision = code_revision
        # A local model selection must stay local. Jina's trusted architecture
        # performs a nested AutoModel.from_config lookup for jina-embeddings-v3,
        # which otherwise may contact the Hub even though model_id is a directory.
        self.local_files_only = bool(local_files_only or Path(model_id).is_dir())
        self.num_hidden_state_layers = int(num_hidden_state_layers)
        if self.num_hidden_state_layers not in (1, 4):
            raise ValueError("num_hidden_state_layers must be either 1 (V2) or 4 (V3).")
        self.hidden_state_layer_indices = (
            tuple(int(index) for index in hidden_state_layer_indices)
            if hidden_state_layer_indices is not None
            else None
        )
        if self.hidden_state_layer_indices is not None:
            if not self.hidden_state_layer_indices:
                raise ValueError("hidden_state_layer_indices cannot be empty.")
            if len(set(self.hidden_state_layer_indices)) != len(self.hidden_state_layer_indices):
                raise ValueError("hidden_state_layer_indices cannot contain duplicates.")
            if tuple(sorted(self.hidden_state_layer_indices)) != self.hidden_state_layer_indices:
                raise ValueError("hidden_state_layer_indices must be in increasing encoder-depth order.")
        self.image_processor = None
        self.tokenizer = None
        if not image_only:
            print(f"Loading tokenizer from {model_id}...")
            self.tokenizer = load_jina_clip_tokenizer(
                model_id,
                local_files_only=self.local_files_only,
                revision=self.revision,
            )
        if keep_vision_model or image_only:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_id,
                trust_remote_code=JINA_CLIP_TRUST_REMOTE_CODE,
                local_files_only=self.local_files_only,
                revision=self.revision,
                use_fast=True
            )
        
        print(f"Loading model from {model_id}...")
        model_load_kwargs = {
            "low_cpu_mem_usage": False,
            "trust_remote_code": JINA_CLIP_TRUST_REMOTE_CODE,
            "local_files_only": self.local_files_only,
            **pretrained_dtype_kwargs(dtype),
        }
        if self.revision is not None:
            model_load_kwargs["revision"] = self.revision
        if self.code_revision is not None:
            model_load_kwargs["code_revision"] = self.code_revision
        with jina_clip_load_context(local_files_only=self.local_files_only):
            self.model = AutoModel.from_pretrained(model_id, **model_load_kwargs)
        print(
            "Loaded Jina custom architecture with trust_remote_code=True: "
            f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
        )
        buffer_repair = repair_jina_clip_nonpersistent_buffers(self.model)
        if buffer_repair["repaired"]:
            print(
                "Reinitialized Jina checkpoint-omitted buffers left uninitialized by the Transformers loader: "
                f"{buffer_repair['lora_dropout_masks']} LoRA masks, "
                f"{buffer_repair['rotary_inv_freq']} text rotary frequency buffers, "
                f"{buffer_repair['eva_rotary_freqs']} EVA vision rotary frequency buffers."
            )
        self.model.to(device)
        
        # 1. Unload the Vision Tower to save massive VRAM
        if hasattr(self.model, "vision_model") and not keep_vision_model:
            del self.model.vision_model
            torch.cuda.empty_cache()
            print("Vision tower successfully unloaded to save VRAM.")

        if image_only:
            self._unload_text_tower()
            self.model.requires_grad_(False)
            self.model.eval()
            self.hidden_states_cache = None
            self.hidden_state_layers_cache = None
            self.encoder_module = None
            self.layer_list = None
            print("Jina image-only mode: text tower unloaded, vision tower kept for IP-Adapter.")
            return
            
        self.model.eval()
        
        # 2. Dynamically locate the main text transformer and attach a forward hook
        # This allows us to intercept the sequence hidden_states regardless of how Jina wraps it.
        self.hidden_states_cache = None
        self.hidden_state_layers_cache = None
        self.encoder_module = None
        self.layer_list = None
        
        for name, module in self.model.named_modules():
            # Exclude anything vision-related just in case
            if 'vision' in name.lower():
                continue
                
            # Transformers house their sequence layers in ModuleLists
            has_layer = hasattr(module, 'layer') and isinstance(getattr(module, 'layer'), torch.nn.ModuleList)
            has_layers = hasattr(module, 'layers') and isinstance(getattr(module, 'layers'), torch.nn.ModuleList)
            has_block = hasattr(module, 'block') and isinstance(getattr(module, 'block'), torch.nn.ModuleList)
            has_blocks = hasattr(module, 'blocks') and isinstance(getattr(module, 'blocks'), torch.nn.ModuleList)
            
            if has_layer or has_layers or has_block or has_blocks:
                layer_list = (getattr(module, 'layer', None) or getattr(module, 'layers', None) or 
                              getattr(module, 'block', None) or getattr(module, 'blocks', None))
                # Ensure it's the main transformer backbone (has multiple layers)
                if layer_list is not None and len(layer_list) > 1:
                    self.encoder_module = module
                    self.layer_list = layer_list
                    break
                    
        if self.encoder_module is None:
            raise RuntimeError("Could not identify the text encoder module to attach a hook. Check model structure.")
        if self.num_hidden_state_layers > len(self.layer_list):
            raise ValueError(
                f"Requested the final {self.num_hidden_state_layers} Jina text layers, "
                f"but the encoder only has {len(self.layer_list)} layers."
            )
        if self.hidden_state_layer_indices is not None:
            invalid_indices = [
                index
                for index in self.hidden_state_layer_indices
                if index < 1 or index > len(self.layer_list)
            ]
            if invalid_indices:
                raise ValueError(
                    "Requested Jina text layer indices are outside the encoder's one-based range "
                    f"1-{len(self.layer_list)}: {invalid_indices}."
                )
        # Capture only the requested layers. Asking the transformer for
        # output_hidden_states materializes all 24 padded layers, which is
        # unnecessarily expensive for large training batches. Flash attention
        # exposes unpadded 2-D layer tensors to these hooks; they are restored
        # with the attention mask in __call__.
        # Added
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        if self.custom_train_jina:
            for name, module in self.model.named_modules():
                if 'vision' in name.lower():
                    continue
                if isinstance(module, torch.nn.Embedding):
                    for param in module.parameters():
                        param.requires_grad = True
                    module.train() 

            num_layers = len(self.layer_list)
            for i in range(num_layers - 2, num_layers):
                for param in self.layer_list[i].parameters():
                    param.requires_grad = True
                self.layer_list[i].train() 
                
            if hasattr(self.model, 'text_projection'):
                for param in getattr(self.model, 'text_projection').parameters():
                    param.requires_grad = True

            print(f"Jina Training ENABLED: Unfrozen text embeddings and the last 2 layers.")
        else:
            print("Jina Training DISABLED: Model is completely frozen.")
        if init_add_artist_tag:
            print("!!!Init add artist tag!!!")
            special_tokens_dict = {'additional_special_tokens': ['<artist>']}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            new_token_id = self.tokenizer.convert_tokens_to_ids("<artist>")
            print(f"Added token <artist> with ID: {new_token_id}")
            # Resize text encoder's embeddings
            self.model.resize_token_embeddings(len(self.tokenizer)) # could be self.model.text_model instead
            
            # Get the ID of a sensible initializer word (e.g., "artist")
            embeddings = self.model.get_input_embeddings() # could be self.model.text_model instead
            init_token_ids = self.tokenizer.encode("artist", add_special_tokens=False)
            init_token_id = init_token_ids[0]
            # Copy the pre-trained weights from "artist" to "<artist>"
            with torch.no_grad():
                embeddings.weight[new_token_id] = embeddings.weight[init_token_id].clone()


        # End of added
        def sequence_tensor(output):
            # Output from encoder is typically BaseModelOutput or a tuple
            if hasattr(output, 'last_hidden_state'):
                output = output.last_hidden_state
            elif isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"Jina text layer hook expected a Tensor output, got {type(output)}."
                )
            return output

        def forward_hook(module, args, output):
            self.hidden_states_cache = sequence_tensor(output)
                
        self.encoder_module.register_forward_hook(forward_hook)
        print(f"Successfully attached hidden-state hook to: {self.encoder_module.__class__.__name__}")

        captured_layer_indices = None
        if self.hidden_state_layer_indices is not None:
            captured_layer_indices = self.hidden_state_layer_indices
        elif self.num_hidden_state_layers > 1:
            first_layer_index = len(self.layer_list) - self.num_hidden_state_layers
            captured_layer_indices = tuple(
                range(first_layer_index + 1, len(self.layer_list) + 1)
            )

        if captured_layer_indices is not None:
            self.hidden_state_layers_cache = [None] * len(captured_layer_indices)
            for cache_index, one_based_index in enumerate(captured_layer_indices):
                layer = self.layer_list[one_based_index - 1]
                def layer_hook(module, args, output, cache_index=cache_index):
                    self.hidden_state_layers_cache[cache_index] = sequence_tensor(output)

                layer.register_forward_hook(layer_hook)

            print(
                "Successfully attached hidden-state hooks to Jina text layers "
                f"{', '.join(str(index) for index in captured_layer_indices)}."
            )

    def _unload_text_tower(self):
        deleted = []
        text_attrs = [
            "text_model",
            "language_model",
            "text_projection",
            "text_proj",
        ]
        for attr in text_attrs:
            if hasattr(self.model, attr):
                obj = getattr(self.model, attr)
                if obj is not getattr(self.model, "vision_model", None):
                    try:
                        delattr(self.model, attr)
                        deleted.append(attr)
                    except Exception:
                        pass
        if hasattr(self.model, "get_text_features"):
            try:
                self.model.get_text_features = None
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if deleted:
            print(f"Unloaded Jina text modules for image-only use: {', '.join(deleted)}")

    def _image_processor_size(self) -> Tuple[int, int]:
        size = getattr(self.image_processor, "crop_size", None) or getattr(self.image_processor, "size", None) or {}
        if isinstance(size, dict):
            height = size.get("height") or size.get("shortest_edge") or 512
            width = size.get("width") or size.get("shortest_edge") or height
        elif isinstance(size, (tuple, list)) and len(size) >= 2:
            height, width = size[0], size[1]
        else:
            height = width = int(size) if size else 512
        return int(height), int(width)

    def encode_image_hidden_states(
        self,
        images: torch.Tensor,
        return_layer_states: bool = False,
        max_layer_states: int = 4,
        selected_layer_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Encode image tensors with Jina CLIP-v2's vision tower.
        `images` are expected in sd-scripts format:[B, 3, H, W] in [-1, 1].
        """
        if not self.keep_vision_model:
            raise RuntimeError("JinaStates was initialized without the vision tower. Set keep_vision_model=True.")
        if not hasattr(self.model, "vision_model") and not hasattr(self.model, "get_image_features"):
            raise RuntimeError("Could not find Jina CLIP-v2 vision_model or get_image_features on the loaded model.")

        images = images.to(self.model.device, dtype=torch.float32)
        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)

        height, width = self._image_processor_size()
        images = F.interpolate(images, size=(height, width), mode="bicubic", align_corners=False)

        mean = getattr(self.image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073))
        std = getattr(self.image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711))
        mean = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images = (images - mean) / std
        images = images.to(dtype=self.dtype)

        if selected_layer_indices is not None:
            selected_layers = tuple(int(layer) for layer in selected_layer_indices)
            if not selected_layers or any(layer <= 0 for layer in selected_layers):
                raise ValueError(
                    "selected_layer_indices must contain one-based positive vision-block indices."
                )
            if len(set(selected_layers)) != len(selected_layers) or tuple(sorted(selected_layers)) != selected_layers:
                raise ValueError(
                    f"selected_layer_indices must be unique and increasing, got {selected_layers}."
                )
            if not hasattr(self.model, "vision_model"):
                raise RuntimeError("Selected vision-layer capture requires Jina's vision_model.")

            vision_model = self.model.vision_model
            block_owner = vision_model
            blocks = getattr(vision_model, "blocks", None)
            if not isinstance(blocks, torch.nn.ModuleList):
                for module in vision_model.modules():
                    candidate = getattr(module, "blocks", None)
                    if isinstance(candidate, torch.nn.ModuleList) and len(candidate) >= max(selected_layers):
                        blocks = candidate
                        block_owner = module
                        break
            if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < max(selected_layers):
                available = len(blocks) if isinstance(blocks, torch.nn.ModuleList) else 0
                raise RuntimeError(
                    "Could not capture the requested Jina EVA vision blocks "
                    f"{selected_layers}; found {available} blocks."
                )

            captured = {}
            handles = []

            def capture(layer_number):
                def hook(_module, _inputs, output):
                    state = output[0] if isinstance(output, (tuple, list)) else output
                    if not torch.is_tensor(state):
                        raise RuntimeError(
                            f"Jina vision block {layer_number} returned unsupported type {type(output)}."
                        )
                    captured[layer_number] = state

                return hook

            try:
                for layer_number in selected_layers:
                    handles.append(blocks[layer_number - 1].register_forward_hook(capture(layer_number)))
                with torch.no_grad():
                    try:
                        vision_model(images, return_all_features=True)
                    except TypeError:
                        vision_model(x=images, return_all_features=True)
            finally:
                for handle in handles:
                    handle.remove()

            missing = [layer for layer in selected_layers if layer not in captured]
            if missing:
                raise RuntimeError(f"Jina vision forward did not execute requested blocks: {missing}.")

            # Intermediate blocks remain in their native spaces and receive
            # independent norms inside V3. V1 has always consumed the tower's
            # final-normalized h24 sequence, so normalize only the actual final
            # EVA block to preserve that path exactly.
            final_norm = getattr(block_owner, "norm", None)
            selected_states = []
            for layer_number in selected_layers:
                state = captured[layer_number]
                if state.dim() != 3:
                    raise RuntimeError(
                        f"Jina vision block {layer_number} must be [B, N, D], got {tuple(state.shape)}."
                    )
                if final_norm is not None and layer_number == len(blocks):
                    state = final_norm(state)
                selected_states.append(state.to(self.dtype))
            return selected_states

        with torch.no_grad():
            hidden_state_layers = None
            if hasattr(self.model, "vision_model"):
                outputs = None
                if return_layer_states:
                    # Prefer full hidden states when the vision tower exposes them.
                    try:
                        outputs = self.model.vision_model(pixel_values=images, output_hidden_states=True, return_dict=True)
                    except TypeError:
                        try:
                            outputs = self.model.vision_model(images, output_hidden_states=True, return_dict=True)
                        except TypeError:
                            outputs = None

                if outputs is None:
                    try:
                        # Jina EVA-02 specific path for unpooled sequence states.
                        outputs = self.model.vision_model(images, return_all_features=True)
                    except TypeError:
                        try:
                            outputs = self.model.vision_model(pixel_values=images, output_hidden_states=True, return_dict=True)
                        except TypeError:
                            try:
                                outputs = self.model.vision_model(images, output_hidden_states=True, return_dict=True)
                            except TypeError:
                                outputs = self.model.vision_model(images)
                            
                # Route the hidden states based on the output type
                if hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                    if return_layer_states and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                        hidden_state_layers = list(outputs.hidden_states)[-int(max_layer_states):]
                elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    hidden_states = outputs["last_hidden_state"]
                    if return_layer_states and outputs.get("hidden_states") is not None:
                        hidden_state_layers = list(outputs["hidden_states"])[-int(max_layer_states):]
                elif isinstance(outputs, dict) and "hidden_states" in outputs:
                    hidden_states = outputs["hidden_states"][-1]
                    if return_layer_states:
                        hidden_state_layers = list(outputs["hidden_states"])[-int(max_layer_states):]
                elif isinstance(outputs, dict) and "image_embeds" in outputs:
                    hidden_states = outputs["image_embeds"]
                elif isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                elif isinstance(outputs, torch.Tensor):
                    # Jina returns a raw torch.Tensor here when return_all_features=True
                    hidden_states = outputs
                else:
                    raise RuntimeError(f"Jina vision_model returned unsupported output type: {type(outputs)}")
            else:
                try:
                    hidden_states = self.model.get_image_features(pixel_values=images)
                except TypeError:
                    hidden_states = self.model.get_image_features(images)

        def normalize_vision_state(state: torch.Tensor) -> torch.Tensor:
            # Jina's EVAVisionTransformer entirely skips the final normalization when `return_all_features=True`.
            # Standard IP-Adapters expect these sequence tokens to be normalized, so we apply it manually here.
            if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "norm"):
                if state.dim() == 3:
                    state = self.model.vision_model.norm(state)
            return state

        if hidden_state_layers is not None:
            normalized_layers = []
            for layer_state in hidden_state_layers:
                if layer_state.dim() == 2:
                    layer_state = layer_state.unsqueeze(1)
                if layer_state.dim() != 3:
                    continue
                normalized_layers.append(normalize_vision_state(layer_state).to(self.dtype))
            if normalized_layers:
                return normalized_layers

        hidden_states = normalize_vision_state(hidden_states)

        # Sanity Check
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            print("Pooled State :( Vision")
        if hidden_states.dim() != 3:
            raise RuntimeError(f"Jina vision hidden states must be [B, N, D], got {tuple(hidden_states.shape)}")

        return hidden_states.to(self.dtype)

    def pool_image_hidden_state(self, final_hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply Jina EVA's native pooling/head/projection to a captured final state.

        ``encode_image_hidden_states(..., selected_layer_indices=...)`` returns
        the final block in the same normalized state used by EVA's ordinary
        forward.  Reusing it here avoids a second vision-tower pass merely to
        obtain Jina's official pooled image feature.
        """

        if final_hidden_state.dim() != 3:
            raise ValueError(
                "final_hidden_state must be [batch, patches, width], "
                f"got {tuple(final_hidden_state.shape)}."
            )
        if not hasattr(self.model, "vision_model"):
            raise RuntimeError("Native Jina image pooling requires the loaded vision_model.")

        vision_model = self.model.vision_model
        block_owner = vision_model
        blocks = getattr(vision_model, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            for module in vision_model.modules():
                candidate = getattr(module, "blocks", None)
                if isinstance(candidate, torch.nn.ModuleList):
                    block_owner = module
                    break

        with torch.no_grad():
            fc_norm = getattr(block_owner, "fc_norm", None)
            if fc_norm is not None:
                pooled = fc_norm(final_hidden_state.mean(dim=1))
            else:
                pooled = final_hidden_state[:, 0]

            head = getattr(block_owner, "head", None)
            if head is not None:
                pooled = head(pooled)
            visual_projection = getattr(self.model, "visual_projection", None)
            if visual_projection is not None:
                pooled = visual_projection(pooled)

        if pooled.dim() != 2:
            raise RuntimeError(
                "Jina's pooled image feature must be [batch, width], "
                f"got {tuple(pooled.shape)}."
            )
        return pooled.to(self.dtype)

    def encode_image_conditioning_states(
        self,
        images: torch.Tensor,
        selected_layer_indices: Sequence[int] = (8, 16, 24),
    ) -> Dict[str, torch.Tensor]:
        """Return selected EVA states and the native pooled image feature."""

        selected_states = self.encode_image_hidden_states(
            images,
            selected_layer_indices=selected_layer_indices,
        )
        if not isinstance(selected_states, (list, tuple)) or not selected_states:
            raise RuntimeError("Selected Jina vision-layer capture returned no states.")
        pooled_state = self.pool_image_hidden_state(selected_states[-1])
        return {
            "jina_image_hidden_states_selected_layers": torch.stack(list(selected_states), dim=1),
            "jina_image_pooled_state": pooled_state,
        }

    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Fallback mean pooling mirroring Jina-embeddings-v3."""
        hidden_states_f32 = hidden_states.to(torch.float32)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states_f32.size()).float()
        
        sum_embeddings = torch.sum(hidden_states_f32 * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        pooled = sum_embeddings / sum_mask
        return pooled.to(self.dtype)

    def prepare_text_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize exactly as the training path does and pad to an SDXL-compatible length."""
        import math

        texts = [str(text) for text in texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # --- Pad sequence to a multiple of 77 for SDXL compatibility ---
        batch_size, seq_len = inputs["input_ids"].shape
        target_len = math.ceil(seq_len / 77) * 77
        pad_len = target_len - seq_len
        
        if pad_len > 0:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id or 0
                
            # Create padding tensors for the batch
            pad_ids = torch.full(
                (batch_size, pad_len), 
                pad_token_id, 
                dtype=inputs["input_ids"].dtype
            )
            pad_mask = torch.zeros(
                (batch_size, pad_len), 
                dtype=inputs["attention_mask"].dtype
            )
            
            # Concatenate along the sequence dimension
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad_ids], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pad_mask], dim=1)
            
            # If the tokenizer outputs token_type_ids, pad those with zeros as well
            if "token_type_ids" in inputs:
                pad_type_ids = torch.zeros(
                    (batch_size, pad_len), 
                    dtype=inputs["token_type_ids"].dtype
                )
                inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], pad_type_ids], dim=1)
        # ---------------------------------------------------------------

        return inputs

    def inspect_text_inputs(self, texts: List[str]) -> List[Dict[str, object]]:
        """Describe truncation and padding for captions without running the Jina model."""
        texts = [str(text) for text in texts]
        raw_inputs = self.tokenizer(texts, padding=False, truncation=False)
        prepared_inputs = self.prepare_text_inputs(texts)
        attention_mask = prepared_inputs["attention_mask"]
        padded_token_count = int(prepared_inputs["input_ids"].shape[1])

        summaries = []
        for index, text in enumerate(texts):
            raw_token_count = len(raw_inputs["input_ids"][index])
            retained_token_count = int(attention_mask[index].sum().item())
            summaries.append(
                {
                    "text": text,
                    "character_count": len(text),
                    "raw_token_count": raw_token_count,
                    "retained_token_count": retained_token_count,
                    "padded_token_count": padded_token_count,
                    "max_length": self.max_length,
                    "truncated": raw_token_count > self.max_length,
                }
            )
        return summaries

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.prepare_text_inputs(texts)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.set_grad_enabled(self.custom_train_jina):
            self.hidden_states_cache = None
            if self.hidden_state_layer_indices is not None:
                self.hidden_state_layers_cache = [None] * len(self.hidden_state_layer_indices)
            elif self.num_hidden_state_layers > 1:
                self.hidden_state_layers_cache = [None] * self.num_hidden_state_layers
            
            hidden_state_layers = None
            # Forward pass: Standard extraction that triggers our hooks.
            if hasattr(self.model, 'get_text_features'):
                pooled_state = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            else:
                out = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                if hasattr(out, 'text_embeds'):
                    pooled_state = out.text_embeds
                elif hasattr(out, 'pooler_output'):
                    pooled_state = out.pooler_output
                elif isinstance(out, tuple):
                    pooled_state = out[1] if len(out) > 1 else out[0]
                else:
                    pooled_state = out
            
            # Ensure the hook successfully captured the sequence states
            if self.hidden_states_cache is None:
                raise RuntimeError("Forward hook did not capture hidden states. The encoder module was bypassed.")

            hidden_states = restore_text_sequence_shape(
                self.hidden_states_cache,
                inputs["attention_mask"],
                "Jina final text state",
            )
            hidden_states = hidden_states.to(self.dtype)
            final_mean_pooled_state = self.mean_pooling(
                hidden_states,
                inputs["attention_mask"],
            )
            # Preserve Jina's official get_text_features result. Redesigned V3
            # intentionally uses this pre-retrieval-normalization representation.
            if not isinstance(pooled_state, torch.Tensor):
                pooled_state = final_mean_pooled_state
            pooled_state = pooled_state.to(self.dtype)

            if (
                hidden_state_layers is None
                and (self.hidden_state_layer_indices is not None or self.num_hidden_state_layers > 1)
            ):
                missing_layers = [
                    index
                    for index, state in enumerate(self.hidden_state_layers_cache)
                    if state is None
                ]
                if missing_layers:
                    raise RuntimeError(
                        "Forward hooks did not capture all requested Jina text layers. "
                        f"Missing cache indices: {missing_layers}."
                    )
                layer_numbers = (
                    self.hidden_state_layer_indices
                    if self.hidden_state_layer_indices is not None
                    else tuple(
                        range(
                            len(self.layer_list) - self.num_hidden_state_layers + 1,
                            len(self.layer_list) + 1,
                        )
                    )
                )
                hidden_state_layers = assemble_selected_text_states(
                    self.hidden_state_layers_cache,
                    hidden_states,
                    inputs["attention_mask"],
                    layer_numbers,
                    len(self.layer_list),
                )
            if hidden_state_layers is not None:
                hidden_state_layers = hidden_state_layers.to(self.dtype)
            
        outputs = {
            "jina_hidden_states": hidden_states,
            "jina_mean_pooled_state": pooled_state,
            "jina_final_mean_pooled_state": final_mean_pooled_state,
            "attention_mask": inputs["attention_mask"]
        }
        if self.hidden_state_layer_indices is not None and hidden_state_layers is not None:
            outputs["jina_hidden_states_selected_layers"] = hidden_state_layers
        elif hidden_state_layers is not None:
            outputs["jina_hidden_states_last4"] = hidden_state_layers
        return outputs


def process_caption_directory(
    model_path: str,
    caption_dir: str,
    output_dir: str,
    batch_size: int = 1,
    num_workers_input: int = 0,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
    skip_existing: bool = True,
    num_hidden_state_layers: int = 1,
    hidden_state_layer_indices: Optional[Tuple[int, ...]] = None,
) -> Dict[str, int]:
    """
    Process all text files in a directory and save their hidden states and attention masks.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    caption_files = sorted(Path(caption_dir).glob("*.txt"))
    if len(caption_files) == 0:
        print(f"No .txt files found in {caption_dir}")
        return {"total": 0, "generated": 0, "skipped": 0}

    files_to_process = []
    skipped = 0
    for caption_file in caption_files:
        save_path = output_dir_path / f"{caption_file.stem}.safetensors"
        if skip_existing and save_path.exists():
            skipped += 1
            continue
        files_to_process.append(caption_file)

    if len(files_to_process) == 0:
        print(f"All {len(caption_files)} caption files already have cached Jina states in {output_dir}")
        return {"total": len(caption_files), "generated": 0, "skipped": skipped}

    # Initialize the extractor only when work is actually needed.
    jina_extractor = JinaStates(
        model_id=model_path,
        device=device,
        dtype=dtype,
        max_length=max_length,
        num_hidden_state_layers=num_hidden_state_layers,
        hidden_state_layer_indices=hidden_state_layer_indices,
    )

    # Setup dataset and dataloader
    dataset = TextDataset(caption_dir, files=files_to_process)
    num_workers_to_use = 4 if torch.cuda.is_available() else 0
    if num_workers_input is not None:
        num_workers_to_use = num_workers_input

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=num_workers_to_use,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(
        f"Processing {len(dataset)} caption files into Jina caches across {len(dataloader)} batches "
        f"({skipped} skipped)"
    )

    generated = 0
    for batch_files, batch_texts in tqdm(dataloader, desc="Caching Jina embeddings"):
        # Get tensors
        outputs = jina_extractor(batch_texts)
        
        # Save each item in the batch individually
        for i, file_path in enumerate(batch_files):
            stem = Path(file_path).stem
            save_path = output_dir_path / f"{stem}.safetensors"
            
            # We slice [i:i+1] instead of [i] to maintain the (1, seq_len, dim) 
            # batch dimension that the AdapterDataset will expect.
            out_dict = {
                "jina_hidden_states": outputs["jina_hidden_states"][i:i+1].cpu().contiguous(),
                "jina_mean_pooled_state": outputs["jina_mean_pooled_state"][i:i+1].cpu().contiguous(),
                "jina_final_mean_pooled_state": outputs["jina_final_mean_pooled_state"][i:i+1].cpu().contiguous(),
                "attention_mask": outputs["attention_mask"][i:i+1].cpu().contiguous()
            }
            if "jina_hidden_states_last4" in outputs:
                out_dict["jina_hidden_states_last4"] = (
                    outputs["jina_hidden_states_last4"][i:i+1].cpu().contiguous()
                )
            if "jina_hidden_states_selected_layers" in outputs:
                out_dict["jina_hidden_states_selected_layers"] = (
                    outputs["jina_hidden_states_selected_layers"][i:i+1].cpu().contiguous()
                )
            
            save_file(out_dict, save_path)
            generated += 1

    print(
        f"Done! Jina cache summary for {caption_dir}: {generated} generated, {skipped} skipped, "
        f"{len(caption_files)} total"
    )
    return {"total": len(caption_files), "generated": generated, "skipped": skipped}


def generate_caption_states(
    model_path: str,
    caption_dir: str,
    batch_size: int = 1,
    num_workers_input: int = 0,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
    num_hidden_state_layers: int = 1,
    hidden_state_layer_indices: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Process all text files in a directory and return their Jina states in memory.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    caption_files = sorted(Path(caption_dir).glob("*.txt"))
    if len(caption_files) == 0:
        print(f"No .txt files found in {caption_dir}")
        return {}

    jina_extractor = JinaStates(
        model_id=model_path,
        device=device,
        dtype=dtype,
        max_length=max_length,
        num_hidden_state_layers=num_hidden_state_layers,
        hidden_state_layer_indices=hidden_state_layer_indices,
    )

    dataset = TextDataset(caption_dir, files=caption_files)
    num_workers_to_use = 4 if torch.cuda.is_available() else 0
    if num_workers_input is not None:
        num_workers_to_use = num_workers_input

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=num_workers_to_use,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(
        f"Processing {len(dataset)} caption files into in-memory Jina states across {len(dataloader)} batches"
    )

    outputs_by_stem: Dict[str, Dict[str, torch.Tensor]] = {}
    for batch_files, batch_texts in tqdm(dataloader, desc="Generating Jina embeddings"):
        outputs = jina_extractor(batch_texts)

        for i, file_path in enumerate(batch_files):
            stem = Path(file_path).stem
            outputs_by_stem[stem] = {
                "jina_hidden_states": outputs["jina_hidden_states"][i:i+1].cpu().contiguous(),
                "jina_mean_pooled_state": outputs["jina_mean_pooled_state"][i:i+1].cpu().contiguous(),
                "jina_final_mean_pooled_state": outputs["jina_final_mean_pooled_state"][i:i+1].cpu().contiguous(),
                "attention_mask": outputs["attention_mask"][i:i+1].cpu().contiguous(),
            }
            if "jina_hidden_states_last4" in outputs:
                outputs_by_stem[stem]["jina_hidden_states_last4"] = (
                    outputs["jina_hidden_states_last4"][i:i+1].cpu().contiguous()
                )
            if "jina_hidden_states_selected_layers" in outputs:
                outputs_by_stem[stem]["jina_hidden_states_selected_layers"] = (
                    outputs["jina_hidden_states_selected_layers"][i:i+1].cpu().contiguous()
                )

    print(f"Done! Generated in-memory Jina states for {len(outputs_by_stem)} captions from {caption_dir}")
    return outputs_by_stem


if __name__ == "__main__":
    """
    Example usage demonstrating how to process a directory of text files 
    and extract their states for the JinaToSDXLAdapter.
    """
    # Use jinaai/jina-clip-v2 directly to pull from HuggingFace, or point to a local directory
    model_id = "jinaai/jina-clip-v2"  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Info: Device is {device}")
    batch_size = 4  # Can comfortably increase this since vision_model is deleted
    max_length = 512
    num_workers_input = None
    caption_dir = input("Caption Directory: ")
    output_path = input("Output Directory: ") # Maps to the 'input_states' folder in your trainer

    # Ensure input directory exists for the script to run out of the box
    os.makedirs(caption_dir, exist_ok=True)
    
    process_caption_directory(
        model_path=model_id, 
        caption_dir=caption_dir, 
        output_dir=output_path, 
        batch_size=batch_size,
        num_workers_input = num_workers_input, 
        device=device,
        max_length=max_length
    )

