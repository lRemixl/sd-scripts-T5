import argparse
import os
import glob
from tqdm import tqdm
import torch
from safetensors.torch import save_file, load_file
from transformers import Qwen3VLForConditionalGeneration, T5GemmaEncoderModel, AutoTokenizer  # type: ignore

# Import your initialization logic
from library.device_utils import init_ipex

def setup_args():
    parser = argparse.ArgumentParser(description="Pre-cache LLM outputs for SDXL Adapter training")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing .txt/.caption files")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the LLM model")
    parser.add_argument("--use_qwen3VL_as_text_encoder", action="store_true", help="Use Qwen instead of T5 Gemma")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for processing ")
    return parser.parse_args()

def load_models(args):
    """
    Loads the LLM and Tokenizer based on arguments.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_path,
        trust_remote_code=True
    )

    if args.use_qwen3VL_as_text_encoder:
        llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"Loaded Qwen-3-VL from: {args.llm_model_path}")
    else:
        llm_model = T5GemmaEncoderModel.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
        )
        print(f"Loaded T5Gemma from: {args.llm_model_path}")

    llm_model.requires_grad_(False)
    llm_model.eval()
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model.to(device=device)
    
    return tokenizer, llm_model, device

def process_dataset(args):
    tokenizer, llm_model, device = load_models(args)
    
    # Find text files
    extensions = ['*.txt', '*.caption']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(args.train_data_dir, '**', ext), recursive=True))
    
    # Remove duplicates and sort
    all_files = sorted(list(set(files)))

    files = []
    skipped_count = 0
    for f_path in all_files:
        base, _ = os.path.splitext(f_path)
        expected_output_path = f"{base}.llm_embed.safetensors"
        
        if os.path.exists(expected_output_path):
            skipped_count += 1
        else:
            files.append(f_path)
    
    print(f"Found {len(files)} files to process, Found {skipped_count} already pre-computed. Starting pre-caching LLM outputs...")

    if not files:
        print("No new files to process.")
        del llm_model
        del tokenizer
        torch.cuda.empty_cache()
        return

    # Process in batches
    for i in tqdm(range(0, len(files), args.train_batch_size)):
        batch_files = files[i : i + args.train_batch_size]
        captions = []
        valid_files = []

        # Read captions
        for f_path in batch_files:
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    captions.append(content + tokenizer.eos_token) 
                    valid_files.append(f_path)
            except Exception as e:
                print(f"Error reading {f_path}: {e}")
                continue
        
        if not captions:
            continue

        # Tokenize
        tokenized_input = tokenizer(
            captions,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )

        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device)

        # Forward Pass 
        with torch.no_grad():
            if args.use_qwen3VL_as_text_encoder:
                outputs = llm_model.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            else:
                outputs = llm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            
            # Extract Last Hidden State
            hidden_states = outputs.last_hidden_state.to(torch.float32)

        # Save files
        for idx, f_path in enumerate(valid_files):
            base, _ = os.path.splitext(f_path)
            output_path = f"{base}.llm_embed.safetensors"
            
            # Extract single item from batch
            single_hidden = hidden_states[idx].cpu().clone()
            single_mask = attention_mask[idx].cpu().clone()

            # Save as safetensors
            tensors_to_save = {
                "last_hidden_state": single_hidden,
                "attention_mask": single_mask
            }
            
            save_file(tensors_to_save, output_path)
            
    del llm_model
    del tokenizer
    torch.cuda.empty_cache()

    print("Caching complete.")

if __name__ == "__main__":
    args = setup_args()
    process_dataset(args)