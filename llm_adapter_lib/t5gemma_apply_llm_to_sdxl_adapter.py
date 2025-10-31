import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")

class t5gemmaApplyLLMToSDXLAdapter:
    """
    ComfyUI node that applies the LLM→SDXL adapter to hidden states to produce SDXL conditioning.
    Adds pooled_output and optional SDXL size/crop params to the metadata dict.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_hidden_states": ("LLM_HIDDEN_STATES",),
                "llm_attention_mask": ("LLM_ATTENTION_MASK",),
                "llm_adapter": ("LLM_ADAPTER",),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "llm_sdxl"

    def apply(self, llm_hidden_states, llm_attention_mask, llm_adapter,
              width=None, height=None, target_width=None, target_height=None, crop_w=None, crop_h=None):
        try:
            with torch.no_grad():
                prompt_embeds, pooled_output = llm_adapter(llm_hidden_states, attention_mask=llm_attention_mask)

            prompt_embeds = prompt_embeds.cpu().contiguous()
            pooled_output = pooled_output.cpu().contiguous()

            meta = {"pooled_output": pooled_output}
            if width is not None and height is not None:
                meta.update({"width": int(width), "height": int(height)})
            if target_width is not None and target_height is not None:
                meta.update({"target_width": int(target_width), "target_height": int(target_height)})
            if crop_w is not None and crop_h is not None:
                meta.update({"crop_w": int(crop_w), "crop_h": int(crop_h)})

            conditioning = [[prompt_embeds, meta]]
            return (conditioning,)
        except Exception as e:
            logger.error("Failed to apply adapter: %s", str(e))
            raise

NODE_CLASS_MAPPINGS = {
    "t5gemmaApplyLLMToSDXLAdapter": t5gemmaApplyLLMToSDXLAdapter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "t5gemmaApplyLLMToSDXLAdapter": "Apply T5Gemma LLM to Adapter"
}
