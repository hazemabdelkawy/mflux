"""
Fill/Inpainting Nodes for MFLUX
================================
Nodes for fill and inpainting operations using FLUX.1 Fill.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import tempfile
from pathlib import Path


class MfluxFill:
    """Fill/inpaint regions in images using FLUX.1 Fill"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fill"
    CATEGORY = "MFLUX/fill"

    def fill(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        quantize: str,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        from mflux.models.flux import Flux1
        from mflux.models.common.config import ModelConfig

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model - auto-downloads from HuggingFace if not cached
        mflux_model = Flux1(
            model_config=ModelConfig.from_name(model_name="dev-fill"),
            quantize=quant,
        )

        # Convert ComfyUI image tensors to PIL Images
        def tensor_to_pil(tensor):
            image_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(image_np)

        image_pil = tensor_to_pil(image)
        mask_pil = tensor_to_pil(mask)

        # Convert mask to grayscale if needed
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        # Save to temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_image,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask,
        ):
            image_path = Path(tmp_image.name)
            mask_path = Path(tmp_mask.name)

            image_pil.save(image_path)
            mask_pil.save(mask_path)

        try:
            # Fill/inpaint image
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=image_path,
                masked_image_path=mask_path,
            )
        finally:
            image_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)
