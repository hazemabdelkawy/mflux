"""
Upscaling Nodes for MFLUX
==========================
Nodes for upscaling images using SeedVR2.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import tempfile
from pathlib import Path


class MfluxSeedVR2Upscale:
    """Upscale images using SeedVR2 model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "upscale_factor": (["2x", "3x", "4x"], {"default": "2x"}),
            },
            "optional": {
                "target_resolution": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "MFLUX/upscaling"

    def upscale(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        upscale_factor: str,
        target_resolution: int = 0,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        # Get the actual model instance
        mflux_model = model["model"]

        # Convert ComfyUI image tensor to PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Determine resolution parameter
            if target_resolution > 0:
                resolution = target_resolution
            else:
                resolution = upscale_factor

            # Upscale image
            result_image = mflux_model.upscale_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=tmp_path,
                resolution=resolution,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)
