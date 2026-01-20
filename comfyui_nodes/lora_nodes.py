"""
LoRA Nodes for MFLUX
=====================
Nodes for managing LoRA loading and configuration.
"""

import torch
from typing import Tuple, Dict, Any


class MfluxLoRALoader:
    """Load and configure LoRA models for use with MFLUX models"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
            },
            "optional": {
                "lora_path_1": ("STRING", {"default": "", "multiline": False}),
                "lora_scale_1": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "lora_path_2": ("STRING", {"default": "", "multiline": False}),
                "lora_scale_2": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "lora_path_3": ("STRING", {"default": "", "multiline": False}),
                "lora_scale_3": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "lora_path_4": ("STRING", {"default": "", "multiline": False}),
                "lora_scale_4": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_loras"
    CATEGORY = "MFLUX/lora"

    def load_loras(
        self,
        model: Dict[str, Any],
        lora_path_1: str = "",
        lora_scale_1: float = 1.0,
        lora_path_2: str = "",
        lora_scale_2: float = 1.0,
        lora_path_3: str = "",
        lora_scale_3: float = 1.0,
        lora_path_4: str = "",
        lora_scale_4: float = 1.0,
    ) -> Tuple[Dict[str, Any]]:
        # Collect LoRA paths and scales
        lora_paths = []
        lora_scales = []

        if lora_path_1:
            lora_paths.append(lora_path_1)
            lora_scales.append(lora_scale_1)
        if lora_path_2:
            lora_paths.append(lora_path_2)
            lora_scales.append(lora_scale_2)
        if lora_path_3:
            lora_paths.append(lora_path_3)
            lora_scales.append(lora_scale_3)
        if lora_path_4:
            lora_paths.append(lora_path_4)
            lora_scales.append(lora_scale_4)

        # Get the model instance
        mflux_model = model["model"]

        # Apply LoRAs if any are specified
        if lora_paths:
            mflux_model.load_loras(
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )

        # Return the model with updated LoRAs
        return (model,)
