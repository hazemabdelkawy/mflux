"""
Utility Nodes for MFLUX
========================
Utility nodes for configuration, model saving, and other helper functions.
"""

import torch
from typing import Tuple, Dict, Any
from pathlib import Path


class MfluxSaveQuantized:
    """Save a quantized model to disk for faster loading"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "save_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_model"
    CATEGORY = "MFLUX/utilities"

    def save_model(
        self,
        model: Dict[str, Any],
        save_path: str,
    ) -> Dict[str, Any]:
        # Get the model instance
        mflux_model = model["model"]

        # Save the quantized model
        save_path_obj = Path(save_path)
        mflux_model.save_model(save_path_obj)

        return {
            "ui": {"text": [f"Model saved to: {save_path}"]},
        }


class MfluxConfigNode:
    """Configure global MFLUX settings and options"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "low_ram_mode": ("BOOLEAN", {"default": False}),
                "battery_saver": ("BOOLEAN", {"default": False}),
                "battery_stop_threshold": ("INT", {"default": 20, "min": 0, "max": 100}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "MFLUX/utilities"

    def configure(
        self,
        low_ram_mode: bool,
        battery_saver: bool,
        battery_stop_threshold: int,
        cache_dir: str = "",
    ) -> Tuple[Dict[str, Any]]:
        import os

        # Set environment variables if specified
        if cache_dir:
            os.environ["MFLUX_CACHE_DIR"] = cache_dir

        # Create config dictionary
        config = {
            "low_ram": low_ram_mode,
            "battery_saver": battery_saver,
            "battery_stop_threshold": battery_stop_threshold,
            "cache_dir": cache_dir,
        }

        return (config,)
