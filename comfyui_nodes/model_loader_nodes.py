"""
Model Loader Nodes for MFLUX
============================
Nodes for loading various MFLUX models with quantization support.
"""

import torch
from typing import Tuple, Dict, Any


class MfluxFlux1Loader:
    """Load FLUX.1 models (dev, schnell, dev-fill, dev-redux, dev-depth, dev-kontext, krea-dev)"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_variant": (
                    [
                        "dev",
                        "schnell",
                        "dev-fill",
                        "dev-fill-catvton",
                        "dev-redux",
                        "dev-depth",
                        "dev-kontext",
                        "krea-dev",
                    ],
                    {"default": "schnell"},
                ),
                "quantize": (
                    ["none", "4", "8"],
                    {"default": "8"},
                ),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_variant: str,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.flux import Flux1
        from mflux.models.common.config import ModelConfig

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model - if custom path provided, use it; otherwise use variant config
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = Flux1(
                model_config=ModelConfig.from_name(model_name=model_variant),
                quantize=quant,
                model_path=model_path,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )
        else:
            # Use default config which will auto-download from HuggingFace
            model = Flux1(
                model_config=ModelConfig.from_name(model_name=model_variant),
                quantize=quant,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )

        return (
            {
                "model": model,
                "model_type": "flux1",
                "variant": model_variant,
            },
        )


class MfluxFlux2KleinLoader:
    """Load FLUX.2 Klein models (4B, 9B)"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_size": (["4B", "9B"], {"default": "4B"}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_size: str,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.flux2 import Flux2Klein
        from mflux.models.common.config import ModelConfig

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Determine model name for config
        model_name = f"flux2-klein-{model_size.lower()}"

        # Load model - if custom path provided, use it; otherwise use default config
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = Flux2Klein(
                model_config=ModelConfig.from_name(model_name=model_name),
                quantize=quant,
                model_path=model_path,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )
        else:
            # Use default config which will auto-download from HuggingFace
            model = Flux2Klein(
                model_config=ModelConfig.from_name(model_name=model_name),
                quantize=quant,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )

        return (
            {
                "model": model,
                "model_type": "flux2_klein",
                "size": model_size,
            },
        )


class MfluxZImageTurboLoader:
    """Load Z-Image Turbo model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.z_image import ZImageTurbo

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model - will auto-download from HuggingFace if not cached
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = ZImageTurbo(
                quantize=quant,
                model_path=model_path,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )
        else:
            # Use default config (Tongyi-MAI/Z-Image-Turbo)
            model = ZImageTurbo(
                quantize=quant,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )

        return (
            {
                "model": model,
                "model_type": "z_image_turbo",
            },
        )


class MfluxFiboLoader:
    """Load FIBO model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.fibo import Fibo

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model - will auto-download from HuggingFace if not cached
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = Fibo(
                quantize=quant,
                model_path=model_path,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )
        else:
            # Use default config (briaai/FIBO)
            model = Fibo(
                quantize=quant,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )

        return (
            {
                "model": model,
                "model_type": "fibo",
            },
        )


class MfluxQwenImageLoader:
    """Load Qwen Image models (base, edit)"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_variant": (["qwen-image", "qwen-image-edit"], {"default": "qwen-image"}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_variant: str,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.qwen_image import QwenImage
        from mflux.models.common.config import ModelConfig

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model - if custom path provided, use it; otherwise use variant config
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = QwenImage(
                model_config=ModelConfig.from_name(model_name=model_variant),
                quantize=quant,
                model_path=model_path,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )
        else:
            # Use default config which will auto-download from HuggingFace
            model = QwenImage(
                model_config=ModelConfig.from_name(model_name=model_variant),
                quantize=quant,
                lora_paths=lora_path_list,
                lora_scales=lora_scale_list,
            )

        return (
            {
                "model": model,
                "model_type": "qwen_image",
                "variant": model_variant,
            },
        )


class MfluxSeedVR2Loader:
    """Load SeedVR2 upscaling model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        quantize: str,
        model_path: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.seedvr2 import SeedVR2

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model - will auto-download from HuggingFace if not cached
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = SeedVR2(
                quantize=quant,
                model_path=model_path,
            )
        else:
            # Use default config (numz/SeedVR2_comfyUI)
            model = SeedVR2(
                quantize=quant,
            )

        return (
            {
                "model": model,
                "model_type": "seedvr2",
            },
        )


class MfluxDepthProLoader:
    """Load DepthPro model for depth estimation"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_path: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.depth_pro import DepthPro

        # Load model - will auto-download from HuggingFace if not cached
        if model_path:
            # Custom model path (local or HuggingFace repo)
            model = DepthPro(
                model_path=model_path,
            )
        else:
            # Use default config (apple/ml-depth-pro)
            model = DepthPro()

        return (
            {
                "model": model,
                "model_type": "depth_pro",
            },
        )
