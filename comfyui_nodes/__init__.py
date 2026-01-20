"""
MFLUX ComfyUI Nodes
===================
Custom nodes for ComfyUI that provide access to all MFLUX features.

Supported Models:
- FLUX.1 (dev, schnell)
- FLUX.2 Klein (4B, 9B)
- Z-Image Turbo
- FIBO
- Qwen Image
- SeedVR2

Features:
- Text-to-image generation
- Image-to-image generation
- Image editing (FLUX.2, Qwen, FIBO)
- ControlNet (Canny, Upscaler)
- Redux (style reference)
- Depth conditioning
- Fill/Inpainting
- In-context editing (Kontext, CatVTON)
- Upscaling (SeedVR2, ControlNet)
- LoRA support
- Quantization
"""

from .model_loader_nodes import (
    MfluxFlux1Loader,
    MfluxFlux2KleinLoader,
    MfluxZImageTurboLoader,
    MfluxFiboLoader,
    MfluxQwenImageLoader,
    MfluxSeedVR2Loader,
    MfluxDepthProLoader,
)

from .generation_nodes import (
    MfluxTextToImage,
    MfluxImageToImage,
)

from .editing_nodes import (
    MfluxFlux2Edit,
    MfluxQwenEdit,
    MfluxKontextEdit,
    MfluxCatVTONEdit,
)

from .controlnet_nodes import (
    MfluxControlNetCanny,
    MfluxControlNetUpscaler,
)

from .depth_redux_nodes import (
    MfluxDepthConditioning,
    MfluxRedux,
    MfluxDepthExtraction,
)

from .upscaling_nodes import (
    MfluxSeedVR2Upscale,
)

from .fill_nodes import (
    MfluxFill,
)

from .lora_nodes import (
    MfluxLoRALoader,
)

from .utility_nodes import (
    MfluxSaveQuantized,
    MfluxConfigNode,
)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Model Loaders
    "MfluxFlux1Loader": MfluxFlux1Loader,
    "MfluxFlux2KleinLoader": MfluxFlux2KleinLoader,
    "MfluxZImageTurboLoader": MfluxZImageTurboLoader,
    "MfluxFiboLoader": MfluxFiboLoader,
    "MfluxQwenImageLoader": MfluxQwenImageLoader,
    "MfluxSeedVR2Loader": MfluxSeedVR2Loader,
    "MfluxDepthProLoader": MfluxDepthProLoader,

    # Generation
    "MfluxTextToImage": MfluxTextToImage,
    "MfluxImageToImage": MfluxImageToImage,

    # Editing
    "MfluxFlux2Edit": MfluxFlux2Edit,
    "MfluxQwenEdit": MfluxQwenEdit,
    "MfluxKontextEdit": MfluxKontextEdit,
    "MfluxCatVTONEdit": MfluxCatVTONEdit,

    # ControlNet
    "MfluxControlNetCanny": MfluxControlNetCanny,
    "MfluxControlNetUpscaler": MfluxControlNetUpscaler,

    # Depth & Redux
    "MfluxDepthConditioning": MfluxDepthConditioning,
    "MfluxRedux": MfluxRedux,
    "MfluxDepthExtraction": MfluxDepthExtraction,

    # Upscaling
    "MfluxSeedVR2Upscale": MfluxSeedVR2Upscale,

    # Fill/Inpainting
    "MfluxFill": MfluxFill,

    # LoRA
    "MfluxLoRALoader": MfluxLoRALoader,

    # Utilities
    "MfluxSaveQuantized": MfluxSaveQuantized,
    "MfluxConfigNode": MfluxConfigNode,
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Model Loaders
    "MfluxFlux1Loader": "MFLUX FLUX.1 Loader",
    "MfluxFlux2KleinLoader": "MFLUX FLUX.2 Klein Loader",
    "MfluxZImageTurboLoader": "MFLUX Z-Image Turbo Loader",
    "MfluxFiboLoader": "MFLUX FIBO Loader",
    "MfluxQwenImageLoader": "MFLUX Qwen Image Loader",
    "MfluxSeedVR2Loader": "MFLUX SeedVR2 Loader",
    "MfluxDepthProLoader": "MFLUX DepthPro Loader",

    # Generation
    "MfluxTextToImage": "MFLUX Text to Image",
    "MfluxImageToImage": "MFLUX Image to Image",

    # Editing
    "MfluxFlux2Edit": "MFLUX FLUX.2 Edit",
    "MfluxQwenEdit": "MFLUX Qwen Edit",
    "MfluxKontextEdit": "MFLUX Kontext Edit",
    "MfluxCatVTONEdit": "MFLUX CatVTON Edit",

    # ControlNet
    "MfluxControlNetCanny": "MFLUX ControlNet Canny",
    "MfluxControlNetUpscaler": "MFLUX ControlNet Upscaler",

    # Depth & Redux
    "MfluxDepthConditioning": "MFLUX Depth Conditioning",
    "MfluxRedux": "MFLUX Redux (Style Reference)",
    "MfluxDepthExtraction": "MFLUX Depth Extraction",

    # Upscaling
    "MfluxSeedVR2Upscale": "MFLUX SeedVR2 Upscale",

    # Fill/Inpainting
    "MfluxFill": "MFLUX Fill/Inpaint",

    # LoRA
    "MfluxLoRALoader": "MFLUX LoRA Loader",

    # Utilities
    "MfluxSaveQuantized": "MFLUX Save Quantized Model",
    "MfluxConfigNode": "MFLUX Config",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
