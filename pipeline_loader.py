"""
pipeline_loader.py
------------------
Builds and returns the main generation pipeline:
  - Stable Diffusion XL Inpainting (from the "diffusers" library)
  - IP-Adapter FaceID XL on top of it (injects face-identity conditioning)

DDIM scheduler is used because it converges well at 30 steps without the
extra overhead of more modern samplers.
"""

import torch
from diffusers import StableDiffusionXLInpaintPipeline, DDIMScheduler
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL


def load_pipeline(base_model: str, ip_ckpt: str, device: str) -> IPAdapterFaceIDXL:
    """
    Load the SDXL inpainting backbone and wrap it with IP-Adapter FaceID.

    The returned object (`ip_model`) exposes a `.generate()` method that
    accepts the same arguments as the underlying diffusers pipeline, plus
    `faceid_embeds` and `scale` for identity conditioning.

    Args:
        base_model: HuggingFace repo ID of the SDXL model to use as the UNet.
                    The inpainting pipeline itself is always loaded from
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1".
        ip_ckpt:    Path to the IP-Adapter FaceID checkpoint (.bin file).
        device:     "cuda" or "cpu".

    Returns:
        IPAdapterFaceIDXL instance ready for inference.
    """

    # DDIM is deterministic and produces clean results in 30 steps
    print("  Setting up DDIM scheduler...")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # Load the SDXL inpainting UNet – this is the model that does the actual
    # pixel generation; float16 halves VRAM usage with negligible quality loss
    print("  Loading SDXL inpainting model (this takes a moment)...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,   # skip invisible watermark – not needed here
    )
    pipe.to(device)

    # Wrap the pipeline with IP-Adapter FaceID so we can pass a face embedding
    # alongside the text prompt to guide identity-preserving generation
    print("  Attaching IP-Adapter FaceID...")
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    return ip_model
