"""
generation.py
-------------
The three-step inpainting sequence that transforms an input portrait into a
fully synthetic scene while keeping the subject's face identity intact.

Step 1 – Face:       Re-generate just the face region using IP-Adapter
                     (high identity scale, low strength so the face stays
                     recognisable but gains model-quality sharpness).

Step 2 – Clothing:   Inpaint the body region (face excluded) with a
                     clothing prompt. IP-Adapter scale is 0 here because we
                     don't want facial features leaking into the shirt.

         Blend:      Paste the Step-1 face back onto the clothing result
                     with a Gaussian-feathered mask so the boundary is soft.

Step 3 – Background: Replace everything outside the person mask with a
                     scene prompt. We deliberately scramble the original
                     background with random noise first so the model has
                     total creative freedom rather than following the
                     original scene composition.

All three steps use the same `ip_model.generate()` call with different
prompts, masks, and conditioning strengths.
"""

import cv2
import numpy as np
from PIL import Image
import torch

from config import (
    NEGATIVE_PROMPT, SEED, NUM_SAMPLES,
    WIDTH, HEIGHT, INFERENCE_STEPS,
)


def generate_scene(
    ip_model,
    image: Image.Image,
    faceid_embeds: torch.Tensor,
    face_bbox: tuple,
    bg_mask: np.ndarray,
    person_mask: np.ndarray,
    prompt_person: str,
    prompt_clothing: str,
    prompt_background: str,
) -> Image.Image:
    """
    Run the full 3-step generation and return the composited output image.

    Args:
        ip_model:          Loaded IPAdapterFaceIDXL pipeline.
        image:             Preprocessed input image (1024×1024 PIL).
        faceid_embeds:     ArcFace embedding tensor (1, 512).
        face_bbox:         (x1, y1, x2, y2) padded face bounding box.
        bg_mask:           Boolean mask – True where the person is.
        person_mask:       uint8 mask – 255 where the background is.
        prompt_person:     Text prompt describing the full-body person.
        prompt_clothing:   Text prompt describing the clothing style.
        prompt_background: Text prompt describing the scene/environment.

    Returns:
        Final composited PIL image.
    """

    x1, y1, x2, y2 = face_bbox


    # Step 1 – Face generation
    # Crop the face out of the input, upscale it to model resolution,
    # and let IP-Adapter refine it.  Low strength (0.60) means we're only
    # adding detail on top of the real face rather than replacing it wholesale.
    print("  [Step 1] Generating face...")

    image_np   = np.array(image)
    face_crop  = Image.fromarray(image_np[y1:y2, x1:x2])
    face_crop_resized = face_crop.resize((WIDTH, HEIGHT))

    # An all-white mask tells the inpainting pipeline it may touch every pixel
    mask_face = Image.new("L", (WIDTH, HEIGHT), 255)

    face_images = ip_model.generate(
        prompt=prompt_person,
        negative_prompt=NEGATIVE_PROMPT,
        faceid_embeds=faceid_embeds,
        scale=0.95,               # high identity scale – keep the person's look
        num_samples=NUM_SAMPLES,
        seed=SEED,
        guidance_scale=0.0,       # classifier-free guidance off – let the face speak
        num_inference_steps=INFERENCE_STEPS,
        image=face_crop_resized,
        mask_image=mask_face,
        strength=0.40,            # mild strength – refine, don't replace
        width=WIDTH,
        height=HEIGHT,
    )
    generated_face = face_images[0]   # 1024×1024 refined face


    # Step 2 – Clothing generation

    # Build a mask that covers the body but leaves the face hole untouched
    # so the clothing model never overwrites what Step 1 just produced.
    print("  [Step 2] Generating clothing...")

    person_mask_for_clothing = bg_mask.copy()
    person_mask_for_clothing[y1:y2, x1:x2] = 0  # punch out the face area
    mask_clothing = Image.fromarray((person_mask_for_clothing * 255).astype(np.uint8))

    clothing_images = ip_model.generate(
        prompt=prompt_clothing,
        negative_prompt=NEGATIVE_PROMPT,
        faceid_embeds=faceid_embeds,
        scale=0.0,                # no face conditioning – pure clothing prompt
        num_samples=NUM_SAMPLES,
        seed=SEED,
        guidance_scale=8.5,
        num_inference_steps=INFERENCE_STEPS,
        image=image,
        mask_image=mask_clothing,
        strength=0.99,            # fully replace the body region
        width=WIDTH,
        height=HEIGHT,
    )
    clothing_result = clothing_images[0]

    # Blend: paste the refined face back onto the clothing result
    
    # A hard paste would leave a visible box edge, so we build a soft Gaussian
    # feather mask and alpha-blend the two images in float32.
    print("  [Blend] Compositing face onto clothing...")

    generated_face_resized = generated_face.resize((x2 - x1, y2 - y1))

    # Build a float [0,1] alpha map with soft edges around the face region
    feather = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    feather[y1:y2, x1:x2] = 1.0
    feather = cv2.GaussianBlur(feather, (51, 51), 0)
    feather_3c = np.stack([feather] * 3, axis=-1)

    clothing_np   = np.array(clothing_result).astype(np.float32)
    blended_np    = clothing_np.copy()
    blended_np[y1:y2, x1:x2] = np.array(generated_face_resized).astype(np.float32)

    # Alpha blend: face region fades smoothly into the clothing image
    blended_np    = blended_np * feather_3c + clothing_np * (1.0 - feather_3c)
    clothing_face = Image.fromarray(blended_np.astype(np.uint8))

    # Step 3 – Background generation

    # Destroy the original background with random noise so the diffusion model
    # isn't biased by whatever scene was behind the person originally.
    # Then inpaint just the background region with the scene prompt.
    print("  [Step 3] Generating background...")

    person_3c    = np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)
    noise_bg     = np.clip(np.random.randn(*np.array(clothing_face).shape) * 255, 0, 255)
    noisy_scene  = (
        np.array(clothing_face).astype(np.float32) * person_3c
        + noise_bg * (1.0 - person_3c)
    )
    image_noisy  = Image.fromarray(noisy_scene.astype(np.uint8))

    bg_images = ip_model.generate(
        prompt=prompt_background,
        negative_prompt=NEGATIVE_PROMPT,
        faceid_embeds=faceid_embeds,
        scale=0.0,                # background is all about the scene, not the face
        num_samples=NUM_SAMPLES,
        seed=SEED,
        guidance_scale=8.5,
        num_inference_steps=INFERENCE_STEPS,
        image=image_noisy,
        mask_image=Image.fromarray(person_mask),   # paint over white (background) pixels
        strength=0.99,
        width=WIDTH,
        height=HEIGHT,
    )

    return bg_images[0]
