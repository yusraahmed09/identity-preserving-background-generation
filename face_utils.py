"""
face_utils.py
-------------
Everything related to faces:
  - Extracting a 512-d ArcFace embedding that captures who the person is
    (used later by IP-Adapter to keep the face identity consistent).
  - Finding the face bounding box with a bit of padding so we can crop/blend
    cleanly without hard edges.

"""

import cv2
import torch


def extract_faceid_embeds(image_path: str, device: str) -> torch.Tensor:
    """
    Run InsightFace on the image and return the ArcFace embedding
    for the largest detected face.

    The embedding is a unit-norm 512-d vector that encodes facial identity.
    IP-Adapter will condition the diffusion model on this vector so the
    generated face looks like the same person.

    Args:
        image_path: Path to the input image (BGR, any resolution).
        device:     Torch device string, e.g. "cuda" or "cpu".

    Returns:
        Tensor of shape (1, 512), dtype float16, on the requested device.

    Raises:
        FileNotFoundError: If the image cannot be opened.
        ValueError:        If no face is found in the image.
    """
    from insightface.app import FaceAnalysis

    # InsightFace's "buffalo_l" pack includes the ArcFace recognition model
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    faces = app.get(image)
    if not faces:
        raise ValueError(f"No face detected in: {image_path}")

    # If multiple faces are detected, keep the largest one (most prominent subject)
    print(f"  Detected {len(faces)} face(s) – using the largest.")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # normed_embedding is already L2-normalised by InsightFace
    faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)  # (1, 512)
    return faceid_embeds.to(device, dtype=torch.float16)


def get_face_bbox(image_path: str) -> tuple | None:
    """
    Detect the largest face and return its bounding box with 20 % padding
    on each side so downstream blending has some feathering room.

    Args:
        image_path: Path to the image to analyse.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clipped to image bounds.
        Returns None if no face is found.
    """
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    image_bgr = cv2.imread(image_path)
    faces = app.get(image_bgr)
    if not faces:
        return None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = [int(c) for c in face.bbox]

    # Pad by 20 % of the larger face dimension so crop edges don't look abrupt
    pad = int(max(x2 - x1, y2 - y1) * 0.2)
    h, w = image_bgr.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return (x1, y1, x2, y2)
