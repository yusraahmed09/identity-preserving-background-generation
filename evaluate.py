import torch
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
import torch.nn.functional as F
import piq
import torchvision.transforms as T

DEVICE = "cuda"

def compute_face_identity(source: Image.Image, generated: Image.Image) -> float | None:
    """
    computes cosine similarity - how much is the output image similar to the input image
    """
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    src_bgr = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    gen_bgr = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)

    # Resize generated to match source
    generated = generated.resize(source.size)
    gen_bgr = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)

    src_faces = app.get(src_bgr)
    gen_faces = app.get(gen_bgr)

    if not src_faces or not gen_faces:
        print("Face not detected in one of the images")
        return None

    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    gen_face = max(gen_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    return float(np.dot(src_face.normed_embedding, gen_face.normed_embedding))

@torch.no_grad()
def compute_clip_score(image: Image.Image, text: str, device: str) -> float: 
    
    """
    computes clip score - how much the output matches with the text prompt
    """
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval()

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)

    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)

    return float((image_embeds * text_embeds).sum(dim=-1).item())
    
def func_evaluate(source, generated, PROMPT):
    print("Evaluating...\n")

    face_score       = compute_face_identity(source, generated)
    clip_score       = compute_clip_score(generated, PROMPT, DEVICE)
    tensor = T.ToTensor()(generated).unsqueeze(0)
    brisque_score = piq.brisque(tensor)  
    
    return face_score, clip_score, brisque_score