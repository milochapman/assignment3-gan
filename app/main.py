from fastapi import FastAPI
from pydantic import BaseModel
import torch
import base64
from io import BytesIO
from PIL import Image

from helper_lib.model import GANGenerator

app = FastAPI()


class GanRequest(BaseModel):
    num_samples: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate_with_gan")
def generate_with_gan(req: GanRequest):
    device = "cpu"
    gen = GANGenerator(z_dim=100)
    try:
        gen.load_state_dict(torch.load("gan_generator_mnist.pth", map_location=device))
    except FileNotFoundError:
        pass
    gen.eval()

    with torch.no_grad():
        noise = torch.randn(req.num_samples, 100, device=device)
        fake = gen(noise).cpu()

    images_b64 = []
    for i in range(req.num_samples):
        img = fake[i].squeeze(0)
        img = (img + 1) / 2
        img = (img * 255).clamp(0, 255).byte().numpy()
        pil_img = Image.fromarray(img, mode="L")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return {"images": images_b64}
