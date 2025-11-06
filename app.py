import torch, torchvision
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
import cv2

# ---- UPDATE THESE to match your project ----
WEIGHTS_PATH = "checkpoints/best_model.pth"   # put your trained weights here
IMG_SIZE = 512                                # change to your training size
DEVICE = "cpu"                                # Spaces CPU by default

# ----- Minimal U-Net skeleton matching your repo's model -----
# If you already have a model class in your repo (e.g., models/unet.py),
# import it instead and remove this class.
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, base=64):
        super().__init__()
        self.d1 = DoubleConv(in_c, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8)
        self.p4 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*8, base*16)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.c4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_c, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b  = self.b(p4)
        u4 = self.u4(b);  c4 = self.c4(torch.cat([u4, d4], 1))
        u3 = self.u3(c4); c3 = self.c3(torch.cat([u3, d3], 1))
        u2 = self.u2(c3); c2 = self.c2(torch.cat([u2, d2], 1))
        u1 = self.u1(c2); c1 = self.c1(torch.cat([u1, d1], 1))
        return self.out(c1)

# ---- load model ----
@torch.inference_mode()
def load_model():
    model = UNet(in_c=1, out_c=1)
    sd = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    # support checkpoints that wrap state_dict
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model

MODEL = load_model()

def preprocess(img_pil: Image.Image):
    # convert to grayscale, resize, to tensor
    img = np.array(img_pil.convert("L"))
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = img_resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5  # if you normalized similarly in training; adjust as needed
    x = torch.from_numpy(x)[None, None, ...].to(DEVICE)
    return x, (w, h), img

@torch.inference_mode()
def predict(image: Image.Image, threshold: float = 0.5, overlay_alpha: float = 0.5):
    x, (orig_w, orig_h), orig_gray = preprocess(image)
    logits = MODEL(x)
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # overlay mask in green
    color = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    overlay[mask_resized > 0] = (0, 255, 0)
    blended = cv2.addWeighted(color, 1 - overlay_alpha, overlay, overlay_alpha, 0)

    return Image.fromarray(mask_resized), Image.fromarray(blended)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Panoramic X-ray"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Threshold"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Overlay alpha"),
    ],
    outputs=[
        gr.Image(type="pil", label="Binary Mask"),
        gr.Image(type="pil", label="Overlay"),
    ],
    title="Teeth Segmentation (U-Net)",
    description="Upload a panoramic dental X-ray to segment teeth. Uses your trained U-Net weights."
)

if __name__ == "__main__":
    demo.launch()
