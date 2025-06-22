import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
#from train_cgan_mnist import Generator

#!pip install streamlit

# ─── CONFIG ────────────────────────────────────────────────────────────────
BATCH_SIZE      = 128
Z_DIM           = 100            # noise vector size
NUM_CLASSES     = 10
IMG_SIZE        = 28
LR              = 2e-4
EPOCHS          = 30
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH       = 'cgan_generator.pth'
# ────────────────────────────────────────────────────────────────────────────

# ─── MODEL DEFS ────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: noise Z concatenated with one-hot label (size 100+10)
        self.net = nn.Sequential(
            nn.Linear(Z_DIM + NUM_CLASSES, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, IMG_SIZE*IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        img = self.net(x)
        return img.view(-1, 1, IMG_SIZE, IMG_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: flattened image + one-hot label
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE*IMG_SIZE + NUM_CLASSES, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat([img.view(img.size(0), -1), labels], dim=1)
        return self.net(x)
# ────────────────────────────────────────────────────────────────────────────

# ─── HELPERS ────────────────────────────────────────────────────────────────
def one_hot(labels):
    return torch.eye(NUM_CLASSES, device=DEVICE)[labels]

def sample_noise_and_labels(batch):
    z = torch.randn(batch, Z_DIM, device=DEVICE)
    lbl = torch.randint(0, NUM_CLASSES, (batch,), device=DEVICE)
    return z, one_hot(lbl)
# ────────────────────────────────────────────────────────────────────────────

# ─── CONFIG ────────────────────────────────────────────────────────────
MODEL_PATH   = 'cgan_generator.pth'
Z_DIM        = 100
NUM_CLASSES  = 10
NUM_SAMPLES  = 5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ────────────────────────────────────────────────────────────────────────
def load_generator():
    G = Generator().to(DEVICE).eval()
    G.load_state_dict(torch.load("cgan_generator.pth", map_location=DEVICE))
    return G
    
def sample_images(G, digit, n=NUM_SAMPLES):
    z = torch.randn(n, Z_DIM, device=DEVICE)
    labels = torch.full((n,), digit, dtype=torch.long, device=DEVICE)
    onehot = torch.eye(NUM_CLASSES, device=DEVICE)[labels]
    with torch.no_grad():
        imgs = G(z, onehot).cpu()  # (n,1,28,28)
    return (imgs + 1) / 2  # scale to [0,1]

st.title("✍️ Handwritten Digit CGAN Demo")
digit = st.selectbox("Choose a digit (0–9):", list(range(NUM_CLASSES)))
if st.button(f"Generate {NUM_SAMPLES} samples of {digit}"):
    G = load_generator()
    batch = sample_images(G, digit)
    grid = make_grid(batch, nrow=NUM_SAMPLES, padding=4)
    npimg = grid.permute(1,2,0).numpy()
    st.image(npimg, clamp=True, caption=f"Digit {digit}", width=500)

