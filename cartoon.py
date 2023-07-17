import  torch
from PIL import Image

face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
model = "weights/paprika.pt"
img = Image.open("data/rainier.jpg").convert("RGB")
out = face2paint(model, img)