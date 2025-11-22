import torch, torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transforms(image_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_transforms, val_transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def predict_image_batch(model, device, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds.append(out.cpu().numpy())
    return np.vstack(preds)
