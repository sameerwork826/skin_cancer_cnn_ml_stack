#!/usr/bin/env python3
"""Extract embeddings from a trained torchvision model by removing final fc layer."""
import argparse, os
import torch
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import pandas as pd
from datasets import HAM10000Dataset
from utils import get_transforms
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--metadata', default='ham10000_metadata.csv')
    p.add_argument('--images-dir', default='ham10000_images')
    p.add_argument('--model-path',default='models/resnet18_fold1.pth')
    p.add_argument('--out', default='models/cnn_features.npy')
    p.add_argument('--batch-size', type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    meta_path = os.path.join(args.data_dir, args.metadata)
    images_dir = os.path.join(args.data_dir, args.images_dir)
    df = pd.read_csv(meta_path)
    _, val_tf = get_transforms()
    dataset = HAM10000Dataset(meta_path, images_dir, transform=val_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Identity()  # output embeddings
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    feats = []
    with torch.no_grad():
        for imgs, _, _ in tqdm(loader):
            imgs = imgs.to(device)
            emb = model(imgs)
            emb = emb.cpu().numpy()
            feats.append(emb)
    feats = np.vstack(feats)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, feats)
    print('Saved features to', args.out)

if __name__ == '__main__':
    main()
