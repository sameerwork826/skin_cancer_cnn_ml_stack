#!/usr/bin/env python3
"""Train a ResNet-18 on HAM10000 (template). Produces a saved model and OOF predictions CSV."""

import argparse, os, json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torchvision import models
from datasets import HAM10000Dataset
from utils import get_transforms, pil_loader
from tqdm import tqdm
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data', help='data folder with images and metadata csv')
    p.add_argument('--metadata', default='ham10000_metadata.csv', help='metadata csv name')
    p.add_argument('--images-dir', default='ham10000_images', help='folder inside data/ containing images')
    p.add_argument('--save-dir', default='models', help='where to save models and oof pred')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    meta_path = os.path.join(args.data_dir, args.metadata)
    images_dir = os.path.join(args.data_dir, args.images_dir)
    df = pd.read_csv(meta_path)
    # Encode labels
    df['label_enc'] = df['dx'].astype('category').cat.codes
    n_classes = df['label_enc'].nunique()
    labels = df['label_enc'].values
    # mapping from diagnosis string to integer encoding
    label_map = dict(zip(df['dx'].astype(str), df['label_enc'].astype(int)))

    train_tf, val_tf = get_transforms()
    dataset = HAM10000Dataset(meta_path, images_dir, transform=train_tf)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(dataset), n_classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(df)), labels)):
        print(f"Starting fold {fold}")
        # Simple subset dataloaders
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = 1e9
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for imgs, labs, _ in tqdm(train_loader):
                imgs = imgs.to(device)
                # dataset may return either encoded ints or diagnosis strings; map accordingly
                labs_mapped = [
                    int(x) if (isinstance(x, (int, np.integer)) or str(x).isdigit()) else label_map.get(str(x), -1)
                    for x in labs
                ]
                if any([l == -1 for l in labs_mapped]):
                    raise ValueError(f"Found unknown label in training batch: {labs}")
                labs = torch.tensor(labs_mapped, dtype=torch.long).to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
                running += loss.item()
            # simple validation
            model.eval()
            val_loss = 0.0
            preds = []
            with torch.no_grad():
                for imgs, labs, _ in val_loader:
                    imgs = imgs.to(device)
                    labs_mapped = [
                        int(x) if (isinstance(x, (int, np.integer)) or str(x).isdigit()) else label_map.get(str(x), -1)
                        for x in labs
                    ]
                    if any([l == -1 for l in labs_mapped]):
                        raise ValueError(f"Found unknown label in validation batch: {labs}")
                    labs = torch.tensor(labs_mapped, dtype=torch.long).to(device)
                    out = model(imgs)
                    val_loss += criterion(out, labs).item()
                    preds.append(out.softmax(dim=1).cpu().numpy())
            val_loss /= len(val_loader)
            preds = np.vstack(preds)
            print(f"Fold {fold} Epoch {epoch} val_loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'resnet18_fold{fold}.pth'))
        # After fold training, generate OOF preds for val_idx using best saved model
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f'resnet18_fold{fold}.pth')))
        model.eval()
        all_preds = []
        with torch.no_grad():
            for imgs, labs, _ in DataLoader(val_set, batch_size=args.batch_size):
                imgs = imgs.to(device)
                out = model(imgs)
                all_preds.append(out.softmax(dim=1).cpu().numpy())
        all_preds = np.vstack(all_preds)
        oof_preds[val_idx] = all_preds

    # Save OOF predictions alongside metadata
    oof_df = pd.read_csv(meta_path)
    for c in range(oof_preds.shape[1]):
        oof_df[f'oof_prob_{c}'] = oof_preds[:, c] if c==0 else oof_df.get(f'oof_prob_{c}', oof_preds[:, c])
    # This above assignment ensures columns exist; better to set iteratively
    for c in range(oof_preds.shape[1]):
        oof_df[f'oof_prob_{c}'] = oof_preds[:, c]
    oof_df.to_csv(os.path.join(args.save_dir, 'oof_predictions.csv'), index=False)
    print('Saved OOF predictions to', os.path.join(args.save_dir, 'oof_predictions.csv'))

if __name__ == '__main__':
    main()
