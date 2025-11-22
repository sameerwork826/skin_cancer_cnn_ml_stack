#!/usr/bin/env python3
"""Single-image + metadata inference pipeline combining CNN + stacked ML models."""
import argparse, os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
import pandas as pd

def load_cnn(model_path, device):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    # We will use softmax outputs from the trained folds averaged (simple approach)
    model.fc = torch.nn.Linear(num_ftrs, 7)  # change classes if needed
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(pil_img, image_size=224):
    tf = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tf(pil_img).unsqueeze(0)

def ensemble_predict(image_path, meta_json, models_dir, cnn_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = load_cnn(cnn_model_path, device)
    img = Image.open(image_path).convert('RGB')
    x = preprocess_image(img).to(device)
    with torch.no_grad():
        out = cnn(x)
        probs = out.softmax(dim=1).cpu().numpy()

    # load tabular models folds predictions? For demo, we will load saved meta-learner and base OOF preds mean
    meta = joblib.load(os.path.join(models_dir, 'final_ensemble.pkl'))
    meta_learner = meta['meta_learner']
    label_encoder = meta['label_encoder']

    # For simplicity: create dummy tabular probs by repeating cnn probs for each model
    # (In production, load base models and call predict_proba on concatenated features)
    base_probs_concat = np.hstack([probs, probs, probs])
    pred_label = meta_learner.predict(base_probs_concat)
    pred_prob = meta_learner.predict_proba(base_probs_concat).max()

    return label_encoder.inverse_transform(pred_label)[0], float(pred_prob)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--cnn-model', default='models/resnet18_fold0.pth')
    parser.add_argument('--meta', default='{}')
    args = parser.parse_args()
    label, prob = ensemble_predict(args.image, args.meta, args.models_dir, args.cnn_model)
    print('Predicted:', label, 'with prob', prob)
