# app.py
import streamlit as st
from PIL import Image
import os, joblib
import torch
from torchvision import models, transforms
import numpy as np

# ---- CNN Loader ----
def load_cnn(model_path, device, num_classes=7):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_image(img, size=224):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)

# ---- Inference ----
def predict(image, cnn_model, ensemble, device):
    x = preprocess_image(image).to(device)
    with torch.no_grad():
        probs = cnn_model(x).softmax(dim=1).cpu().numpy()
    # For demo: use stacked meta-learner
    base_probs = np.hstack([probs, probs, probs])  # simulate 3 base learners
    label_enc = ensemble["label_encoder"]
    meta = ensemble["meta_learner"]
    pred = meta.predict(base_probs)
    prob = meta.predict_proba(base_probs).max()
    return label_enc.inverse_transform(pred)[0], prob

# ---- Streamlit UI ----
st.title("Skin Lesion Classifier — CNN + ML Ensemble")

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

models_dir = "models"
cnn_model_path = os.path.join(models_dir, "resnet18_fold0.pth")
ensemble_path = os.path.join(models_dir, "final_ensemble.pkl")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if os.path.exists(cnn_model_path) and os.path.exists(ensemble_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_model = load_cnn(cnn_model_path, device)
        ensemble = joblib.load(ensemble_path)

        if st.button("Predict"):
            label, prob = predict(image, cnn_model, ensemble, device)
            st.success(f"Prediction: **{label}** (confidence: {prob:.3f})")
    else:
        st.warning("Trained models not found in ./models — please train first.")
