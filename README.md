# Skin Lesion Classification — CNN + Gradient-Boosting Stacking Pipeline

### Deployed link for this project 
https://skin-cancer-pred-sameer.streamlit.app/

**What this repo contains**
- An end-to-end template to build a hybrid deep-learning + classical-ML pipeline:
  - Fine-tune a CNN (ResNet-18) on skin lesion images (HAM10000).
  - Use CNN predictions / embeddings as features for tabular models (XGBoost, LightGBM, CatBoost).
  - Stack / ensemble the ML models for better performance.
  - A simple Streamlit demo app for inference.
  - Dockerfile to run the demo.

**Dataset**
- We recommend **HAM10000** (a well-known skin lesion dataset available on Kaggle).
- Dataset name on Kaggle: `kmader/skin-cancer-mnist-ham10000`
- Download the dataset with Kaggle CLI (instructions below) or upload the images & metadata into `data/`.

**Project structure**
```
.
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ prepare_data.sh
├─ src/
│  ├─ train_cnn.py
│  ├─ extract_features.py
│  ├─ train_tabular_models.py
│  ├─ stack_ensemble.py
│  ├─ inference.py
│  ├─ streamlit_app.py
│  ├─ utils.py
│  └─ datasets.py
├─ models/                # where trained models will be saved
└─ data/                  # expected location for images & csv
```

---

## Quick start (local)

1. Install Kaggle CLI and download dataset (one-time)
   - Place your `kaggle.json` in `~/.kaggle/kaggle.json` (or set env vars).
   - Run:
     ```bash
     bash prepare_data.sh
     ```
   - This will create `data/ham10000_images/` and `data/ham10000_metadata.csv` (expected paths).

2. Create a Python environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Train CNN (resnet18) and generate OOF predictions:
   ```bash
   python src/train_cnn.py --data-dir data --save-dir models --epochs 10 --batch-size 32
   ```

4. Extract embeddings / features from images using the trained CNN:
   ```bash
   python src/extract_features.py --data-dir data --model-path models/resnet18_best.pth --out models/cnn_features.npy
   ```

5. Train tabular models (XGBoost, LightGBM, CatBoost) on CNN features + metadata:
   ```bash
   python src/train_tabular_models.py --features models/cnn_features.npy --meta data/ham10000_metadata.csv --out-dir models
   ```

6. Train stack / meta-learner:
   ```bash
   python src/stack_ensemble.py --models-dir models --out models/final_ensemble.pkl
   ```

7. Run Streamlit demo:
   ```bash
   streamlit run src/streamlit_app.py
   ```

---

## Docker (quick demo)
Build:
```
docker build -t skin-cancer-demo:latest .
```
Run (bind a local `models/` and `data/` directory into the container):
```
docker run --rm -p 8501:8501 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data skin-cancer-demo:latest
```
Then visit `http://localhost:8501`.

---

## Notes & tips
- This repo is a **template**. For real experiments:
  - Add proper transforms, augmentations (Albumentations), learning rate schedules, and modern architectures (EfficientNet, Densenet, etc.).
  - Use `StratifiedGroupKFold` if patient/group info is available (reduces leakage).
  - Save OOF predictions carefully so boosting models don't see validation images.
- If you want, I can:
  - Add a Colab notebook that runs a small end-to-end demo on a tiny subset.
  - Provide ready-to-run Docker Compose for GPU.

---

## Reference resume bullet (example)
- **Stacked CNN + Gradient Boosting Ensemble for Skin Lesion Classification:** Fine-tuned ResNet-18 for image feature extraction; generated out-of-fold (OOF) predictions and trained XGBoost, LightGBM and CatBoost on CNN-derived features plus clinical metadata; stacked models with a meta-learner and soft-voting ensemble, improving partial AUC from **0.144** (image-alone) to **0.167** (stacked ensemble).

