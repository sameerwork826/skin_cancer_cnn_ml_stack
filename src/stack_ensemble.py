#!/usr/bin/env python3
"""Train a simple meta-learner (logistic regression) on OOF predictions from base tabular models."""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models-dir', default='models')
    p.add_argument('--meta', default='data/HAM10000_metadata.csv')
    p.add_argument('--out', default='models/final_ensemble.pkl')
    return p.parse_args()

def main():
    args = parse_args()
    oof_xgb = np.load(os.path.join(args.models_dir, 'oof_xgb.npy'))
    oof_lgb = np.load(os.path.join(args.models_dir, 'oof_lgb.npy'))
    oof_cat = np.load(os.path.join(args.models_dir, 'oof_cat.npy'))
    X_stack = np.hstack([oof_xgb, oof_lgb, oof_cat])
    meta = pd.read_csv(args.meta)
    le = LabelEncoder()
    y = le.fit_transform(meta['dx'].values)

    meta_learner = LogisticRegression(max_iter=1000, multi_class='multinomial')
    meta_learner.fit(X_stack, y)
    joblib.dump({'meta_learner': meta_learner, 'label_encoder': le}, args.out)
    print('Saved meta-learner to', args.out)

if __name__ == '__main__':
    main()
