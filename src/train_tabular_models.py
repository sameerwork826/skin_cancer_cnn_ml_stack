#!/usr/bin/env python3
"""Train XGBoost, LightGBM, CatBoost on CNN features + metadata using OOF strategy."""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from tqdm import tqdm
# python src/train_tabular_models.py --features models/cnn_features.npy --meta data/ham10000_metadata.csv --out-dir models
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features',default='models/cnn_features.npy')
    p.add_argument('--meta', default='data/HAM10000_metadata.csv')
    p.add_argument('--out-dir', default='models')
    p.add_argument('--n-splits', type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    feats = np.load(args.features)
    meta = pd.read_csv(args.meta)
    # ensure ordering matches the features (we assume same ordering)
    le = LabelEncoder()
    y = le.fit_transform(meta['dx'].values)
    # Build tabular feature matrix: concat cnn-features + some metadata columns if available
    X = feats
    # Example simple metadata features: age (if exists), sex, localization -> simple encodings
    add_feats = []
    if 'age' in meta.columns:
        add_feats.append(meta['age'].fillna(-1).values.reshape(-1,1))
    if 'sex' in meta.columns:
        add_feats.append(pd.get_dummies(meta['sex'].fillna('na')).values)
    if 'localization' in meta.columns:
        add_feats.append(pd.get_dummies(meta['localization'].fillna('na')).values)
    if add_feats:
        X = np.hstack([X] + add_feats)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    oof_preds_xgb = np.zeros((X.shape[0], len(le.classes_)))
    oof_preds_lgb = np.zeros_like(oof_preds_xgb)
    oof_preds_cat = np.zeros_like(oof_preds_xgb)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print('Fold', fold)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {'objective':'multi:softprob', 'num_class':len(le.classes_), 'eval_metric':'mlogloss', 'verbosity':0}
        model_xgb = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval,'valid')], early_stopping_rounds=20, verbose_eval=False)
        oof_preds_xgb[val_idx] = model_xgb.predict(dval)

        # LightGBM
        ltrain = lgb.Dataset(X_tr, label=y_tr)
        lval = lgb.Dataset(X_val, label=y_val, reference=ltrain)
        params_l = {'objective':'multiclass','num_class':len(le.classes_),'metric':'multi_logloss','verbosity':-1}
        model_lgb = lgb.train(params_l, ltrain, num_boost_round=100, valid_sets=[lval])
        oof_preds_lgb[val_idx] = model_lgb.predict(X_val)

        # CatBoost
        model_cat = CatBoostClassifier(iterations=200, verbose=False, loss_function='MultiClass')
        model_cat.fit(X_tr, y_tr)
        oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)

        # Save fold models
        joblib.dump(model_xgb, os.path.join(args.out_dir, f'xgb_fold{fold}.joblib'))
        joblib.dump(model_lgb, os.path.join(args.out_dir, f'lgb_fold{fold}.joblib'))
        model_cat.save_model(os.path.join(args.out_dir, f'cat_fold{fold}.cbm'))

    # Save OOF preds
    np.save(os.path.join(args.out_dir, 'oof_xgb.npy'), oof_preds_xgb)
    np.save(os.path.join(args.out_dir, 'oof_lgb.npy'), oof_preds_lgb)
    np.save(os.path.join(args.out_dir, 'oof_cat.npy'), oof_preds_cat)
    print('Saved OOF preds for tabular models into', args.out_dir)

if __name__ == '__main__':
    main()
