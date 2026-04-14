"""
train_asset_model — train and persist a per-asset model artefact.

Output layout:

    models/<SYMBOL>/
      ml_filter_v1.pkl     — GradientBoostingClassifier
      scaler.pkl           — StandardScaler used for features
      feature_names.json   — ordered list the scaler expects
      training_metadata.json — symbol, train range, n samples, accuracy

The artefact is the hub-and-spoke architecture's concrete output for
Phase 1 (ETH), Phase 2 (SOL), Phase 3 (XRP). A SignalPod loads the
matching artefact based on its `Asset.model_profile`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train_and_save(
    symbol: str,
    mtf_data: Dict[str, pd.DataFrame],
    mtf_features: Dict[str, pd.DataFrame],
    train_end: str,
    out_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Train a GBM on the hard-gate candidates up to `train_end` and save.

    Returns a metadata dict with training stats.
    """
    # Lazy import to avoid a cycle at module load time.
    from src.ml.nyx_pipeline import NYXPipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = NYXPipeline()
    df_15m = mtf_data['15m']
    ctx_1d = pipe._build_1d_context(
        mtf_data.get('1d', pd.DataFrame()),
        mtf_features.get('1d', pd.DataFrame()),
    )
    ctx_1h = pipe._build_1h_context(
        mtf_data.get('1h', pd.DataFrame()),
        mtf_features.get('1h', pd.DataFrame()),
    )

    train_cands = pipe._generate_candidates(
        df_15m.loc[:train_end],
        mtf_features['15m'].loc[:train_end],
        ctx_1d, ctx_1h,
        feat_1h=mtf_features.get('1h', pd.DataFrame()),
        feat_1d=mtf_features.get('1d', pd.DataFrame()),
    )

    if len(train_cands) < 30:
        raise ValueError(
            f"{symbol}: too few train candidates ({len(train_cands)})"
        )

    feature_names = sorted(train_cands[0]['features'].keys())
    X = np.array([[c['features'].get(f, 0) for f in feature_names]
                  for c in train_cands])
    y = np.array([1 if c['outcome_net'] > 0 else 0 for c in train_cands])

    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=10, neginf=-10), -1e6, 1e6)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=3, neginf=-3)

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        min_samples_leaf=30, subsample=0.7, random_state=42,
    )
    model.fit(X_s, y)

    # Persist artefacts.
    with open(out_dir / 'ml_filter_v1.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(out_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(out_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    accuracy = float(accuracy_score(y, model.predict(X_s)))
    meta: Dict[str, Any] = {
        'symbol': symbol,
        'train_end': train_end,
        'n_train_candidates': int(len(train_cands)),
        'n_features': int(len(feature_names)),
        'in_sample_accuracy': accuracy,
        'positive_class_rate': float(np.mean(y)),
    }
    with open(out_dir / 'training_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def load_artifact(out_dir: Union[str, Path]) -> Dict[str, Any]:
    """Reload a previously-saved artefact for inference."""
    out_dir = Path(out_dir)
    with open(out_dir / 'ml_filter_v1.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(out_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(out_dir / 'feature_names.json') as f:
        feature_names = json.load(f)
    with open(out_dir / 'training_metadata.json') as f:
        meta = json.load(f)
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metadata': meta,
    }
