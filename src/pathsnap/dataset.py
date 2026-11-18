"""Loaders for teammates (Person 2 models / Person 3 HMM). Keeps everyone on the SAME
rows, features and folds so model comparisons are fair.

    from pathsnap.dataset import load_xy
    Xtr, ytr, gtr, ftr, meta_tr = load_xy("train")            # fold>=0, label_valid only
    Xte, yte, gte, _, meta_te   = load_xy("student_test")
"""
import pandas as pd
from . import config


def load_candidates(partition, feature_dir=None):
    d = feature_dir or config.FEATURE_DIR
    return pd.read_parquet(f"{d}/{partition}_candidates.parquet")


def load_points(partition, feature_dir=None):
    d = feature_dir or config.FEATURE_DIR
    return pd.read_parquet(f"{d}/{partition}_points.parquet")


def load_xy(partition, feature_dir=None, only_valid=True):
    df = load_candidates(partition, feature_dir)
    if only_valid:
        df = df[df.label_valid]
    if partition == "train":
        df = df[df.fold >= 0]
    X = df[config.FEATURE_COLUMNS]
    meta = df[["trip_id", "point_idx", "arc_id"]]
    return X, df["label"], df["trip_id"], df.get("fold"), meta
