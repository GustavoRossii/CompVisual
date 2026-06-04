"""Utilitarios de modelagem compartilhados pelos notebooks 03 e 04.

Centraliza o split estratificado treino/val/teste e a fabrica de modelos, de modo
que a classificacao (03) e a explicabilidade (04) usem EXATAMENTE o mesmo split e
os mesmos hiperparametros, garantindo coerencia e reprodutibilidade.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as cfg
from . import features as ft


# --------------------------------------------------------------------------- #
# Dados                                                                        #
# --------------------------------------------------------------------------- #
def load_Xy():
    """Carrega X.csv/y.csv e devolve (X_df_features, y, meta_df)."""
    X = pd.read_csv(cfg.OUTPUTS / "X.csv")
    y = pd.read_csv(cfg.OUTPUTS / "y.csv")
    feat_cols = ft.all_feature_names()
    meta = X[["fruit", "folder", "path"]].copy()
    return X[feat_cols].copy(), y["label"].values, meta


@dataclass
class Split:
    Xtr: pd.DataFrame
    Xval: pd.DataFrame
    Xte: pd.DataFrame
    ytr: np.ndarray
    yval: np.ndarray
    yte: np.ndarray
    idx_tr: np.ndarray
    idx_val: np.ndarray
    idx_te: np.ndarray

    @property
    def Xtrval(self) -> pd.DataFrame:
        return pd.concat([self.Xtr, self.Xval])

    @property
    def ytrval(self) -> np.ndarray:
        return np.concatenate([self.ytr, self.yval])


def make_split(X: pd.DataFrame, y: np.ndarray,
               random_state: int = cfg.RANDOM_STATE) -> Split:
    """Split estratificado 60% treino / 20% validacao / 20% teste.

    O conjunto de teste e separado primeiro e nunca e tocado ate a avaliacao final.
    """
    idx = np.arange(len(y))
    # 20% teste
    idx_trval, idx_te = train_test_split(
        idx, test_size=0.20, stratify=y, random_state=random_state)
    # dos 80% restantes, 25% -> validacao (=> 20% do total)
    idx_tr, idx_val = train_test_split(
        idx_trval, test_size=0.25, stratify=y[idx_trval], random_state=random_state)

    return Split(
        Xtr=X.iloc[idx_tr], Xval=X.iloc[idx_val], Xte=X.iloc[idx_te],
        ytr=y[idx_tr], yval=y[idx_val], yte=y[idx_te],
        idx_tr=idx_tr, idx_val=idx_val, idx_te=idx_te,
    )


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #
def make_models(random_state: int = cfg.RANDOM_STATE) -> dict:
    """Retorna {nome: (pipeline, param_grid)} para os classificadores classicos.

    Cada pipeline embute o StandardScaler, de modo que o scaler e ajustado APENAS
    nos dados de treino dentro de cada fold da validacao cruzada (sem vazamento).
    """
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "KNN": (
            pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [3, 5, 7, 9, 11],
             "clf__weights": ["uniform", "distance"]},
        ),
        "LogReg": (
            pipe(LogisticRegression(max_iter=5000, random_state=random_state)),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            pipe(RandomForestClassifier(random_state=random_state, n_jobs=-1)),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [None, 8, 16]},
        ),
        "SVM": (
            pipe(SVC(probability=True, random_state=random_state)),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", "auto"],
             "clf__kernel": ["rbf"]},
        ),
    }
