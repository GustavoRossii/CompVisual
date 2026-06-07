from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from . import config as cfg

LABELS = ["fresh", "rotten"]


def plot_confusion(y_true, y_pred, title, save_as=None, normalize=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False,
              values_format=".2f" if normalize else "d")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if save_as:
        fig.savefig(cfg.CM_DIR / save_as, dpi=130)
    return fig


def plot_roc(models_fitted, Xte, yte, save_as=None):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, model in models_fitted.items():
        RocCurveDisplay.from_estimator(model, Xte, yte, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.6)
    ax.set_title("Curvas ROC (conjunto de teste)", fontsize=11)
    fig.tight_layout()
    if save_as:
        fig.savefig(cfg.FIG_DIR / save_as, dpi=130)
    return fig


def boxplots_by_class(X, y, feats, save_as=None, ncols=4):
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    lab = np.where(np.asarray(y) == 0, "fresh", "rotten")
    for i, f in enumerate(feats):
        sns.boxplot(x=lab, y=X[f].values, ax=axes[i],
                    hue=lab, palette={"fresh": "#69b34c", "rotten": "#b3593c"},
                    legend=False, order=["fresh", "rotten"])
        axes[i].set_title(f, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distribuição das features por classe", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_as:
        fig.savefig(cfg.FEAT_DIR / save_as, dpi=130)
    return fig
