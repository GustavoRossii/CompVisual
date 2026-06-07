
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as cfg, dataset as ds, segmentation as seg
else:
    from . import config as cfg, dataset as ds, segmentation as seg


def _panel(ax, img, title):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def comparison_grid(index, n_per=1, seed=0):
    """Uma linha por (fruta x condição): original | Otsu | HSV."""
    folders = list(cfg.FOLDER_INFO.keys())
    rng = __import__("numpy").random.default_rng(seed)
    rows = []
    for f in folders:
        sub = index[index.folder == f]
        rows.append(sub.iloc[int(rng.integers(len(sub)))])

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)))
    for i, r in enumerate(rows):
        img = ds.load_image(r["path"])
        m_otsu = seg.segment_otsu(img)
        m_hsv = seg.segment_hsv(img)
        _panel(axes[i, 0], img, f"{r['folder']} (original)")
        _panel(axes[i, 1], seg.apply_mask(img, m_otsu),
               f"Otsu  cov={seg.mask_coverage(m_otsu):.2f}")
        _panel(axes[i, 2], seg.apply_mask(img, m_hsv),
               f"HSV   cov={seg.mask_coverage(m_hsv):.2f}")
    fig.suptitle("Segmentação: Otsu vs HSV (objeto recortado)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = cfg.SEG_DIR / "comparativo_otsu_vs_hsv.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    index = ds.build_sample_index()
    out = comparison_grid(index)
    print("salvo:", out)


if __name__ == "__main__":
    main()
