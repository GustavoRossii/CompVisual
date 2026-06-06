"""Amostragem reproduzivel e carregamento de imagens do dataset de frutas."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config as cfg


# Prefixos das imagens AUMENTADAS presentes no dataset Kaggle. Sao versoes
# derivadas das originais (mesma fruta rotacionada/transladada/com ruido). Inclui-las
# causaria VAZAMENTO DE DADOS (a mesma fruta cairia em train e test) e ainda
# prejudica a segmentacao (cantos pretos da rotacao, ruido sal-e-pimenta).
# As originais sao exatamente as que comecam por "Screen Shot".
_AUG_PREFIXES = ("rotated_by", "translation_", "saltandpepper_", "vertical_flip_")


def _is_original(path: Path) -> bool:
    name = path.name.lower()
    return not name.startswith(_AUG_PREFIXES)


def _list_images(folder: Path, originals_only: bool = True) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    if originals_only:
        files = [p for p in files if _is_original(p)]
    return sorted(files)


def build_sample_index(n_per_group: int = cfg.N_PER_GROUP,
                       random_state: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Sorteia n_per_group imagens de cada pasta (fruta x condicao).

    Retorna um DataFrame com colunas: path, fruit, label, label_name, folder.
    A amostragem e estavel para um dado random_state (reprodutibilidade).
    """
    if not cfg.DATASET_DIR.exists():
        raise FileNotFoundError(
            "Dataset nao encontrado. Organize as imagens em "
            f"{cfg.DATASET_DIR} com as subpastas: "
            + ", ".join(cfg.FOLDER_INFO.keys())
        )

    rng = np.random.default_rng(random_state)
    rows = []
    for folder, (fruit, label) in cfg.FOLDER_INFO.items():
        files = _list_images(cfg.DATASET_DIR / folder)
        if len(files) < n_per_group:
            raise ValueError(f"{folder}: so existem {len(files)} imagens (<{n_per_group}).")
        idx = rng.choice(len(files), size=n_per_group, replace=False)
        for i in sorted(idx):
            rows.append({
                "path": str(files[i]),
                "fruit": fruit,
                "label": label,
                "label_name": cfg.LABEL_NAMES[label],
                "folder": folder,
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


def load_image(path: str | Path, max_side: int = cfg.MAX_SIDE) -> np.ndarray:
    """Le uma imagem em RGB e redimensiona para que o lado maior == max_side."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img
