"""Constroi a tabela de features (X.csv) e os rotulos (y.csv).

Executa o pipeline ate a extracao de features:
  amostragem -> segmentacao (metodo escolhido) -> extracao de features.

Uso:
    python -m src.build_dataset            # a partir da raiz do projeto
    python src/build_dataset.py            # tambem funciona
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# permite executar tanto como modulo (-m) quanto como script direto
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as cfg
    from src import dataset as ds
    from src import features as ft
    from src import segmentation as seg
else:
    from . import config as cfg
    from . import dataset as ds
    from . import features as ft
    from . import segmentation as seg

SEG_METHOD = "hsv"   # metodo escolhido para o pipeline principal (ver notebook 01)


def main() -> None:
    t0 = time.time()
    index = ds.build_sample_index()
    print(f"Amostra: {len(index)} imagens "
          f"({(index.label == 0).sum()} fresh / {(index.label == 1).sum()} rotten)")
    print(index.groupby(["fruit", "label_name"]).size())

    records = []
    n = len(index)
    for i, row in index.iterrows():
        img = ds.load_image(row["path"])
        mask = seg.segment(img, SEG_METHOD)
        feats = ft.extract_all(img, mask)
        feats["coverage"] = seg.mask_coverage(mask)
        records.append(feats)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  features {i+1}/{n}")

    X = pd.DataFrame(records)
    feature_cols = ft.all_feature_names()
    X = X[feature_cols + ["coverage"]]

    # metadados ajudam na analise de erros (nao entram no treino)
    meta = index[["path", "fruit", "label", "label_name", "folder"]].copy()

    X_out = pd.concat([meta[["fruit", "folder", "path"]].reset_index(drop=True),
                       X.reset_index(drop=True)], axis=1)
    y_out = meta[["label", "label_name", "fruit"]].reset_index(drop=True)

    X_out.to_csv(cfg.OUTPUTS / "X.csv", index=False)
    y_out.to_csv(cfg.OUTPUTS / "y.csv", index=False)

    print(f"\nX.csv  -> {cfg.OUTPUTS / 'X.csv'}  shape={X_out.shape}")
    print(f"y.csv  -> {cfg.OUTPUTS / 'y.csv'}  shape={y_out.shape}")
    print(f"Cobertura media da mascara: {X['coverage'].mean():.3f} "
          f"(min {X['coverage'].min():.3f}, max {X['coverage'].max():.3f})")
    print(f"Concluido em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
