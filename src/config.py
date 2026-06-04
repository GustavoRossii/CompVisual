"""Configuracao central do pipeline de inspecao visual de frutas.

Todos os caminhos sao relativos a raiz do projeto (PipelineFrutas/), de modo que
os notebooks e scripts funcionem independente de onde forem executados.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos                                                                     #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # .../PipelineFrutas
DATASET_DIR = ROOT / "dataset" / "train"            # usamos o split 'train' do Kaggle como fonte
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figuras"
SEG_DIR = OUTPUTS / "segmentacao"
FEAT_DIR = OUTPUTS / "features"
CM_DIR = OUTPUTS / "matrizes"
METRIC_DIR = OUTPUTS / "metricas"
ERR_DIR = OUTPUTS / "erros"

for _d in (OUTPUTS, FIG_DIR, SEG_DIR, FEAT_DIR, CM_DIR, METRIC_DIR, ERR_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Classes do dataset                                                          #
# --------------------------------------------------------------------------- #
# Pastas originais do dataset Kaggle "Fruits fresh and rotten for classification".
FRESH_FOLDERS = ["freshapples", "freshbanana", "freshoranges"]
ROTTEN_FOLDERS = ["rottenapples", "rottenbanana", "rottenoranges"]

# Mapa pasta -> (fruta, condicao). condicao 0 = fresh (OK), 1 = rotten (defeituoso).
FOLDER_INFO = {
    "freshapples":   ("apple",  0),
    "freshbanana":   ("banana", 0),
    "freshoranges":  ("orange", 0),
    "rottenapples":  ("apple",  1),
    "rottenbanana":  ("banana", 1),
    "rottenoranges": ("orange", 1),
}

LABEL_NAMES = {0: "fresh", 1: "rotten"}

# --------------------------------------------------------------------------- #
# Amostragem                                                                   #
# --------------------------------------------------------------------------- #
# ~200 imagens por classe binaria, balanceado entre as 3 frutas.
# 3 frutas x 67 = 201 imagens fresh e 201 rotten.
N_PER_GROUP = 67

# Redimensionamento de trabalho (lado maior). Acelera a extracao de features.
# Features de forma usadas sao invariantes a escala; cor/textura ficam estaveis.
MAX_SIDE = 384

# --------------------------------------------------------------------------- #
# Reprodutibilidade                                                            #
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
