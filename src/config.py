"""Configuração central do pipeline de inspeção visual de frutas.

Todos os caminhos são relativos à raiz do projeto (PipelineFrutas/), de modo que
os notebooks e scripts funcionem independente de onde forem executados.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]         
DATASET_DIR = ROOT / "dataset" / "train"     
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figuras"
SEG_DIR = OUTPUTS / "segmentação"
FEAT_DIR = OUTPUTS / "features"
CM_DIR = OUTPUTS / "matrizes"
METRIC_DIR = OUTPUTS / "metricas"
ERR_DIR = OUTPUTS / "erros"

for _d in (OUTPUTS, FIG_DIR, SEG_DIR, FEAT_DIR, CM_DIR, METRIC_DIR, ERR_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FRESH_FOLDERS = ["freshapples", "freshbanana", "freshoranges"]
ROTTEN_FOLDERS = ["rottenapples", "rottenbanana", "rottenoranges"]

FOLDER_INFO = {
    "freshapples":   ("apple",  0),
    "freshbanana":   ("banana", 0),
    "freshoranges":  ("orange", 0),
    "rottenapples":  ("apple",  1),
    "rottenbanana":  ("banana", 1),
    "rottenoranges": ("orange", 1),
}

LABEL_NAMES = {0: "fresh", 1: "rotten"}

N_PER_GROUP = 67

MAX_SIDE = 384

RANDOM_STATE = 42
