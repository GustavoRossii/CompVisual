"""Extracao de features manuais por fruta (vetor X de uma imagem).

Famílias (todas pedidas no enunciado, Aula 8):
  - FORMA:    area_frac, perimeter_norm, eccentricity, solidity, extent,
              circularity, aspect_ratio, equiv_diam_norm
  - INERCIAL: 7 momentos de Hu em escala log (hu1..hu7)
  - COR:      media e desvio de R,G,B e H,S,V dentro da mascara (12),
              histograma de matiz em 6 bins (hue_hist0..5)
  - TEXTURA:  GLCM contrast/homogeneity/energy/correlation/dissimilarity/ASM (6),
              histograma LBP uniforme em 10 bins (lbp0..9)
  - PODRIDAO: dark_ratio, brown_ratio, sat_std (indicadores de mancha/escurecimento)

Todas as estatisticas de cor/textura usam SOMENTE os pixels dentro da mascara,
para nao contaminar com o fundo branco.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, label as sk_label

# Parametros LBP
_LBP_P = 8
_LBP_R = 1
_LBP_BINS = _LBP_P + 2  # padrao 'uniform' -> P+2 bins


# --------------------------------------------------------------------------- #
# Forma                                                                        #
# --------------------------------------------------------------------------- #
def shape_features(mask: np.ndarray) -> dict:
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    img_area = float(H * W)
    diag = float(np.hypot(H, W))

    lab = sk_label(m)
    props = regionprops(lab)
    if not props:
        return {k: 0.0 for k in
                ["area_frac", "perimeter_norm", "eccentricity", "solidity",
                 "extent", "circularity", "aspect_ratio", "equiv_diam_norm"]}
    p = max(props, key=lambda r: r.area)   # maior regiao

    area = float(p.area)
    perim = float(p.perimeter) if p.perimeter > 0 else 1.0
    circularity = float(4.0 * np.pi * area / (perim ** 2))
    minr, minc, maxr, maxc = p.bbox
    bb_h, bb_w = (maxr - minr), (maxc - minc)
    aspect = float(max(bb_h, bb_w) / max(1, min(bb_h, bb_w)))

    return {
        "area_frac": area / img_area,
        "perimeter_norm": perim / diag,
        "eccentricity": float(p.eccentricity),
        "solidity": float(p.solidity),
        "extent": float(p.extent),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "equiv_diam_norm": float(p.equivalent_diameter_area) / diag,
    }


# --------------------------------------------------------------------------- #
# Momentos de Hu (inerciais)                                                   #
# --------------------------------------------------------------------------- #
def hu_features(mask: np.ndarray) -> dict:
    m = (mask > 0).astype(np.uint8) * 255
    moments = cv2.moments(m)
    hu = cv2.HuMoments(moments).flatten()
    # escala logaritmica com sinal (padrao para tornar comparavel)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return {f"hu{i+1}": float(hu_log[i]) for i in range(7)}


# --------------------------------------------------------------------------- #
# Cor                                                                          #
# --------------------------------------------------------------------------- #
def color_features(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    m = mask > 0
    feats: dict[str, float] = {}
    if m.sum() == 0:
        for c in "RGB":
            feats[f"mean_{c}"] = 0.0
            feats[f"std_{c}"] = 0.0
        for c in "HSV":
            feats[f"mean_{c}"] = 0.0
            feats[f"std_{c}"] = 0.0
        for i in range(6):
            feats[f"hue_hist{i}"] = 0.0
        return feats

    for i, c in enumerate("RGB"):
        vals = img_rgb[:, :, i][m].astype(np.float32)
        feats[f"mean_{c}"] = float(vals.mean())
        feats[f"std_{c}"] = float(vals.std())

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    for i, c in enumerate("HSV"):
        vals = hsv[:, :, i][m].astype(np.float32)
        feats[f"mean_{c}"] = float(vals.mean())
        feats[f"std_{c}"] = float(vals.std())

    # histograma de matiz (6 bins) usando a mascara
    hist = cv2.calcHist([hsv], [0], mask, [6], [0, 180]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    for i in range(6):
        feats[f"hue_hist{i}"] = float(hist[i])
    return feats


# --------------------------------------------------------------------------- #
# Textura                                                                      #
# --------------------------------------------------------------------------- #
def texture_features(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    m = mask > 0
    feats: dict[str, float] = {}

    # --- GLCM sobre o recorte do bounding box (fundo zerado) ---
    g = gray.copy()
    g[~m] = 0
    ys, xs = np.where(m)
    if len(ys) > 0:
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    levels = 32
    gq = (g.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    gq = np.clip(gq, 0, levels - 1)
    glcm = graycomatrix(gq, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation",
                 "dissimilarity", "ASM"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())

    # --- LBP (textura local) sobre pixels da fruta ---
    lbp = local_binary_pattern(gray, _LBP_P, _LBP_R, method="uniform")
    lbp_vals = lbp[m]
    hist, _ = np.histogram(lbp_vals, bins=_LBP_BINS, range=(0, _LBP_BINS),
                           density=True)
    for i in range(_LBP_BINS):
        feats[f"lbp{i}"] = float(hist[i])
    return feats


# --------------------------------------------------------------------------- #
# Indicadores de podridao                                                      #
# --------------------------------------------------------------------------- #
def rot_features(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Heuristicas diretamente ligadas a podridao: manchas escuras e marrons."""
    m = mask > 0
    if m.sum() == 0:
        return {"dark_ratio": 0.0, "brown_ratio": 0.0, "sat_std": 0.0}

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0][m].astype(np.float32)
    s = hsv[:, :, 1][m].astype(np.float32)
    v = hsv[:, :, 2][m].astype(np.float32)

    # fracao de pixels muito escuros (manchas pretas / podridao)
    dark_ratio = float((v < 60).mean())
    # fracao de pixels marrons (matiz alaranjado/marrom com saturacao media-baixa)
    brown = (h >= 8) & (h <= 28) & (s > 40) & (v < 150)
    brown_ratio = float(brown.mean())
    sat_std = float(s.std())
    return {"dark_ratio": dark_ratio, "brown_ratio": brown_ratio, "sat_std": sat_std}


# --------------------------------------------------------------------------- #
# Vetor completo                                                               #
# --------------------------------------------------------------------------- #
def extract_all(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    feats = {}
    feats.update(shape_features(mask))
    feats.update(hu_features(mask))
    feats.update(color_features(img_rgb, mask))
    feats.update(texture_features(img_rgb, mask))
    feats.update(rot_features(img_rgb, mask))
    return feats


# Grupos de features (usados na analise por grupos e no ablation study)
FEATURE_GROUPS = {
    "forma": ["area_frac", "perimeter_norm", "eccentricity", "solidity",
              "extent", "circularity", "aspect_ratio", "equiv_diam_norm"],
    "hu": [f"hu{i+1}" for i in range(7)],
    "cor": (["mean_R", "std_R", "mean_G", "std_G", "mean_B", "std_B",
             "mean_H", "std_H", "mean_S", "std_S", "mean_V", "std_V"]
            + [f"hue_hist{i}" for i in range(6)]),
    "textura": (["glcm_contrast", "glcm_homogeneity", "glcm_energy",
                 "glcm_correlation", "glcm_dissimilarity", "glcm_ASM"]
                + [f"lbp{i}" for i in range(_LBP_BINS)]),
    "podridao": ["dark_ratio", "brown_ratio", "sat_std"],
}


def all_feature_names() -> list[str]:
    names: list[str] = []
    for g in FEATURE_GROUPS.values():
        names.extend(g)
    return names
