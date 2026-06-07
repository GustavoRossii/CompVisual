from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------- #
# Pós-processamento comum                                                      #
# --------------------------------------------------------------------------- #
def _postprocess(mask: np.ndarray) -> np.ndarray:
    """Limpeza morfológica + preenchimento + maior componente conectado."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return (mask * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # preenche buracos internos (reflexos, manchas claras dentro da fruta)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    # mantém apenas o maior componente conectado
    n, lab = cv2.connectedComponents(mask)
    if n > 1:
        sizes = [(lab == i).sum() for i in range(1, n)]
        biggest = 1 + int(np.argmax(sizes))
        mask = (lab == biggest).astype(np.uint8)

    return (mask * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Método A - Otsu                                                              #
# --------------------------------------------------------------------------- #
def segment_otsu(img_rgb: np.ndarray) -> np.ndarray:
    """Segmentação por limiar de Otsu sobre tons de cinza."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY_INV: fruta (escura) vira 255, fundo (claro) vira 0
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _postprocess(mask)


# --------------------------------------------------------------------------- #
# Método B - Cor em HSV                                                        #
# --------------------------------------------------------------------------- #
def segment_hsv(img_rgb: np.ndarray,
                s_thresh: int = 35,
                v_high: int = 235,
                v_low: int = 30) -> np.ndarray:
    
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_rgb, (5, 5), 0), cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    background = (s < s_thresh) & ((v > v_high) | (v < v_low))
    mask = (~background).astype(np.uint8) * 255
    return _postprocess(mask)


SEGMENTERS = {"otsu": segment_otsu, "hsv": segment_hsv}


def segment(img_rgb: np.ndarray, method: str = "hsv") -> np.ndarray:
    return SEGMENTERS[method](img_rgb)


def apply_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(mask, (out.shape[1], out.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    out[mask == 0] = 0
    return out


def mask_coverage(mask: np.ndarray) -> float:
    return float((mask > 0).mean())
