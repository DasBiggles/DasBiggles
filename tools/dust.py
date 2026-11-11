"""Find small bright specks on the dark mat (never on paper or the clear holder)."""
import cv2, numpy as np

def dust_mask(img, protect):
    """img: full-res BGR. protect: uint8 mask (255 = never touch)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local background level (what the mat looks like around each pixel)
    bg = cv2.medianBlur(gray, 31)
    tophat = cv2.subtract(gray, bg)
    # noise-adaptive threshold: a speck must stand well clear of the mat's grain
    dark = bg < 70
    vals = tophat[dark].astype(np.float32)
    signed = gray[dark].astype(np.float32) - bg[dark]
    mad = np.median(np.abs(signed - np.median(signed))) * 1.4826 if vals.size else 5
    thr = max(25.0, 7.0 * mad)
    cand = np.uint8((tophat > thr) & dark) * 255
    # anything bright and big (paper, holder, its edges) is protected, with a margin
    big = np.uint8(cv2.GaussianBlur(gray, (0, 0), 2) > 80) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(big)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] > 1500
    prot = np.uint8(keep[lab]) * 255 | protect
    prot = cv2.dilate(prot, np.ones((41, 41), np.uint8))
    cand[prot > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    ok = np.zeros(n, bool)
    a = st[:, cv2.CC_STAT_AREA]; w = st[:, cv2.CC_STAT_WIDTH]; h = st[:, cv2.CC_STAT_HEIGHT]
    ok[1:] = (a[1:] >= 4) & (a[1:] <= 900) & (np.maximum(w, h)[1:] <= 60)
    m = np.uint8(ok[lab]) * 255
    return cv2.dilate(m, np.ones((7, 7), np.uint8)), int(ok.sum())

def remove_dust(img, protect):
    m, n = dust_mask(img, protect)
    if n == 0:
        return img, m, 0
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA), m, n
