"""Find the comic paper in each photo; return its rotated rectangle."""
import cv2, numpy as np
from PIL import Image, ImageOps

def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def paper_mask(img, scale):
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)
    # paper is much brighter than the mat; the clear holder is mid-grey
    t, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = np.uint8(lab == k) * 255
    return small, m, t
