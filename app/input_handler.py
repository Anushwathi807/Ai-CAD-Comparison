"""
Phase 3: Data Prep & Input Handling
====================================
Loads PDF/JPG/PNG inputs, normalizes to same dimensions,
and optionally aligns using ORB feature matching + homography.

Supports:
  - File paths (str / pathlib.Path)
  - Raw bytes (for Streamlit file uploads)
  - PDF (single or multi-page — uses first page)
  - JPG / PNG / BMP / TIFF

Public API:
  load_drawing_pair(path_a, path_b, dpi=300, align=True)
    → dict with keys: img_a, img_b, gray_a, gray_b, meta
"""

import os
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_DPI = 300
MIN_ORB_MATCHES = 10  # Minimum good matches required for alignment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array."""
    rgb = np.array(pil_img)
    if len(rgb.shape) == 2:
        # Grayscale → convert to BGR
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bytes_to_cv2(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPG/PNG) to OpenCV BGR array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes — unsupported or corrupt data.")
    return img


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def pdf_to_image(source, dpi: int = DEFAULT_DPI, page: int = 1) -> np.ndarray:
    """
    Rasterize a PDF to an OpenCV BGR image.

    Parameters
    ----------
    source : str, Path, or bytes
        PDF file path or raw PDF bytes.
    dpi : int
        Rendering resolution (default 300).
    page : int
        Which page to extract (1-indexed, default 1).

    Returns
    -------
    np.ndarray  — BGR image (H, W, 3)
    """
    if isinstance(source, (str, Path)):
        source = str(source)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"PDF not found: {source}")
        pages = convert_from_path(source, dpi=dpi, first_page=page, last_page=page)
    elif isinstance(source, bytes):
        pages = convert_from_bytes(source, dpi=dpi, first_page=page, last_page=page)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if not pages:
        raise ValueError(f"No pages extracted from PDF (requested page {page}).")

    return _pil_to_cv2(pages[0])


# ---------------------------------------------------------------------------
# Generic image loading
# ---------------------------------------------------------------------------

def load_image(source, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """
    Load an image from a file path or raw bytes.
    Automatically detects PDF vs raster format.

    Parameters
    ----------
    source : str, Path, or bytes
        File path or raw file bytes.
    dpi : int
        DPI for PDF rasterization (ignored for raster images).

    Returns
    -------
    np.ndarray  — BGR image (H, W, 3)
    """
    # --- From file path ---
    if isinstance(source, (str, Path)):
        path = Path(source)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if ext == '.pdf':
            return pdf_to_image(source, dpi=dpi)

        # Raster image
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    # --- From raw bytes ---
    if isinstance(source, bytes):
        # Try PDF first (check magic bytes: %PDF)
        if source[:5] == b'%PDF-':
            return pdf_to_image(source, dpi=dpi)
        # Otherwise treat as raster
        return _bytes_to_cv2(source)

    raise TypeError(f"Unsupported source type: {type(source)}. Use str, Path, or bytes.")


# ---------------------------------------------------------------------------
# Normalization (resize pair to same dimensions)
# ---------------------------------------------------------------------------

def normalize_pair(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """
    Normalize a pair of images to the same dimensions.

    Strategy:
      - Preserve aspect ratio.
      - Downscale the larger image to match the smaller image's dimensions.
      - Pad the images with white borders so their exact dimensions match.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Downscale the larger image to prevent memory explosion and maintain aspect ratio
    if h_a * w_a > h_b * w_b:
        scale = min(w_b / w_a, h_b / h_a)
        new_w, new_h = int(w_a * scale), int(h_a * scale)
        img_a = cv2.resize(img_a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = min(w_a / w_b, h_a / h_b)
        new_w, new_h = int(w_b * scale), int(h_b * scale)
        img_b = cv2.resize(img_b, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to exact same dimensions
    target_h = max(img_a.shape[0], img_b.shape[0])
    target_w = max(img_a.shape[1], img_b.shape[1])

    def pad_image(img, tw, th):
        h, w = img.shape[:2]
        return cv2.copyMakeBorder(img, 0, th - h, 0, tw - w, cv2.BORDER_CONSTANT, value=[255,255,255])
        
    img_a = pad_image(img_a, target_w, target_h)
    img_b = pad_image(img_b, target_w, target_h)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    return {
        'img_a': img_a,
        'img_b': img_b,
        'gray_a': gray_a,
        'gray_b': gray_b,
        'original_size_a': (h_a, w_a),
        'original_size_b': (h_b, w_b),
        'target_size': (target_h, target_w),
    }


# ---------------------------------------------------------------------------
# Alignment (ORB feature matching + homography)
# ---------------------------------------------------------------------------

def align_images(img_ref: np.ndarray, img_cmp: np.ndarray,
                 gray_ref: np.ndarray = None, gray_cmp: np.ndarray = None,
                 max_features: int = 5000, good_match_ratio: float = 0.75) -> dict:
    """
    Align img_cmp to img_ref using ORB features + homography.

    Parameters
    ----------
    img_ref : np.ndarray
        Reference image (BGR). This stays fixed.
    img_cmp : np.ndarray
        Comparison image (BGR). This gets warped to match img_ref.
    gray_ref, gray_cmp : np.ndarray, optional
        Pre-computed grayscale versions. Computed if not provided.
    max_features : int
        Maximum ORB features to detect.
    good_match_ratio : float
        Lowe's ratio test threshold.

    Returns
    -------
    dict with keys:
        aligned_img   — warped BGR comparison image
        aligned_gray  — warped grayscale comparison image
        homography    — 3×3 transformation matrix (or None if alignment skipped)
        num_matches   — number of good matches found
        aligned       — bool, whether alignment was actually applied
    """
    if gray_ref is None:
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if gray_cmp is None:
        gray_cmp = cv2.cvtColor(img_cmp, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
    kp_cmp, desc_cmp = orb.detectAndCompute(gray_cmp, None)

    # Handle cases where features aren't found
    if desc_ref is None or desc_cmp is None or len(kp_ref) < 4 or len(kp_cmp) < 4:
        print("[align] Not enough features detected — skipping alignment.")
        return {
            'aligned_img': img_cmp.copy(),
            'aligned_gray': gray_cmp.copy(),
            'homography': None,
            'num_matches': 0,
            'aligned': False,
        }

    # BFMatcher with Hamming distance (for ORB binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc_cmp, desc_ref, k=2)

    # Lowe's ratio test
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

    num_good = len(good_matches)
    print(f"[align] ORB matches: {num_good} good / {len(raw_matches)} raw")

    if num_good < MIN_ORB_MATCHES:
        print(f"[align] Too few matches ({num_good} < {MIN_ORB_MATCHES}) — skipping alignment.")
        return {
            'aligned_img': img_cmp.copy(),
            'aligned_gray': gray_cmp.copy(),
            'homography': None,
            'num_matches': num_good,
            'aligned': False,
        }

    # Compute homography
    pts_cmp = np.float32([kp_cmp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_cmp, pts_ref, cv2.RANSAC, 5.0)

    if H is None:
        print("[align] Homography computation failed — skipping alignment.")
        return {
            'aligned_img': img_cmp.copy(),
            'aligned_gray': gray_cmp.copy(),
            'homography': None,
            'num_matches': num_good,
            'aligned': False,
        }

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"[align] Homography inliers: {inliers}/{num_good}")
    
    if inliers < 50:
        print(f"[align] Too few inliers ({inliers} < 50) — skipping alignment to prevent distortion.")
        return {
            'aligned_img': img_cmp.copy(),
            'aligned_gray': gray_cmp.copy(),
            'homography': None,
            'num_matches': num_good,
            'aligned': False,
        }

    # Warp comparison image to align with reference
    h, w = img_ref.shape[:2]
    aligned_img = cv2.warpPerspective(img_cmp, H, (w, h),
                                       borderMode=cv2.BORDER_REPLICATE)
    aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    return {
        'aligned_img': aligned_img,
        'aligned_gray': aligned_gray,
        'homography': H,
        'num_matches': num_good,
        'aligned': True,
    }


# ---------------------------------------------------------------------------
# Main convenience function
# ---------------------------------------------------------------------------

def load_drawing_pair(source_a, source_b, dpi: int = DEFAULT_DPI,
                      align: bool = True) -> dict:
    """
    Full pipeline: load two drawings → normalize → optionally align.

    Parameters
    ----------
    source_a : str, Path, or bytes
        Version A (reference / old drawing).
    source_b : str, Path, or bytes
        Version B (comparison / new drawing).
    dpi : int
        DPI for PDF rasterization.
    align : bool
        Whether to attempt ORB-based alignment.

    Returns
    -------
    dict with keys:
        img_a, img_b       — BGR images (same dimensions)
        gray_a, gray_b     — grayscale images (same dimensions)
        original_size_a     — (H, W) of raw loaded image A
        original_size_b     — (H, W) of raw loaded image B
        target_size         — (H, W) after normalization
        alignment_info      — dict from align_images() or None
    """
    print(f"[input] Loading image A...")
    raw_a = load_image(source_a, dpi=dpi)
    print(f"[input] Image A loaded: {raw_a.shape[1]}×{raw_a.shape[0]} px")

    print(f"[input] Loading image B...")
    raw_b = load_image(source_b, dpi=dpi)
    print(f"[input] Image B loaded: {raw_b.shape[1]}×{raw_b.shape[0]} px")

    # Normalize to same dimensions
    norm = normalize_pair(raw_a, raw_b)
    print(f"[input] Normalized to: {norm['target_size'][1]}×{norm['target_size'][0]} px")

    alignment_info = None
    if align:
        print("[input] Attempting alignment...")
        alignment_info = align_images(
            norm['img_a'], norm['img_b'],
            norm['gray_a'], norm['gray_b']
        )
        if alignment_info['aligned']:
            # Replace img_b and gray_b with aligned versions
            norm['img_b'] = alignment_info['aligned_img']
            norm['gray_b'] = alignment_info['aligned_gray']
            print("[input] Alignment applied successfully.")
        else:
            print("[input] Alignment skipped (not enough feature matches).")

    return {
        'img_a': norm['img_a'],
        'img_b': norm['img_b'],
        'gray_a': norm['gray_a'],
        'gray_b': norm['gray_b'],
        'original_size_a': norm['original_size_a'],
        'original_size_b': norm['original_size_b'],
        'target_size': norm['target_size'],
        'alignment_info': alignment_info,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import time

    pairs = [
        ('v1.pdf', 'v2.pdf', 'pair1'),
        ('v1-new.pdf', 'v2-new.pdf', 'pair2'),
    ]

    for file_a, file_b, label in pairs:
        if not os.path.isfile(file_a) or not os.path.isfile(file_b):
            print(f"\n⚠️  Skipping {label}: files not found")
            continue

        print(f"\n{'='*60}")
        print(f"  TESTING {label}: {file_a} ↔ {file_b}")
        print(f"{'='*60}")

        t0 = time.time()
        result = load_drawing_pair(file_a, file_b, dpi=300, align=True)
        elapsed = time.time() - t0

        print(f"\n[result] Processing time: {elapsed:.2f}s")
        print(f"[result] img_a shape: {result['img_a'].shape}")
        print(f"[result] img_b shape: {result['img_b'].shape}")
        print(f"[result] gray_a shape: {result['gray_a'].shape}")
        print(f"[result] gray_b shape: {result['gray_b'].shape}")
        print(f"[result] Original A: {result['original_size_a']}")
        print(f"[result] Original B: {result['original_size_b']}")
        print(f"[result] Target size: {result['target_size']}")

        if result['alignment_info']:
            ai = result['alignment_info']
            print(f"[result] Aligned: {ai['aligned']}")
            print(f"[result] Matches: {ai['num_matches']}")

        # Save side-by-side preview
        out_dir = 'outputs'
        os.makedirs(out_dir, exist_ok=True)

        # Save individual images
        cv2.imwrite(f'{out_dir}/{label}_img_a.png', result['img_a'])
        cv2.imwrite(f'{out_dir}/{label}_img_b.png', result['img_b'])

        # Create side-by-side comparison (scaled down for preview)
        scale = 0.3
        h, w = result['img_a'].shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        small_a = cv2.resize(result['img_a'], (new_w, new_h))
        small_b = cv2.resize(result['img_b'], (new_w, new_h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(small_a, 'Version A (Reference)', (10, 30),
                     font, 0.8, (0, 0, 255), 2)
        cv2.putText(small_b, 'Version B (Comparison)', (10, 30),
                     font, 0.8, (0, 0, 255), 2)

        side_by_side = np.hstack([small_a, small_b])
        cv2.imwrite(f'{out_dir}/{label}_side_by_side.png', side_by_side)
        print(f"[result] Saved: {out_dir}/{label}_side_by_side.png")

    print(f"\n{'='*60}")
    print("  ALL TESTS COMPLETE ✅")
    print(f"{'='*60}")
