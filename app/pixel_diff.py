"""
Phase 4: Pixel-Level Diff Engine
==================================
SSIM / absdiff → binary mask → bounding boxes → heatmap.

Satisfies FR-3 (visual difference output) and FR-4 (bounding boxes) baseline.

Public API:
  run_pixel_diff(gray_a, gray_b, img_a, img_b, ...)
    → dict with keys: ssim_score, diff_map, mask, regions, heatmap,
                       overlay_a, overlay_b, bbox_img_a, bbox_img_b
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_THRESH = 30          # Binary threshold for absdiff mask (0-255)
MIN_REGION_AREA = 100        # Minimum contour area to keep (filters noise)
MORPH_KERNEL_SIZE = 5        # Morphological kernel for mask cleanup
HEATMAP_COLORMAP = cv2.COLORMAP_JET   # Colormap for heatmap
HEATMAP_ALPHA = 0.4          # Overlay transparency (0=invisible, 1=opaque)
BBOX_COLOR_DEFAULT = (0, 0, 255)      # Red bounding boxes
BBOX_THICKNESS = 3
MAX_PIXELS_SSIM = 20_000_000  # Max pixels for SSIM (downscale if larger)


# ---------------------------------------------------------------------------
# SSIM Computation
# ---------------------------------------------------------------------------

def compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray,
                 win_size: int = 7) -> dict:
    """
    Compute Structural Similarity Index between two grayscale images.

    Parameters
    ----------
    gray_a, gray_b : np.ndarray
        Grayscale images of same shape (H, W).
    win_size : int
        SSIM window size (must be odd, ≤ min dimension).

    Returns
    -------
    dict with keys:
        score     — float, overall SSIM (1.0 = identical, 0.0 = completely different)
        diff_map  — np.ndarray (H, W), per-pixel SSIM values [0, 1]
        diff_inv  — np.ndarray (H, W) uint8, inverted diff (higher = more different)
    """
    # Ensure win_size doesn't exceed image dimensions
    min_dim = min(gray_a.shape[0], gray_a.shape[1])
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
    if win_size < 3:
        win_size = 3

    score, diff_map = ssim(gray_a, gray_b, full=True, win_size=win_size)

    # diff_map is float64 in [0, 1] where 1 = identical
    # Invert so higher values = more different, scale to uint8
    diff_inv = ((1.0 - diff_map) * 255).clip(0, 255).astype(np.uint8)

    return {
        'score': float(score),
        'diff_map': diff_map,
        'diff_inv': diff_inv,
    }


# ---------------------------------------------------------------------------
# Absolute Difference → Binary Mask
# ---------------------------------------------------------------------------

def compute_absdiff(gray_a: np.ndarray, gray_b: np.ndarray,
                    threshold: int = DEFAULT_THRESH) -> dict:
    """
    Compute absolute difference and threshold to binary mask.

    Parameters
    ----------
    gray_a, gray_b : np.ndarray
        Grayscale images of same shape.
    threshold : int
        Pixel intensity threshold (0-255). Pixels with diff > threshold → white.

    Returns
    -------
    dict with keys:
        raw_diff  — np.ndarray (H, W) uint8, raw absolute difference
        mask      — np.ndarray (H, W) uint8, binary mask (0 or 255)
        mask_clean — np.ndarray (H, W) uint8, morphologically cleaned mask
    """
    raw_diff = cv2.absdiff(gray_a, gray_b)

    _, mask = cv2.threshold(raw_diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    # Close small gaps
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove tiny specks
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    return {
        'raw_diff': raw_diff,
        'mask': mask,
        'mask_clean': mask_clean,
    }


# ---------------------------------------------------------------------------
# Contour Detection → Bounding Boxes
# ---------------------------------------------------------------------------

def find_diff_regions(mask: np.ndarray,
                      min_area: int = MIN_REGION_AREA,
                      merge_distance: int = 20) -> list:
    """
    Find contours in binary mask and extract bounding boxes.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) uint8, white = changed.
    min_area : int
        Minimum contour area to keep (filters noise).
    merge_distance : int
        Dilate mask by this amount before finding contours to merge
        nearby regions into single boxes.

    Returns
    -------
    list of dict, each with keys:
        bbox    — (x, y, w, h) bounding rectangle
        area    — contour area in pixels
        center  — (cx, cy) center of bounding box
        contour — np.ndarray, the raw contour points
    """
    # Dilate to merge nearby regions
    if merge_distance > 0:
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (merge_distance, merge_distance)
        )
        mask_merged = cv2.dilate(mask, merge_kernel, iterations=1)
    else:
        mask_merged = mask.copy()

    contours, _ = cv2.findContours(
        mask_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        regions.append({
            'bbox': (x, y, w, h),
            'area': int(area),
            'center': (cx, cy),
            'contour': cnt,
        })

    # Sort by area (largest first)
    regions.sort(key=lambda r: r['area'], reverse=True)

    return regions


# ---------------------------------------------------------------------------
# Heatmap Generation
# ---------------------------------------------------------------------------

def generate_heatmap(diff_intensity: np.ndarray,
                     colormap: int = HEATMAP_COLORMAP) -> np.ndarray:
    """
    Convert a grayscale difference map to a color heatmap.

    Parameters
    ----------
    diff_intensity : np.ndarray
        Grayscale image (H, W) uint8 where brighter = more different.
    colormap : int
        OpenCV colormap constant (default: COLORMAP_JET).

    Returns
    -------
    np.ndarray  — BGR heatmap image (H, W, 3)
    """
    # Normalize to full 0-255 range for best contrast
    if diff_intensity.max() > 0:
        normalized = cv2.normalize(diff_intensity, None, 0, 255, cv2.NORM_MINMAX)
    else:
        normalized = diff_intensity.copy()

    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), colormap)
    return heatmap


# ---------------------------------------------------------------------------
# Visualization: Bounding Boxes
# ---------------------------------------------------------------------------

def draw_bounding_boxes(img: np.ndarray, regions: list,
                        color: tuple = BBOX_COLOR_DEFAULT,
                        thickness: int = BBOX_THICKNESS,
                        label: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on an image for detected difference regions.

    Parameters
    ----------
    img : np.ndarray
        BGR image to draw on (will be copied).
    regions : list
        Output from find_diff_regions().
    color : tuple
        BGR color for boxes.
    thickness : int
        Line thickness.
    label : bool
        Whether to draw region number labels.

    Returns
    -------
    np.ndarray  — BGR image with boxes drawn.
    """
    result = img.copy()

    for i, region in enumerate(regions):
        x, y, w, h = region['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        if label:
            label_text = f"#{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(w, h) / 200)
            label_size = cv2.getTextSize(label_text, font, font_scale, 2)[0]

            # Background rectangle for label
            cv2.rectangle(result,
                          (x, y - label_size[1] - 10),
                          (x + label_size[0] + 6, y),
                          color, -1)
            cv2.putText(result, label_text,
                        (x + 3, y - 5),
                        font, font_scale, (255, 255, 255), 2)

    return result


# ---------------------------------------------------------------------------
# Visualization: Heatmap Overlay
# ---------------------------------------------------------------------------

def create_diff_overlay(img: np.ndarray, heatmap: np.ndarray,
                        mask: np.ndarray = None,
                        alpha: float = HEATMAP_ALPHA) -> np.ndarray:
    """
    Overlay a heatmap on the original image.

    If mask is provided, only overlay where mask is non-zero
    (so unchanged areas show the clean original).

    Parameters
    ----------
    img : np.ndarray
        Original BGR image.
    heatmap : np.ndarray
        BGR heatmap (same size as img).
    mask : np.ndarray, optional
        Binary mask — overlay only where mask > 0.
    alpha : float
        Blend factor (0 = only original, 1 = only heatmap).

    Returns
    -------
    np.ndarray  — BGR blended image.
    """
    result = img.copy()

    if mask is not None:
        # Dilate mask slightly for smoother overlay edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_3ch = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2BGR) / 255.0

        blended = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        result = (img * (1 - mask_3ch) + blended * mask_3ch).astype(np.uint8)
    else:
        result = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pixel_diff(gray_a: np.ndarray, gray_b: np.ndarray,
                   img_a: np.ndarray, img_b: np.ndarray,
                   threshold: int = DEFAULT_THRESH,
                   min_area: int = MIN_REGION_AREA,
                   merge_distance: int = 20) -> dict:
    """
    Full pixel-level diff pipeline.

    Parameters
    ----------
    gray_a, gray_b : np.ndarray
        Grayscale images (same shape).
    img_a, img_b : np.ndarray
        BGR color images (same shape).
    threshold : int
        Binary threshold for absdiff.
    min_area : int
        Minimum contour area to keep.
    merge_distance : int
        Dilation for merging nearby change regions.

    Returns
    -------
    dict with keys:
        ssim_score    — float, overall SSIM
        ssim_diff_inv — uint8 inverted SSIM diff map
        raw_diff      — uint8 absolute difference
        mask          — binary mask (raw)
        mask_clean    — binary mask (morphologically cleaned)
        regions       — list of detected change regions (bbox, area, center)
        heatmap       — color heatmap from SSIM diff
        overlay_a     — heatmap overlaid on img_a
        overlay_b     — heatmap overlaid on img_b
        bbox_img_a    — img_a with bounding boxes
        bbox_img_b    — img_b with bounding boxes
        num_regions   — number of detected change regions
        total_changed_area — total changed pixel area
        change_percentage  — % of image area that changed
    """
    h, w = gray_a.shape[:2]
    total_area = h * w

    print(f"[pixel_diff] Image size: {w}×{h} ({total_area:,} pixels)")

    # Step 1: SSIM (downscale if image is too large to avoid OOM)
    print("[pixel_diff] Computing SSIM...")
    if total_area > MAX_PIXELS_SSIM:
        scale_factor = (MAX_PIXELS_SSIM / total_area) ** 0.5
        ssim_h = int(h * scale_factor)
        ssim_w = int(w * scale_factor)
        print(f"[pixel_diff] Image too large for SSIM — downscaling to {ssim_w}×{ssim_h}")
        gray_a_small = cv2.resize(gray_a, (ssim_w, ssim_h), interpolation=cv2.INTER_AREA)
        gray_b_small = cv2.resize(gray_b, (ssim_w, ssim_h), interpolation=cv2.INTER_AREA)
        ssim_result = compute_ssim(gray_a_small, gray_b_small)
        # Upscale SSIM diff map back to original resolution
        ssim_result['diff_inv'] = cv2.resize(
            ssim_result['diff_inv'], (w, h), interpolation=cv2.INTER_LINEAR
        )
    else:
        ssim_result = compute_ssim(gray_a, gray_b)
    print(f"[pixel_diff] SSIM score: {ssim_result['score']:.4f}")

    # Step 2: Absolute difference → mask
    print(f"[pixel_diff] Computing absdiff (threshold={threshold})...")
    absdiff_result = compute_absdiff(gray_a, gray_b, threshold=threshold)

    # Step 3: Find change regions
    print(f"[pixel_diff] Finding diff regions (min_area={min_area})...")
    regions = find_diff_regions(
        absdiff_result['mask_clean'],
        min_area=min_area,
        merge_distance=merge_distance
    )
    print(f"[pixel_diff] Found {len(regions)} change regions")

    # Calculate changed area
    total_changed = int(np.sum(absdiff_result['mask_clean'] > 0))
    change_pct = (total_changed / total_area) * 100 if total_area > 0 else 0

    # Step 4: Heatmap (from SSIM diff, which is perceptually better)
    print("[pixel_diff] Generating heatmap...")
    heatmap = generate_heatmap(ssim_result['diff_inv'])

    # Step 5: Overlays
    print("[pixel_diff] Creating overlays...")
    overlay_a = create_diff_overlay(
        img_a, heatmap, mask=absdiff_result['mask_clean']
    )
    overlay_b = create_diff_overlay(
        img_b, heatmap, mask=absdiff_result['mask_clean']
    )

    # Step 6: Bounding boxes
    print("[pixel_diff] Drawing bounding boxes...")
    bbox_img_a = draw_bounding_boxes(img_a, regions)
    bbox_img_b = draw_bounding_boxes(img_b, regions)

    # Summary
    print(f"[pixel_diff] Changed area: {total_changed:,} px ({change_pct:.2f}%)")
    for i, r in enumerate(regions):
        x, y, rw, rh = r['bbox']
        print(f"  Region #{i+1}: ({x},{y}) {rw}×{rh} area={r['area']:,}px")

    return {
        'ssim_score': ssim_result['score'],
        'ssim_diff_inv': ssim_result['diff_inv'],
        'raw_diff': absdiff_result['raw_diff'],
        'mask': absdiff_result['mask'],
        'mask_clean': absdiff_result['mask_clean'],
        'regions': regions,
        'heatmap': heatmap,
        'overlay_a': overlay_a,
        'overlay_b': overlay_b,
        'bbox_img_a': bbox_img_a,
        'bbox_img_b': bbox_img_b,
        'num_regions': len(regions),
        'total_changed_area': total_changed,
        'change_percentage': change_pct,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    import time
    from app.input_handler import load_drawing_pair

    pairs = [
        ('v1.pdf', 'v2.pdf', 'pair1'),
        ('v1-new.pdf', 'v2-new.pdf', 'pair2'),
    ]

    for file_a, file_b, label in pairs:
        if not os.path.isfile(file_a) or not os.path.isfile(file_b):
            print(f"\n⚠️  Skipping {label}: files not found")
            continue

        print(f"\n{'='*60}")
        print(f"  PIXEL DIFF: {label} ({file_a} ↔ {file_b})")
        print(f"{'='*60}")

        # Load and prepare
        t0 = time.time()
        pair = load_drawing_pair(file_a, file_b, dpi=300, align=True)

        # Run pixel diff
        result = run_pixel_diff(
            pair['gray_a'], pair['gray_b'],
            pair['img_a'], pair['img_b']
        )
        elapsed = time.time() - t0

        print(f"\n[summary] Total time: {elapsed:.2f}s")
        print(f"[summary] SSIM: {result['ssim_score']:.4f}")
        print(f"[summary] Regions: {result['num_regions']}")
        print(f"[summary] Changed: {result['change_percentage']:.2f}%")

        # Save outputs
        out_dir = 'outputs'
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(f'{out_dir}/{label}_heatmap.png', result['heatmap'])
        cv2.imwrite(f'{out_dir}/{label}_mask.png', result['mask_clean'])
        cv2.imwrite(f'{out_dir}/{label}_bbox_a.png', result['bbox_img_a'])
        cv2.imwrite(f'{out_dir}/{label}_bbox_b.png', result['bbox_img_b'])
        cv2.imwrite(f'{out_dir}/{label}_overlay_a.png', result['overlay_a'])
        cv2.imwrite(f'{out_dir}/{label}_overlay_b.png', result['overlay_b'])

        # Create a 2×3 composite panel
        scale = 0.25
        h, w = result['heatmap'].shape[:2]
        nw, nh = int(w * scale), int(h * scale)

        def resize(img):
            return cv2.resize(img, (nw, nh))

        # Row 1: Original A | Original B | Heatmap
        row1 = np.hstack([resize(pair['img_a']), resize(pair['img_b']),
                           resize(result['heatmap'])])
        # Row 2: BBox A | BBox B | Mask (as BGR)
        mask_bgr = cv2.cvtColor(result['mask_clean'], cv2.COLOR_GRAY2BGR)
        row2 = np.hstack([resize(result['bbox_img_a']),
                           resize(result['bbox_img_b']),
                           resize(mask_bgr)])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels_r1 = ['Original A', 'Original B', 'SSIM Heatmap']
        labels_r2 = ['BBoxes on A', 'BBoxes on B', 'Binary Mask']
        for i, lbl in enumerate(labels_r1):
            cv2.putText(row1, lbl, (i * nw + 10, 25), font, 0.7, (0, 0, 255), 2)
        for i, lbl in enumerate(labels_r2):
            cv2.putText(row2, lbl, (i * nw + 10, 25), font, 0.7, (0, 0, 255), 2)

        composite = np.vstack([row1, row2])
        cv2.imwrite(f'{out_dir}/{label}_composite.png', composite)
        print(f"[saved] {out_dir}/{label}_composite.png")

    print(f"\n{'='*60}")
    print("  ALL PIXEL DIFF TESTS COMPLETE ✅")
    print(f"{'='*60}")
