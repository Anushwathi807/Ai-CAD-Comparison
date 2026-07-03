"""
Phase 8: Combined Visualization
=================================
Generates the final visual report by layering the semantic element matches
(Added, Removed, Modified, Moved) on top of the original drawing or heatmap.

Public API:
  create_combined_overlay(image, match_results, heatmap=None) → annotated image
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants & Styling
# ---------------------------------------------------------------------------
# Colors in BGR format
COLOR_ADDED = (0, 200, 0)      # Green
COLOR_REMOVED = (0, 0, 255)    # Red
COLOR_MODIFIED = (0, 215, 255) # Gold/Yellow
COLOR_MOVED = (255, 140, 0)    # Blue/Orange (using Blue-ish here: 255,140,0 is actually blue/cyan in BGR? No, B,G,R. 255,140,0 is Blue=255, Green=140, Red=0. Let's use Cyan)
COLOR_MOVED = (255, 255, 0)    # Cyan
COLOR_UNCHANGED = (150, 150, 150) # Gray

THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# Helper: Draw Tag
# ---------------------------------------------------------------------------

def draw_tagged_box(img: np.ndarray, bbox: tuple, label: str, 
                    color: tuple, thickness: int = THICKNESS):
    """Draws a bounding box with a background-filled text label on top."""
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Calculate font scale based on box size, with a min/max limit
    w, h = x2 - x1, y2 - y1
    font_scale = max(0.4, min(w, h) / 200.0)
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, 2)
    
    # Draw label background
    label_y1 = max(0, y1 - text_h - 10)
    cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 10, y1), color, -1)
    
    # Draw text
    # Choose black or white text depending on background brightness for contrast
    brightness = (color[2] * 299 + color[1] * 587 + color[0] * 114) / 1000
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    
    cv2.putText(img, label, (x1 + 5, y1 - 5), FONT, font_scale, text_color, 2)

# ---------------------------------------------------------------------------
# Main Visualization function
# ---------------------------------------------------------------------------

def create_combined_overlay(image_b: np.ndarray, 
                            match_results: dict, 
                            image_a: np.ndarray = None,
                            heatmap: np.ndarray = None, 
                            heatmap_alpha: float = 0.3) -> np.ndarray:
    """
    Creates the final combined visualization overlay.
    
    Parameters
    ----------
    image_b : np.ndarray
        The "AFTER" image (comparison). We primarily draw on this.
    match_results : dict
        Output from Phase 7 (match_elements).
    image_a : np.ndarray
        The "BEFORE" image (reference). Used to show where removed elements were.
    heatmap : np.ndarray
        Optional heatmap from Phase 4 to blend into the background.
    heatmap_alpha : float
        Transparency of the heatmap overlay.

    Returns
    -------
    np.ndarray — BGR annotated image
    """
    # Start with the base image B
    result = image_b.copy()

    # Blend heatmap if provided
    if heatmap is not None:
        result = cv2.addWeighted(result, 1.0 - heatmap_alpha, heatmap, heatmap_alpha, 0)

    # 1. Draw Removed elements (They exist in A, but not in B)
    # Since we are drawing on B, we'll draw them as "ghost" dashed boxes where they used to be,
    # or just solid red boxes.
    for item in match_results.get('removed', []):
        bbox = item['bbox']
        text = item.get('text', '')[:15] + "..." if len(item.get('text', '')) > 15 else item.get('text', '')
        if not text: text = item['class_name']
        label = f"REMOVED: {text}"
        draw_tagged_box(result, bbox, label, COLOR_REMOVED)

    # 2. Draw Modified elements
    for match in match_results.get('modified', []):
        bbox = match['new']['bbox']  # Draw at the new location
        t_old = match['text_old'][:10]
        t_new = match['text_new'][:10]
        label = f"MODIFIED: {t_old} -> {t_new}"
        draw_tagged_box(result, bbox, label, COLOR_MODIFIED)

    # 3. Draw Moved elements
    for match in match_results.get('moved', []):
        bbox = match['new']['bbox']
        label = f"MOVED (IoU: {match['iou']:.2f})"
        draw_tagged_box(result, bbox, label, COLOR_MOVED)

    # 4. Draw Added elements
    for item in match_results.get('added', []):
        bbox = item['bbox']
        text = item.get('text', '')[:15]
        if not text: text = item['class_name']
        label = f"ADDED: {text}"
        draw_tagged_box(result, bbox, label, COLOR_ADDED)

    # (Optional) We skip unchanged elements to reduce clutter, or draw them faintly
    # for match in match_results.get('unchanged', []):
    #     draw_tagged_box(result, match['new']['bbox'], "Unchanged", COLOR_UNCHANGED, thickness=1)

    return result

def create_before_overlay(image_a: np.ndarray, match_results: dict) -> np.ndarray:
    """Draws Removed, Modified (old loc), and Moved (old loc) on Image A."""
    result = image_a.copy()
    
    # Removed
    for item in match_results.get('removed', []):
        text = item.get('text', '')[:15]
        if not text: text = item['class_name']
        draw_tagged_box(result, item['bbox'], f"REMOVED: {text}", COLOR_REMOVED)
        
    # Modified
    for match in match_results.get('modified', []):
        draw_tagged_box(result, match['old']['bbox'], f"MODIFIED: {match['text_old'][:10]}", COLOR_MODIFIED)
        
    # Moved
    for match in match_results.get('moved', []):
        draw_tagged_box(result, match['old']['bbox'], "MOVED FROM HERE", COLOR_MOVED)
        
    return result

def create_after_overlay(image_b: np.ndarray, match_results: dict) -> np.ndarray:
    """Draws Added, Modified (new loc), and Moved (new loc) on Image B."""
    result = image_b.copy()
    
    # Added
    for item in match_results.get('added', []):
        text = item.get('text', '')[:15]
        if not text: text = item['class_name']
        draw_tagged_box(result, item['bbox'], f"ADDED: {text}", COLOR_ADDED)
        
    # Modified
    for match in match_results.get('modified', []):
        draw_tagged_box(result, match['new']['bbox'], f"MODIFIED: {match['text_new'][:10]}", COLOR_MODIFIED)
        
    # Moved
    for match in match_results.get('moved', []):
        draw_tagged_box(result, match['new']['bbox'], "MOVED TO HERE", COLOR_MOVED)
        
    return result

# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    import time
    from app.input_handler import load_drawing_pair
    from app.pixel_diff import run_pixel_diff
    from app.yolo_detector import load_model, detect_on_pair
    from app.ocr_engine import init_ocr, ocr_all_detections
    from app.element_matcher import match_elements

    model = load_model()
    ocr = init_ocr()

    pairs = [
        ('v1.pdf', 'v2.pdf', 'pair1'),
        ('v1-new.pdf', 'v2-new.pdf', 'pair2'),
    ]

    for file_a, file_b, label in pairs:
        print(f"\n{'='*60}")
        print(f"  COMBINED VISUALIZATION: {label}")
        print(f"{'='*60}")

        t0 = time.time()
        
        # 1. Input & Align
        pair = load_drawing_pair(file_a, file_b, align=True)
        
        # 2. Pixel Diff (for heatmap)
        print("[pipeline] Running Pixel Diff...")
        diff_res = run_pixel_diff(pair['gray_a'], pair['gray_b'], pair['img_a'], pair['img_b'])
        
        # 3. Detect
        print("[pipeline] Running YOLO Detection...")
        det_res = detect_on_pair(model, pair['img_a'], pair['img_b'])
        
        # 4. OCR
        print("[pipeline] Running OCR...")
        det_a = ocr_all_detections(ocr, pair['img_a'], det_res['detections_a'])
        det_b = ocr_all_detections(ocr, pair['img_b'], det_res['detections_b'])
        
        # 5. Match
        print("[pipeline] Running Element Matching...")
        match_res = match_elements(det_a, det_b)
        
        # 6. Visualize!
        print("[pipeline] Generating Final Overlay...")
        
        # Overlay with heatmap
        final_with_heatmap = create_combined_overlay(
            image_b=pair['img_b'], 
            match_results=match_res, 
            image_a=pair['img_a'],
            heatmap=diff_res['heatmap'],
            heatmap_alpha=0.4
        )
        
        # Overlay without heatmap (just clean boxes)
        final_clean = create_combined_overlay(
            image_b=pair['img_b'], 
            match_results=match_res, 
            image_a=pair['img_a']
        )
        
        elapsed = time.time() - t0
        print(f"[summary] Total Pipeline Time: {elapsed:.2f}s")
        
        out_dir = 'outputs'
        os.makedirs(out_dir, exist_ok=True)
        
        out_path_hm = f"{out_dir}/{label}_final_heatmap.png"
        out_path_cl = f"{out_dir}/{label}_final_clean.png"
        
        cv2.imwrite(out_path_hm, final_with_heatmap)
        cv2.imwrite(out_path_cl, final_clean)
        
        print(f"[saved] {out_path_hm}")
        print(f"[saved] {out_path_cl}")

    print(f"\n{'='*60}")
    print("  PHASE 8 TESTS COMPLETE ✅")
    print(f"{'='*60}")
