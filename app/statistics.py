"""
Phase 9: Difference Statistics
================================
Aggregates quantitative data from previous phases (pixel-level stats and
element-level semantic matches) into a structured JSON/dict. Also computes
'Unlabeled Geometry Changes' (pixel changes that have no corresponding YOLO/OCR text).

Public API:
  compute_statistics(pixel_diff_results, match_results) → dict
"""

import numpy as np

def compute_statistics(pixel_diff_results: dict, match_results: dict) -> dict:
    """
    Compute comprehensive statistics for the drawing comparison.
    
    Parameters
    ----------
    pixel_diff_results : dict
        Output from Phase 4 (run_pixel_diff), must contain 'diff_regions'.
    match_results : dict
        Output from Phase 7 (match_elements), containing 'added', 'removed', etc.
        
    Returns
    -------
    dict
        Structured statistics including counts, percentages, and unlabeled geometry changes.
    """
    
    # 1. Pixel-Level Stats
    ssim_val = pixel_diff_results.get('ssim_score', 0.0)
    percent_changed = pixel_diff_results.get('change_percentage', 0.0)
    total_regions = pixel_diff_results.get('num_regions', 0)
    
    # 2. Element-Level Stats
    count_added = len(match_results.get('added', []))
    count_removed = len(match_results.get('removed', []))
    count_modified = len(match_results.get('modified', []))
    count_moved = len(match_results.get('moved', []))
    
    total_semantic_changes = count_added + count_removed + count_modified + count_moved
    
    # 3. Find Unlabeled Geometry Changes
    # These are pixel diff regions that DO NOT intersect with ANY semantic bounding box
    unlabeled_geom = []
    
    # Extract all bounding boxes from semantic matches (both old and new locations)
    semantic_boxes = []
    
    for item in match_results.get('added', []):
        semantic_boxes.append(item['bbox'])
    for item in match_results.get('removed', []):
        semantic_boxes.append(item['bbox'])
    for item in match_results.get('modified', []):
        semantic_boxes.append(item['old']['bbox'])
        semantic_boxes.append(item['new']['bbox'])
    for item in match_results.get('moved', []):
        semantic_boxes.append(item['old']['bbox'])
        semantic_boxes.append(item['new']['bbox'])
    for item in match_results.get('unchanged', []):
        semantic_boxes.append(item['old']['bbox'])
        semantic_boxes.append(item['new']['bbox'])
        
    # Check each pixel diff region against semantic boxes
    for region in pixel_diff_results.get('regions', []):
        rx, ry, rw, rh = region['bbox']
        r_box = (rx, ry, rx + rw, ry + rh)
        
        has_overlap = False
        for s_box in semantic_boxes:
            # Check for intersection
            x1_inter = max(r_box[0], s_box[0])
            y1_inter = max(r_box[1], s_box[1])
            x2_inter = min(r_box[2], s_box[2])
            y2_inter = min(r_box[3], s_box[3])
            
            if x2_inter > x1_inter and y2_inter > y1_inter:
                has_overlap = True
                break
                
        if not has_overlap:
            unlabeled_geom.append({
                'center': region['center'],
                'area': region['area'],
                'bbox': r_box
            })
            
    # Sort unlabeled geometry by area (largest first)
    unlabeled_geom.sort(key=lambda x: x['area'], reverse=True)

    # 4. Compile Final Stats
    stats = {
        'pixel_metrics': {
            'ssim_score': round(ssim_val, 4),
            'percent_changed': round(percent_changed, 2),
            'total_pixel_regions': total_regions
        },
        'element_metrics': {
            'added': count_added,
            'removed': count_removed,
            'modified': count_modified,
            'moved': count_moved,
            'total_semantic_changes': total_semantic_changes
        },
        'unlabeled_geometry_changes': {
            'count': len(unlabeled_geom),
            'regions': unlabeled_geom[:10]  # Only store top 10 largest to avoid huge payloads
        },
        'raw_match_data': match_results  # Pass-through for the summarizer
    }
    
    return stats


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from app.input_handler import load_drawing_pair
    from app.pixel_diff import run_pixel_diff
    from app.yolo_detector import load_model, detect_on_pair
    from app.ocr_engine import init_ocr, ocr_all_detections
    from app.element_matcher import match_elements
    import json
    import time

    model = load_model()
    ocr = init_ocr()

    file_a = 'v1.pdf'
    file_b = 'v2.pdf'

    print("\nRunning full pipeline to generate statistics...")
    t0 = time.time()
    
    pair = load_drawing_pair(file_a, file_b, align=True)
    diff_res = run_pixel_diff(pair['gray_a'], pair['gray_b'], pair['img_a'], pair['img_b'])
    det_res = detect_on_pair(model, pair['img_a'], pair['img_b'])
    det_a = ocr_all_detections(ocr, pair['img_a'], det_res['detections_a'])
    det_b = ocr_all_detections(ocr, pair['img_b'], det_res['detections_b'])
    match_res = match_elements(det_a, det_b)
    
    stats = compute_statistics(diff_res, match_res)
    
    elapsed = time.time() - t0
    
    print("\n" + "="*60)
    print("  PHASE 9: STATISTICS RESULTS")
    print("="*60)
    print(f"Time taken: {elapsed:.2f}s\n")
    
    # We remove the raw data before printing so it's readable
    del stats['raw_match_data']
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*60)
    print("  PHASE 9 TEST COMPLETE ✅")
    print("="*60)
