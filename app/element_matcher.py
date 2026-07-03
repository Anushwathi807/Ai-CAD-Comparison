"""
Phase 7: Element-Level Matching Engine
========================================
Matches OCR-enriched YOLO detections between two drawings
using the Hungarian algorithm to classify changes:
Added, Removed, Modified, Moved, or Unchanged.

Public API:
  match_elements(detections_a, detections_b) → dict of matched categories
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IOU_THRESHOLD_MOVED = 0.8     # Below this IoU, a matched element is considered "Moved"
MAX_COST_MATCH = 1.0          # Maximum cost to allow a match. Cost > this = unmatched

# ---------------------------------------------------------------------------
# Helper Metrics
# ---------------------------------------------------------------------------

def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute Intersection over Union between two bounding boxes (x1, y1, x2, y2)."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return float(inter_area) / float(union_area)

def compute_center_dist(c1: tuple, c2: tuple) -> float:
    """Euclidean distance between two centers."""
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

# ---------------------------------------------------------------------------
# Matching Algorithm
# ---------------------------------------------------------------------------

def match_elements(detections_a: list, detections_b: list) -> dict:
    """
    Match elements between A and B and categorize changes.

    Returns dict:
      'added': list of elements in B not in A
      'removed': list of elements in A not in B
      'modified': list of matched elements with text changes
      'moved': list of matched elements with position changes (IoU < threshold)
      'unchanged': list of identical matched elements
    """
    # Prepare result structures
    results = {
        'added': [],
        'removed': [],
        'modified': [],
        'moved': [],
        'unchanged': []
    }

    n_a = len(detections_a)
    n_b = len(detections_b)

    # Edge cases
    if n_a == 0:
        results['added'] = detections_b.copy()
        return results
    if n_b == 0:
        results['removed'] = detections_a.copy()
        return results

    # Build cost matrix
    # Cost ranges from 0 (perfect match) to >1 (bad match)
    cost_matrix = np.zeros((n_a, n_b), dtype=np.float32)

    # Normalization factor for distance (approximate diagonal of a typical cropped region or just a constant)
    dist_norm = 100.0 

    for i, a in enumerate(detections_a):
        for j, b in enumerate(detections_b):
            # Infinite cost if classes don't match (we only match Notes to Notes, Dimension to Dimension)
            if a['class_id'] != b['class_id']:
                cost_matrix[i, j] = 999.0
                continue

            iou = compute_iou(a['bbox'], b['bbox'])
            dist = compute_center_dist(a['center'], b['center'])
            
            # IoU cost: 1.0 (no overlap) to 0.0 (perfect overlap)
            iou_cost = 1.0 - iou
            
            # Distance cost: 0.0 (same center) up to 1.0 (100px away)
            dist_cost = min(1.0, dist / dist_norm)
            
            # Text exact match bonus (reduces cost)
            text_cost = 0.0
            if a['text'] != b['text'] and a['has_text'] and b['has_text']:
                text_cost = 0.5 # Penalty for text change

            # Combined cost
            cost = (iou_cost * 0.6) + (dist_cost * 0.4) + text_cost
            cost_matrix[i, j] = cost

    # Solve matching using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Keep track of which indices were matched
    matched_a = set()
    matched_b = set()

    for i, j in zip(row_ind, col_ind):
        cost = cost_matrix[i, j]
        
        # If cost is too high, it's not a valid match
        if cost > MAX_COST_MATCH:
            continue
            
        matched_a.add(i)
        matched_b.add(j)
        
        a = detections_a[i]
        b = detections_b[j]
        
        # Categorize the match
        iou = compute_iou(a['bbox'], b['bbox'])
        text_changed = (a['has_text'] or b['has_text']) and (a['text'] != b['text'])
        
        match_info = {
            'old': a,
            'new': b,
            'iou': round(iou, 4),
            'text_old': a['text'],
            'text_new': b['text']
        }
        
        if text_changed:
            results['modified'].append(match_info)
        elif iou < IOU_THRESHOLD_MOVED:
            results['moved'].append(match_info)
        else:
            results['unchanged'].append(match_info)

    # Find unmatched elements
    for i in range(n_a):
        if i not in matched_a:
            results['removed'].append(detections_a[i])
            
    for j in range(n_b):
        if j not in matched_b:
            results['added'].append(detections_b[j])

    return results

# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from app.input_handler import load_drawing_pair
    from app.yolo_detector import load_model, detect_on_pair
    from app.ocr_engine import init_ocr, ocr_all_detections

    # Initialize models
    model = load_model()
    ocr = init_ocr()

    pairs = [
        ('v1.pdf', 'v2.pdf', 'pair1'),
        ('v1-new.pdf', 'v2-new.pdf', 'pair2'),
    ]

    for file_a, file_b, label in pairs:
        print(f"\n{'='*60}")
        print(f"  ELEMENT MATCHING: {label} ({file_a} ↔ {file_b})")
        print(f"{'='*60}")

        # 1. Load and Align
        pair = load_drawing_pair(file_a, file_b, align=True)
        
        # 2. YOLO Detect
        det = detect_on_pair(model, pair['img_a'], pair['img_b'])
        
        # 3. OCR Envelop
        det_a = ocr_all_detections(ocr, pair['img_a'], det['detections_a'])
        det_b = ocr_all_detections(ocr, pair['img_b'], det['detections_b'])
        
        # 4. Element Matching
        results = match_elements(det_a, det_b)
        
        print("\n[MATCHING SUMMARY]")
        print(f"  Added:     {len(results['added'])}")
        print(f"  Removed:   {len(results['removed'])}")
        print(f"  Modified:  {len(results['modified'])}")
        print(f"  Moved:     {len(results['moved'])}")
        print(f"  Unchanged: {len(results['unchanged'])}")
        
        # Print details
        if results['added']:
            print("\n  [ADDED]")
            for x in results['added']:
                t = x['text'] if x['has_text'] else "(no text)"
                print(f"    + {x['class_name']} @ {x['center']}: '{t}'")
                
        if results['removed']:
            print("\n  [REMOVED]")
            for x in results['removed']:
                t = x['text'] if x['has_text'] else "(no text)"
                print(f"    - {x['class_name']} @ {x['center']}: '{t}'")
                
        if results['modified']:
            print("\n  [MODIFIED (Text changed)]")
            for x in results['modified']:
                print(f"    ~ {x['old']['class_name']} @ {x['old']['center']}: '{x['text_old']}' -> '{x['text_new']}'")
                
        if results['moved']:
            print("\n  [MOVED]")
            for x in results['moved']:
                t = x['text_old'] if x['old']['has_text'] else "(no text)"
                print(f"    > {x['old']['class_name']}: '{t}' moved from {x['old']['center']} to {x['new']['center']} (IoU: {x['iou']})")
