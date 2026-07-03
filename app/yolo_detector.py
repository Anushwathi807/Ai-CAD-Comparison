"""
Phase 5: YOLOv8 Detector — Inference Module
=============================================
Loads the trained YOLOv8 model and detects 'Notes' and 'dimension'
regions in CAD drawing images.

Classes:
  0 = Notes
  1 = dimension

Public API:
  load_model(weights_path)  → YOLO model
  detect(model, image, conf=0.25)  → list of detections
  detect_on_pair(model, img_a, img_b)  → dict with detections for both
  draw_detections(image, detections)  → annotated image
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'models', 'best.pt')
CLASS_NAMES = {0: 'Notes', 1: 'dimension'}
CLASS_COLORS = {
    'Notes': (255, 140, 0),       # Orange (BGR)
    'dimension': (0, 200, 0),     # Green (BGR)
}
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str = None, device: str = 'auto') -> YOLO:
    """
    Load a trained YOLOv8 model.

    Parameters
    ----------
    weights_path : str, optional
        Path to .pt weights file. Uses default if not provided.
    device : str
        'auto', 'cpu', '0' (GPU 0), etc.

    Returns
    -------
    YOLO model instance
    """
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Run Phase 5 training first, or check the path."
        )

    model = YOLO(weights_path)
    print(f"[yolo] Model loaded: {weights_path}")
    print(f"[yolo] Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(model: YOLO, image: np.ndarray,
           conf: float = DEFAULT_CONF,
           iou: float = DEFAULT_IOU) -> list:
    """
    Run object detection on a single image.

    Parameters
    ----------
    model : YOLO
        Loaded YOLOv8 model.
    image : np.ndarray
        BGR image (H, W, 3).
    conf : float
        Confidence threshold (0-1).
    iou : float
        IoU threshold for NMS.

    Returns
    -------
    list of dict, each with:
        bbox       — (x1, y1, x2, y2) pixel coordinates
        class_id   — int (0=Notes, 1=dimension)
        class_name — str
        confidence — float
        center     — (cx, cy)
        width      — int
        height     — int
        area       — int
    """
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            cls_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            w = int(x2 - x1)
            h = int(y2 - y1)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': round(confidence, 4),
                'center': (cx, cy),
                'width': w,
                'height': h,
                'area': w * h,
            })

    # Sort by confidence (highest first)
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Pair Detection
# ---------------------------------------------------------------------------

def detect_on_pair(model: YOLO, img_a: np.ndarray, img_b: np.ndarray,
                   conf: float = DEFAULT_CONF) -> dict:
    """
    Run detection on both images of a drawing pair.

    Returns
    -------
    dict with:
        detections_a  — list of detections for image A
        detections_b  — list of detections for image B
        count_a       — total detections in A
        count_b       — total detections in B
        by_class_a    — dict of class_name → count for A
        by_class_b    — dict of class_name → count for B
    """
    print("[yolo] Detecting on image A...")
    det_a = detect(model, img_a, conf=conf)
    print(f"[yolo] Image A: {len(det_a)} detections")

    print("[yolo] Detecting on image B...")
    det_b = detect(model, img_b, conf=conf)
    print(f"[yolo] Image B: {len(det_b)} detections")

    def count_by_class(dets):
        counts = {}
        for d in dets:
            name = d['class_name']
            counts[name] = counts.get(name, 0) + 1
        return counts

    by_class_a = count_by_class(det_a)
    by_class_b = count_by_class(det_b)

    print(f"[yolo] Image A by class: {by_class_a}")
    print(f"[yolo] Image B by class: {by_class_b}")

    return {
        'detections_a': det_a,
        'detections_b': det_b,
        'count_a': len(det_a),
        'count_b': len(det_b),
        'by_class_a': by_class_a,
        'by_class_b': by_class_b,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_detections(image: np.ndarray, detections: list,
                    thickness: int = 2) -> np.ndarray:
    """
    Draw detection bounding boxes with class labels on an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (will be copied).
    detections : list
        Output from detect().
    thickness : int
        Box line thickness.

    Returns
    -------
    np.ndarray — annotated BGR image
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class_name']
        conf = det['confidence']
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"{cls_name} {conf:.2f}"
        font_scale = max(0.4, min(det['width'], det['height']) / 300)
        label_size = cv2.getTextSize(label, font, font_scale, 1)[0]

        # Label background
        label_y = max(y1 - 5, label_size[1] + 5)
        cv2.rectangle(result,
                      (x1, label_y - label_size[1] - 5),
                      (x1 + label_size[0] + 4, label_y + 2),
                      color, -1)
        cv2.putText(result, label,
                    (x1 + 2, label_y - 2),
                    font, font_scale, (0, 0, 0), 1)

    return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from app.input_handler import load_drawing_pair

    # Load model
    model = load_model()

    pairs = [
        ('v1.pdf', 'v2.pdf', 'pair1'),
        ('v1-new.pdf', 'v2-new.pdf', 'pair2'),
    ]

    for file_a, file_b, label in pairs:
        if not os.path.isfile(file_a) or not os.path.isfile(file_b):
            print(f"\n⚠️  Skipping {label}: files not found")
            continue

        print(f"\n{'='*60}")
        print(f"  YOLO DETECTION: {label} ({file_a} ↔ {file_b})")
        print(f"{'='*60}")

        t0 = time.time()
        pair = load_drawing_pair(file_a, file_b, dpi=300, align=True)

        result = detect_on_pair(model, pair['img_a'], pair['img_b'])
        elapsed = time.time() - t0

        print(f"\n[summary] Time: {elapsed:.2f}s")
        print(f"[summary] Detections A: {result['count_a']}")
        print(f"[summary] Detections B: {result['count_b']}")

        # Print detailed detections
        for side, dets in [('A', result['detections_a']),
                           ('B', result['detections_b'])]:
            print(f"\n  --- Image {side} ---")
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d['bbox']
                print(f"  [{i+1}] {d['class_name']:>10s} "
                      f"conf={d['confidence']:.3f}  "
                      f"box=({x1},{y1})-({x2},{y2})  "
                      f"{d['width']}×{d['height']}")

        # Save annotated images
        out_dir = 'outputs'
        os.makedirs(out_dir, exist_ok=True)

        ann_a = draw_detections(pair['img_a'], result['detections_a'])
        ann_b = draw_detections(pair['img_b'], result['detections_b'])
        cv2.imwrite(f'{out_dir}/{label}_yolo_a.png', ann_a)
        cv2.imwrite(f'{out_dir}/{label}_yolo_b.png', ann_b)

        # Side-by-side at reduced scale
        scale = 0.3
        h, w = ann_a.shape[:2]
        nw, nh = int(w * scale), int(h * scale)
        small_a = cv2.resize(ann_a, (nw, nh))
        small_b = cv2.resize(ann_b, (nw, nh))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(small_a, f'A: {result["count_a"]} detections',
                     (10, 30), font, 0.8, (0, 0, 255), 2)
        cv2.putText(small_b, f'B: {result["count_b"]} detections',
                     (10, 30), font, 0.8, (0, 0, 255), 2)

        side = np.hstack([small_a, small_b])
        cv2.imwrite(f'{out_dir}/{label}_yolo_side.png', side)
        print(f"\n[saved] {out_dir}/{label}_yolo_side.png")

    print(f"\n{'='*60}")
    print("  ALL YOLO TESTS COMPLETE ✅")
    print(f"{'='*60}")
