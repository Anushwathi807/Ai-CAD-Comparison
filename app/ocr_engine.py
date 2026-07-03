"""
Phase 6: OCR on Detected Regions
==================================
Uses PaddleOCR to read text inside each YOLO-detected bounding box.
Enriches detections with text content for element-level matching in Phase 7.

Public API:
  init_ocr(lang='en')  → PaddleOCR instance
  ocr_region(ocr, image, bbox)  → dict with text, confidence
  ocr_all_detections(ocr, image, detections)  → enriched detections list
  ocr_drawing(ocr, model, image)  → full pipeline (detect + OCR)
"""

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_PADDING = 5       # Pixels of padding around bbox when cropping
MIN_OCR_CONF = 0.3     # Minimum OCR confidence to keep a text line
_ocr_instance = None   # Singleton OCR instance


# ---------------------------------------------------------------------------
# OCR Initialization
# ---------------------------------------------------------------------------

def init_ocr(lang: str = 'en', use_gpu: bool = False):
    """
    Initialize PaddleOCR engine (singleton — reuses if already initialized).

    Parameters
    ----------
    lang : str
        Language for OCR ('en' for English).
    use_gpu : bool
        Whether to use GPU acceleration (default False — CPU is fast
        enough for small cropped regions).

    Returns
    -------
    PaddleOCR instance
    """
    global _ocr_instance

    if _ocr_instance is not None:
        return _ocr_instance

    # Suppress PaddleOCR verbose logging
    import logging
    logging.getLogger('ppocr').setLevel(logging.WARNING)

    from paddleocr import PaddleOCR

    print("[ocr] Initializing PaddleOCR engine...")
    _ocr_instance = PaddleOCR(
        use_angle_cls=True,   # Handle rotated text in CAD drawings
        lang=lang,
        use_gpu=use_gpu,
        show_log=False,
    )
    print("[ocr] PaddleOCR ready.")
    return _ocr_instance


# ---------------------------------------------------------------------------
# Region Cropping
# ---------------------------------------------------------------------------

def crop_region(image: np.ndarray, bbox: tuple,
                padding: int = CROP_PADDING) -> np.ndarray:
    """
    Crop a bounding box region from an image with padding.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3).
    bbox : tuple
        (x1, y1, x2, y2) pixel coordinates.
    padding : int
        Extra pixels around the bbox.

    Returns
    -------
    np.ndarray — cropped BGR image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply padding with bounds checking
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]

    # Ensure minimum size for OCR
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return None

    return crop


# ---------------------------------------------------------------------------
# OCR on a Single Region
# ---------------------------------------------------------------------------

def ocr_region(ocr, image: np.ndarray, bbox: tuple) -> dict:
    """
    Run OCR on a single detected region.

    Parameters
    ----------
    ocr : PaddleOCR instance
    image : np.ndarray
        Full BGR image.
    bbox : tuple
        (x1, y1, x2, y2) of the region.

    Returns
    -------
    dict with:
        text         — str, all text found (joined)
        text_lines   — list of individual text strings
        confidences  — list of float confidence values
        avg_conf     — float, average confidence
        has_text     — bool, whether any text was found
    """
    crop = crop_region(image, bbox)

    if crop is None:
        return {
            'text': '',
            'text_lines': [],
            'confidences': [],
            'avg_conf': 0.0,
            'has_text': False,
        }

    try:
        results = ocr.ocr(crop, cls=True)
    except Exception as e:
        print(f"[ocr] OCR error on region {bbox}: {e}")
        return {
            'text': '',
            'text_lines': [],
            'confidences': [],
            'avg_conf': 0.0,
            'has_text': False,
        }

    text_lines = []
    confidences = []

    # PaddleOCR returns list of lists; handle both formats
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                text_str = str(line[1][0]).strip()
                conf = float(line[1][1])

                if conf >= MIN_OCR_CONF and text_str:
                    text_lines.append(text_str)
                    confidences.append(conf)

    combined_text = ' '.join(text_lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        'text': combined_text,
        'text_lines': text_lines,
        'confidences': confidences,
        'avg_conf': round(avg_conf, 4),
        'has_text': len(text_lines) > 0,
    }


# ---------------------------------------------------------------------------
# OCR on All Detections
# ---------------------------------------------------------------------------

def ocr_all_detections(ocr, image: np.ndarray,
                       detections: list) -> list:
    """
    Run OCR on all YOLO detections, enriching each with text content.

    Parameters
    ----------
    ocr : PaddleOCR instance
    image : np.ndarray
        BGR image (full drawing).
    detections : list
        Output from yolo_detector.detect().

    Returns
    -------
    list of dict — same as input detections but with added keys:
        text, text_lines, confidences, avg_conf, has_text
    """
    enriched = []

    for i, det in enumerate(detections):
        ocr_result = ocr_region(ocr, image, det['bbox'])

        # Merge OCR results into detection dict
        enriched_det = {**det, **ocr_result}
        enriched.append(enriched_det)

        text_preview = ocr_result['text'][:60] if ocr_result['text'] else '(no text)'
        print(f"  [{i+1}] {det['class_name']:>10s} | "
              f"conf={det['confidence']:.2f} | "
              f"ocr_conf={ocr_result['avg_conf']:.2f} | "
              f"text: {text_preview}")

    return enriched


# ---------------------------------------------------------------------------
# Full Pipeline: Detect + OCR
# ---------------------------------------------------------------------------

def ocr_drawing(ocr, model, image: np.ndarray,
                det_conf: float = 0.25) -> list:
    """
    Full pipeline: run YOLO detection then OCR on each detected region.

    Parameters
    ----------
    ocr : PaddleOCR instance
    model : YOLO model
    image : np.ndarray
        BGR image.
    det_conf : float
        Detection confidence threshold.

    Returns
    -------
    list of enriched detections (detection + OCR results)
    """
    from app.yolo_detector import detect

    detections = detect(model, image, conf=det_conf)
    print(f"[ocr] {len(detections)} detections to OCR...")

    enriched = ocr_all_detections(ocr, image, detections)
    return enriched


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from app.input_handler import load_drawing_pair
    from app.yolo_detector import load_model, detect_on_pair

    # Initialize
    ocr = init_ocr()
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
        print(f"  OCR TEST: {label} ({file_a} ↔ {file_b})")
        print(f"{'='*60}")

        t0 = time.time()

        # Load pair
        pair = load_drawing_pair(file_a, file_b, dpi=300, align=True)

        # Detect on both
        det_result = detect_on_pair(model, pair['img_a'], pair['img_b'])

        # OCR on image A detections
        print(f"\n  --- OCR on Image A ({det_result['count_a']} detections) ---")
        enriched_a = ocr_all_detections(
            ocr, pair['img_a'], det_result['detections_a']
        )

        # OCR on image B detections
        print(f"\n  --- OCR on Image B ({det_result['count_b']} detections) ---")
        enriched_b = ocr_all_detections(
            ocr, pair['img_b'], det_result['detections_b']
        )

        elapsed = time.time() - t0

        # Summary
        texts_a = [d['text'] for d in enriched_a if d['has_text']]
        texts_b = [d['text'] for d in enriched_b if d['has_text']]

        print(f"\n[summary] Time: {elapsed:.2f}s")
        print(f"[summary] Image A: {len(texts_a)}/{len(enriched_a)} "
              f"detections have text")
        print(f"[summary] Image B: {len(texts_b)}/{len(enriched_b)} "
              f"detections have text")

        # Print all extracted text
        print(f"\n  === All Text from A ===")
        for d in enriched_a:
            if d['has_text']:
                x1, y1, x2, y2 = d['bbox']
                print(f"    {d['class_name']:>10s} @ ({x1},{y1}): "
                      f"\"{d['text']}\"")

        print(f"\n  === All Text from B ===")
        for d in enriched_b:
            if d['has_text']:
                x1, y1, x2, y2 = d['bbox']
                print(f"    {d['class_name']:>10s} @ ({x1},{y1}): "
                      f"\"{d['text']}\"")

    print(f"\n{'='*60}")
    print("  ALL OCR TESTS COMPLETE ✅")
    print(f"{'='*60}")
