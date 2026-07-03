"""
Phase 10: AI Summary Generator
================================
Generates a structured, natural language Markdown report from the comparison
statistics and semantic element matches.

Public API:
  generate_summary_report(stats: dict) → str
"""

def _guess_drawing_type(stats: dict) -> str:
    """Guess the type of CAD drawing based on extracted OCR text."""
    all_text = []
    raw = stats.get('raw_match_data', {})
    
    # Gather all text we found
    for cat in ['added', 'removed', 'unchanged']:
        for item in raw.get(cat, []):
            if item.get('has_text'):
                all_text.append(item['text'].upper())
                
    for item in raw.get('modified', []) + raw.get('moved', []):
        if item.get('text_old'): all_text.append(item['text_old'].upper())
        if item.get('text_new'): all_text.append(item['text_new'].upper())
        
    combined_text = " ".join(all_text)
    
    if any(k in combined_text for k in ['BRIDGE', 'GIRDER', 'PIER', 'ABUTMENT', 'DECK']):
        return "Bridge / Infrastructure"
    elif any(k in combined_text for k in ['ELEVATION', 'HOUSE', 'FLOOR PLAN', 'ROOF']):
        return "Architectural / Building"
    elif any(k in combined_text for k in ['PIPING', 'P&ID', 'VALVE']):
        return "Piping and Instrumentation"
    elif any(k in combined_text for k in ['PCB', 'CIRCUIT', 'SCHEMATIC', 'WIRING']):
        return "Electrical Schematic"
        
    return "Engineering / Mechanical"


def generate_summary_report(stats: dict) -> str:
    """
    Generate a Markdown-formatted text report summarizing the changes.
    Matches the user's requested sample format.
    """
    drawing_type = _guess_drawing_type(stats)
    
    p_metrics = stats.get('pixel_metrics', {})
    e_metrics = stats.get('element_metrics', {})
    unlabeled = stats.get('unlabeled_geometry_changes', {})
    raw = stats.get('raw_match_data', {})
    
    total_semantic = e_metrics.get('total_semantic_changes', 0)
    total_unlabeled = unlabeled.get('count', 0)
    
    # Header
    md = [
        f"# CAD Drawing Comparison Report",
        f"**Drawing Type:** {drawing_type}  ",
        f"**Similarity Score (SSIM):** {p_metrics.get('ssim_score', 0):.4f}  ",
        f"**Total Pixel Change Area:** {p_metrics.get('percent_changed', 0)}%  ",
        f"\n*{total_semantic} semantic change(s) found.*",
        f"\n---"
    ]
    
    # 1. Added
    added_list = raw.get('added', [])
    if added_list:
        md.append(f"\n## New Elements (Added)")
        for i, item in enumerate(added_list):
            cls = item.get('class_name', 'Element')
            text = f"'{item.get('text')}' " if item.get('has_text') else ""
            pos = item.get('center', (0,0))
            md.append(f"{i+1}. New {cls} {text}found in AFTER, not present in BEFORE, near position {pos}.")
            
    # 2. Removed
    removed_list = raw.get('removed', [])
    if removed_list:
        md.append(f"\n## Missing Elements (Removed)")
        for i, item in enumerate(removed_list):
            cls = item.get('class_name', 'Element')
            text = f"'{item.get('text')}' " if item.get('has_text') else ""
            pos = item.get('center', (0,0))
            md.append(f"{i+1}. {cls.capitalize()} {text}from BEFORE is missing in AFTER, near position {pos}.")
            
    # 3. Modified
    modified_list = raw.get('modified', [])
    if modified_list:
        md.append(f"\n## Modified Text/Values")
        for i, item in enumerate(modified_list):
            cls = item['old'].get('class_name', 'Element')
            pos = item['new'].get('center', (0,0))
            t_old = item.get('text_old', '')
            t_new = item.get('text_new', '')
            md.append(f"{i+1}. {cls.capitalize()} value changed from '{t_old}' to '{t_new}' near position {pos}.")
            
    # 4. Moved
    moved_list = raw.get('moved', [])
    if moved_list:
        md.append(f"\n## Moved Elements")
        for i, item in enumerate(moved_list):
            cls = item['old'].get('class_name', 'Element')
            text = f"'{item.get('text_old')}' " if item['old'].get('has_text') else ""
            old_pos = item['old'].get('center', (0,0))
            new_pos = item['new'].get('center', (0,0))
            md.append(f"{i+1}. {cls.capitalize()} {text}moved from {old_pos} to {new_pos}.")

    # 5. Unlabeled Geometry
    if total_unlabeled > 0:
        md.append(f"\n## Unlabeled Geometry Changes (Needs manual review)")
        regions = unlabeled.get('regions', [])
        for i, region in enumerate(regions):
            pos = region.get('center', (0,0))
            area = region.get('area', 0)
            md.append(f"{i+1}. Possible geometry change detected near position {pos} (Area: {area}px) — no matching dimension text found nearby; recommend manual review.")
        
        if total_unlabeled > len(regions):
            md.append(f"*... and {total_unlabeled - len(regions)} smaller geometry changes omitted.*")
            
    if total_semantic == 0 and total_unlabeled == 0:
        md.append("\n*No significant structural or semantic changes were detected between these drawings.*")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from app.input_handler import load_drawing_pair
    from app.pixel_diff import run_pixel_diff
    from app.yolo_detector import load_model, detect_on_pair
    from app.ocr_engine import init_ocr, ocr_all_detections
    from app.element_matcher import match_elements
    from app.statistics import compute_statistics
    
    model = load_model()
    ocr = init_ocr()

    pairs = [
        ('v1.pdf', 'v2.pdf', 'pair1'),
        ('v1-new.pdf', 'v2-new.pdf', 'pair2'),
    ]

    for file_a, file_b, label in pairs:
        print(f"\n{'='*60}")
        print(f"  AI SUMMARY: {label}")
        print(f"{'='*60}")
        
        pair = load_drawing_pair(file_a, file_b, align=True)
        diff_res = run_pixel_diff(pair['gray_a'], pair['gray_b'], pair['img_a'], pair['img_b'])
        det_res = detect_on_pair(model, pair['img_a'], pair['img_b'])
        det_a = ocr_all_detections(ocr, pair['img_a'], det_res['detections_a'])
        det_b = ocr_all_detections(ocr, pair['img_b'], det_res['detections_b'])
        match_res = match_elements(det_a, det_b)
        
        stats = compute_statistics(diff_res, match_res)
        report = generate_summary_report(stats)
        
        print("\n\n" + report + "\n")
        
        with open(f"outputs/{label}_report.md", "w") as f:
            f.write(report)
        print(f"[saved] outputs/{label}_report.md")

    print(f"\n{'='*60}")
    print("  PHASE 10 TESTS COMPLETE ✅")
    print(f"{'='*60}")
