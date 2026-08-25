import fitz  # PyMuPDF
import os
import cv2
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

def is_blank_or_spacer(img_path):
    """
    Detects if an extracted image is a blank spacer page, publisher placeholder,
    or solid uniform color with no narrative art.
    """
    img = cv2.imread(img_path)
    if img is None:
        return True
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Uniformity check
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    max_bin_ratio = float(np.max(hist) / (h * w))
    
    # 2. Edge density check (art line count)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (h * w))
    
    # Blank/spacer pages have >85% uniform background and <0.008 edge density
    if max_bin_ratio > 0.85 and edge_density < 0.008:
        return True
    return False

def slice_page_into_panels(img_path, output_dir=None):
    """
    Slices a tall multi-panel manhwa page into individual chronological story panel frames.
    Preserves full-bleed splash pages while extracting individual panels on multi-panel pages.
    """
    img = cv2.imread(img_path)
    if img is None:
        return [img_path]
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect background color (white vs black gutters)
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    is_white_bg = np.median(border_pixels) > 200
    
    # 2. Horizontal Gutter Bisection (Find distinct panel rows)
    row_means = np.mean(gray, axis=1)
    is_content = (row_means < 248) if is_white_bg else (row_means > 20)
    
    row_segments = []
    in_seg = False
    start_y = 0
    
    for y in range(h):
        if is_content[y] and not in_seg:
            in_seg = True
            start_y = y
        elif not is_content[y] and in_seg:
            in_seg = False
            if (y - start_y) > 0.05 * h:
                row_segments.append((start_y, y))
    if in_seg and (h - start_y) > 0.05 * h:
        row_segments.append((start_y, h))
        
    if not row_segments:
        row_segments = [(0, h)]
        
    final_panel_boxes = []
    
    for (r_start, r_end) in row_segments:
        row_crop = img[r_start:r_end, :]
        row_gray = gray[r_start:r_end, :]
        rh, rw = row_crop.shape[:2]
        
        # Check column splits in this row
        col_means = np.mean(row_gray, axis=0)
        is_col_content = (col_means < 248) if is_white_bg else (col_means > 20)
        
        col_segments = []
        in_col = False
        start_x = 0
        for x in range(rw):
            if is_col_content[x] and not in_col:
                in_col = True
                start_x = x
            elif not is_col_content[x] and in_col:
                in_col = False
                if (x - start_x) > 0.18 * rw:
                    col_segments.append((start_x, x))
        if in_col and (rw - start_x) > 0.18 * rw:
            col_segments.append((start_x, rw))
            
        if len(col_segments) > 1:
            for (c_start, c_end) in col_segments:
                final_panel_boxes.append((c_start, r_start, c_end, r_end))
        else:
            # Check for offset sub-panels using contour boxes
            row_edges = cv2.Canny(row_gray, 30, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
            closed = cv2.morphologyEx(row_edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            sub_boxes = []
            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                if area > 0.06 * (rh * rw) and area < 0.85 * (rh * rw):
                    sub_boxes.append((bx, r_start + by, bx + bw, r_start + by + bh))
                    
            if len(sub_boxes) >= 2:
                sub_boxes.sort(key=lambda b: (b[1] // 100, b[0]))
                final_panel_boxes.extend(sub_boxes)
            else:
                final_panel_boxes.append((0, r_start, w, r_end))
                
    # If only 1 full-page box found, return original image
    if len(final_panel_boxes) <= 1:
        return [img_path]
        
    cropped_paths = []
    target_dir = output_dir or os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    for i, (x1, y1, x2, y2) in enumerate(final_panel_boxes):
        pad_x = int(0.015 * w)
        pad_y = int(0.015 * h)
        xs = max(0, x1 - pad_x)
        ys = max(0, y1 - pad_y)
        xe = min(w, x2 + pad_x)
        ye = min(h, y2 + pad_y)
        
        crop = img[ys:ye, xs:xe]
        if crop.shape[0] < 120 or crop.shape[1] < 120:
            continue
            
        out_file = os.path.join(target_dir, f"{base_name}_p{i+1:02d}.png")
        cv2.imwrite(out_file, crop)
        cropped_paths.append(out_file)
        
    # Remove original unsliced page if panels were successfully extracted
    if cropped_paths:
        try:
            os.remove(img_path)
        except Exception:
            pass
            
    return cropped_paths or [img_path]

def _process_page(pdf_path, page_num, output_dir, zoom, slice_panels=True):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    output_filename = f"page_{page_num + 1:04d}.png"
    output_path = os.path.join(output_dir, output_filename)
    pix.save(output_path)
    doc.close()
    
    # Auto-filter blank spacer pages
    if is_blank_or_spacer(output_path):
        try:
            os.remove(output_path)
            logger.info(f"Skipping blank/spacer page {page_num + 1}")
            return []
        except Exception:
            return []
            
    # Auto-slice into individual panel frames if enabled
    if slice_panels:
        panel_paths = slice_page_into_panels(output_path, output_dir=output_dir)
        return panel_paths
            
    return [output_path]

def pdf_to_images(pdf_path, output_dir, start_page=None, end_page=None, zoom=2.0, slice_panels=True):
    """
    Converts a PDF file to high-resolution images in parallel, automatically skipping blank spacer pages
    and slicing multi-panel manhwa pages into individual story frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    p_start = max(0, (start_page - 1)) if start_page is not None else 0
    p_end = min(total_pages, end_page) if end_page is not None else total_pages
    pages_to_process = list(range(p_start, p_end))
    
    logger.info(f"Extracting pages {p_start+1} to {p_end} from {os.path.basename(pdf_path)} (Panel Slicing: {slice_panels})...")
    
    image_paths = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_process_page, pdf_path, i, output_dir, zoom, slice_panels)
            for i in pages_to_process
        ]
        for future in futures:
            res = future.result()
            if res:
                image_paths.extend(res)
                
    return sorted(image_paths)
