
import os
from pathlib import Path
import fitz  
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io

BASE_OUTPUT_DIR = Path(r"C:\Users\kapil\OneDrive\Desktop\Multimodal Medical Report Analyzer\output")

def detect_file_type(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        return "image"
    elif ext == ".dcm":
        return "dicom"
    else:
        return "unknown"

# Extracted page-level images from PDF
def extract_images_from_pdf(pdf_path):
    output_dir = BASE_OUTPUT_DIR / "page_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    extracted = []

    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)

        for idx, img_info in enumerate(images):
            xref = img_info[0]
            base_img = doc.extract_image(xref)

            img_bytes = base_img["image"]
            ext = base_img["ext"]

            img_name = f"{pdf_path.stem}_page{page_num+1}_img{idx+1}.{ext}"
            img_path = output_dir / img_name

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            extracted.append(str(img_path))

    doc.close()
    return extracted


# Detected if an image is a report page (OCR text check)
def is_report_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    text = pytesseract.image_to_string(img)
    return len(text.strip()) > 30


# Cropped images

def crop_xrays_from_image(image_path):
    output_dir = BASE_OUTPUT_DIR / "cropped_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    count = 1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 200 or h < 200:
            continue

        aspect = w / float(h)
        if aspect < 0.3 or aspect > 3.5:
            continue

        crop = img[y:y+h, x:x+w]
        crop_path = output_dir / f"{image_path.stem}_crop{count}.png"
        cv2.imwrite(str(crop_path), crop)

        crops.append(str(crop_path))
        count += 1

    return crops

def process_file(path):
    file_type = detect_file_type(path)
    print(f"Detected file type: {file_type}")

    if file_type == "pdf":
        print("Extracting images from PDF...")
        page_images = extract_images_from_pdf(path)

        all_crops = []
        for img in page_images:
            if is_report_image(img):
                print(f"[Cropping Images from report page: {img}")
                crops = crop_xrays_from_image(img)
                all_crops.extend(crops)
            else:
                all_crops.append(img)

        return all_crops

    elif file_type == "image":
        if is_report_image(path):
            print("[Image looks like a report — cropping images...")
            return crop_xrays_from_image(path)
        else:
            print("[INFO] Pure medical image — no cropping needed.")
            return [path]

    elif file_type == "dicom":
        print("DICOM support to be added later.")
        return []

    else:
        raise ValueError("Unsupported file format")