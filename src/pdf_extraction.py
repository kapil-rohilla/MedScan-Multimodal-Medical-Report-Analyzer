from pathlib import Path
import re
import fitz
from pdf2image import convert_from_path
import pytesseract


def clean_text_basic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf_text_smart(
    pdf_path,
    force_ocr: bool = False,
    scanned_threshold: int = 50,
    dpi: int = 300,
    lang: str = "eng",
):

    pdf_path = Path(pdf_path)

    doc = fitz.open(pdf_path)
    pages_data = []
    full_text_parts = []

    total_pages = len(doc)

    for page_index, page in enumerate(doc, start=1):
        # 1) Direct extraction
        direct_text = page.get_text("text") or ""
        direct_char_count = len(direct_text)

        use_ocr = force_ocr or (direct_char_count < scanned_threshold)

        if use_ocr:
            images = convert_from_path(
                str(pdf_path),
                first_page=page_index,
                last_page=page_index,
                dpi=dpi,
            )
            if images:
                ocr_text = pytesseract.image_to_string(images[0], lang=lang)
            else:
                ocr_text = ""
            raw_text = ocr_text
            mode = "ocr"
        else:
            raw_text = direct_text
            mode = "direct"

        cleaned_text = clean_text_basic(raw_text)

        pages_data.append(
            {
                "page_number": page_index,
                "mode": mode,
                "char_count": len(cleaned_text),
                "text": cleaned_text,
            }
        )

        full_text_parts.append(f"\n\n--- PAGE {page_index} ({mode}) ---\n\n")
        full_text_parts.append(cleaned_text)

    doc.close()

    full_text = "".join(full_text_parts)

    return {
        "file_name": pdf_path.name,
        "total_pages": total_pages,
        "pages": pages_data,
        "full_text": full_text,
    }