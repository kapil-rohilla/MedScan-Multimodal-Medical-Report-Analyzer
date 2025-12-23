import re
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

def detect_pdf_type(pdf_path):
    reader = PdfReader(str(pdf_path))
    text = reader.pages[0].extract_text()
    if text and len(text.strip()) > 30:
        print("Detected: TEXT PDF")
        return "text_pdf"
    else:
        print("Detected: IMAGE PDF (needs OCR)")
        return "image_pdf"

def extract_with_camelot(pdf_path):
    import camelot
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), flavor=flavor)
            if tables and len(tables) > 0:
                print(f"Camelot {flavor.upper()} worked!")
                return tables[0].df
        except Exception as e:
            print(f"Camelot {flavor} error: {e}")
    print("Camelot failed on this PDF.")
    return None

def ocr_extract(pdf_path):
    images = convert_from_path(str(pdf_path), dpi=300)
    img = images[0]
    text = pytesseract.image_to_string(img, config="--psm 6 -c preserve_interword_spaces=1")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    results = []
    buffer_test = None

    p_full = re.compile(
        r"(?P<test>[A-Za-z \(\)/]+)\s+"
        r"(?P<result>[\d\.]+)\s*"
        r"(?P<status>Normal|High|Low|Abnormal)?\s*"
        r"(?P<ref>\d+\s*[-–]\s*\d+(\.\d+)?)?\s*"
        r"(?P<unit>[A-Za-z/%]+)?"
    )
    p_result = re.compile(r"(?P<result>[\d\.]+)")

    for line in lines:
        m = p_full.match(line)
        if m and m.group("test") and m.group("result"):
            results.append(m.groupdict())
            buffer_test = None
            continue
        if re.match(r"^[A-Za-z \(\)/]+$", line):
            buffer_test = line
            continue

        if buffer_test:
            m2 = p_result.match(line)
            if m2:
                results.append({
                    "test": buffer_test,
                    "result": m2.group("result"),
                    "status": None,
                    "ref": None,
                    "unit": None
                })
            buffer_test = None

    df = pd.DataFrame(results)
    if not df.empty and "test" in df.columns:
        df = df[df["test"].str.len() > 2].reset_index(drop=True)
    return df

def extract_pdf_table(pdf_path):
    """Unified function: returns DataFrame from digital or scanned medical PDF."""
    pdf_type = detect_pdf_type(pdf_path)
    if pdf_type == "text_pdf":
        df = extract_with_camelot(pdf_path)
        if df is not None:
            return df
    return ocr_extract(pdf_path)

from pathlib import Path

def save_table_outputs(df, pdf_path):
    base = str(Path(pdf_path).with_suffix(''))
    csv_path = base + "_table.csv"
    json_path = base + "_table.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"Results saved to: {csv_path} and {json_path}")
    return csv_path, json_path