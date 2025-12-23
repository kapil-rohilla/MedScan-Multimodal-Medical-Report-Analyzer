import re
from typing import List, Dict

# Public API

def build_lab_rows_from_text(raw_text: str) -> List[Dict]:
    
    if not raw_text:
        return []

    # Preserved line structure
    text = normalize_text(raw_text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    rows: List[Dict] = []
    
    # CBC
    cbc_lines = extract_block_lines(
        lines,
        start_keywords=["complete blood counts", "cbc"],
        end_keywords=["widal"]
    )

    if cbc_lines:
        rows.extend(parse_cbc_lines(cbc_lines))

    # WIDAL
    widal_lines = extract_block_lines(
        lines,
        start_keywords=["widal"],
        end_keywords=["interpretation", "comment"]
    )

    if widal_lines:
        rows.extend(parse_widal_lines(widal_lines))

    return rows

# Block extraction (LINE-BASED)


def extract_block_lines(lines, start_keywords, end_keywords):
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if any(k in line.lower() for k in start_keywords):
            start_idx = i
            break

    if start_idx is None:
        return []

    for j in range(start_idx + 1, len(lines)):
        if any(k in lines[j].lower() for k in end_keywords):
            end_idx = j
            break

    return lines[start_idx:end_idx] if end_idx else lines[start_idx:]

# CBC parsing

CBC_SECTION_HEADERS = {
    "COMPLETE BLOOD COUNTS",
    "DIFFERENTIAL LEUCOCYTE COUNT"
}
def parse_cbc_lines(lines: List[str]) -> List[Dict]:
    rows = []
    i = 0

    while i < len(lines) - 3:
        name = lines[i]

        # CBC test names are usually ALL CAPS
        if (
            not name.isupper()
            or len(name) < 3
            or name in CBC_SECTION_HEADERS
        ):
            i += 1
            continue


        unit = None
        ref_low = ref_high = None
        value = None

        for j in range(1, 5):
            if i + j >= len(lines):
                break

            line = lines[i + j]

            # Reference range
            m = re.search(r"\((\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)\)", line)
            if m:
                ref_low = float(m.group(1))
                ref_high = float(m.group(3))
                continue

            # Numeric value
            if re.fullmatch(r"\d+(\.\d+)?", line):
                value = float(line)
                continue

            # Unit
            if unit is None and re.search(r"[a-zA-Z/%]+", line):
                unit = line
                continue

        if value is not None and ref_low is not None and ref_high is not None:
            rows.append({
                "panel": "CBC",
                "test_name": name,
                "value": value,
                "unit": unit,
                "reference_range": f"{ref_low} - {ref_high}",
                "flag": compute_flag(value, ref_low, ref_high)
            })
            i += 4
        else:
            i += 1

    return rows

# WIDAL parsing

WIDAL_PATTERN = re.compile(
    r"(O\s*ANTIGEN|H\s*ANTIGEN|T\s*H|PARATYPHI\s*A\s*H|PARATYPHI\s*B\s*H).*?(1:\d+)",
    re.IGNORECASE
)


def parse_widal_lines(lines: List[str]) -> List[Dict]:
    rows = []
    text = " ".join(lines).upper()

    for m in WIDAL_PATTERN.finditer(text):
        rows.append({
            "panel": "WIDAL",
            "test_name": m.group(1).strip(),
            "value": m.group(2),
            "unit": "titer",
            "reference_range": None,
            "flag": None
        })

    return rows

# Utilities

def compute_flag(value, low, high):
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Normal"


def normalize_text(text: str) -> str:
    """
    Remove obvious noise but PRESERVE line breaks.
    """
    cleaned_lines = []
    for line in text.splitlines():
        if re.search(r"Tech:-", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
