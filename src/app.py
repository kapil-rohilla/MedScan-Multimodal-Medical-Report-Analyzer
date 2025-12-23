import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
import base64

from pdf_extraction import extract_pdf_text_smart
from table_extraction import extract_pdf_table
from image_extraction_0 import process_file
from image_classification import MedicalImageClassifier
from hybrid_report_parser import parse_universal_report
from table_cleaner import clean_lab_table
from table_type_classifier import classify_table_type
from llm_explainer import explain_report

# Background Image

def set_background(image_path: str):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255,255,255,0.88),
                    rgba(255,255,255,0.88)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit Configuration

st.set_page_config(page_title="Multimodal Medical Analyzer", layout="wide")
set_background(
    r"C:\Users\kapil\OneDrive\Desktop\Multimodal Medical Report Analyzer\src\bg_image.png"
)

st.title("🩺 MedScan")

# ================================================
# Session State
# ================================================
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "parsed_report" not in st.session_state:
    st.session_state.parsed_report = None
if "llm_result" not in st.session_state:
    st.session_state.llm_result = None
if "llm_mode" not in st.session_state:
    st.session_state.llm_mode = None


# Sidebar Controls

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
ocr_dpi = st.sidebar.slider("OCR DPI", 200, 500, 300)
scanned_threshold = st.sidebar.slider("Scanned Page Threshold", 10, 200, 50)

st.sidebar.markdown("---")
st.sidebar.info("Upload a PDF to begin processing.")

if st.sidebar.button("🔄 Reset / Upload New PDF"):
    st.session_state.clear()
    st.rerun()

# Highlight lab flags

def highlight_lab_flags(row):
    if row.get("flag") == "Low":
        return ["background-color: #ffcccc"] * len(row)
    if row.get("flag") == "High":
        return ["background-color: #ffe0b3"] * len(row)
    return [""] * len(row)

# MAIN UI

if uploaded_file:

    pdf_path = Path("uploaded_temp.pdf")
    pdf_path.write_bytes(uploaded_file.read())

    col_left, col_right = st.columns([2, 1])

    # RIGHT COLUMN(PDF PREVIEW)
 
    with col_right:
        st.subheader("PDF Preview")
        try:
            images = convert_from_path(pdf_path, dpi=150)

            if len(images) == 1:
                st.image(images[0], caption="Page 1", use_container_width=True)
            else:
                page = st.slider("Select Page", 1, len(images), 1)
                st.image(
                    images[page - 1],
                    caption=f"Page {page}",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Preview error: {e}")

    # LEFT COLUMN 
    with col_left:
        st.success("PDF uploaded successfully.")

        if st.button("Run Full Extraction Pipeline") and not st.session_state.pipeline_done:

            # TEXT EXTRACTION 
            with st.spinner("Extracting text..."):
                text_result = extract_pdf_text_smart(
                    pdf_path=pdf_path,
                    scanned_threshold=scanned_threshold,
                    dpi=ocr_dpi,
                    lang="eng"
                )

            st.text_area("Extracted Text", text_result["full_text"], height=250)

            # TABLE EXTRACTION
            table_results = None
            table_df = extract_pdf_table(pdf_path)

            if table_df is not None and not table_df.empty:
                st.subheader("📊 Raw Table")
                st.dataframe(table_df)

                table_type = classify_table_type(table_df)
                st.write(f"Detected Table Type: `{table_type}`")

                if table_type == "lab":
                    cleaned_df = clean_lab_table(table_df)
                    if not cleaned_df.empty:
                        st.subheader("Cleaned Lab Table")
                        st.dataframe(
                            cleaned_df.style.apply(highlight_lab_flags, axis=1)
                        )
                        table_results = cleaned_df.to_dict(orient="records")

            # STRUCTURED PARSING 
            parsed_report = parse_universal_report(
                text_result["full_text"],
                table_results=table_results
            )

            st.session_state.parsed_report = parsed_report
            st.session_state.pipeline_done = True

        # LLM REPORT SUMMARY
        if st.session_state.pipeline_done and st.session_state.parsed_report:

            st.markdown("---")
            st.header("Report Summary")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Patient View"):
                    st.session_state.llm_mode = "patient"
                    st.session_state.llm_result = explain_report(
                        st.session_state.parsed_report,
                        mode="patient"
                    )

            with col2:
                if st.button("Clinical View"):
                    st.session_state.llm_mode = "clinician"
                    st.session_state.llm_result = explain_report(
                        st.session_state.parsed_report,
                        mode="clinician"
                    )

            if st.session_state.llm_result:
                result = st.session_state.llm_result

                title = (
                    "Patient Summary"
                    if st.session_state.llm_mode == "patient"
                    else "Clinical Summary"
                )

                st.subheader(title)

                st.markdown("### Summary")
                st.write(result["summary"])

                st.markdown("### Explanation")
                st.write(result["explanation"])

                st.caption(result["disclaimer"])

            st.markdown("---")
            st.subheader("Full Parsed JSON")
            st.json(st.session_state.parsed_report)

        # ================= IMAGE PIPELINE =================
        if st.session_state.pipeline_done:
            st.markdown("---")
            st.header("Image Extraction & Classification")

            extracted_images = process_file(pdf_path)

            if extracted_images:
                clf = MedicalImageClassifier(
                    r"C:\Users\kapil\OneDrive\Desktop\Multimodal Medical Report Analyzer\image_classification_model"
                )

                results = []
                for img in extracted_images:
                    label, conf, _ = clf.predict(img)
                    results.append({
                        "image": Path(img).name,
                        "type": label,
                        "confidence": round(conf * 100, 1),
                        "path": img
                    })

                df = pd.DataFrame(results)
                st.dataframe(df[["image", "type", "confidence"]])

                for cls in df["type"].unique():
                    st.markdown(f"### {cls}")
                    subset = df[df["type"] == cls]
                    cols = st.columns(3)
                    idx = 0
                    for _, r in subset.iterrows():
                        with cols[idx]:
                            st.image(Image.open(r["path"]), use_container_width=True)
                            st.caption(f"{r['type']} ({r['confidence']}%)")
                        idx = (idx + 1) % 3

            st.success("Pipeline completed successfully!")

else:
    st.info("Upload a PDF to start.")
