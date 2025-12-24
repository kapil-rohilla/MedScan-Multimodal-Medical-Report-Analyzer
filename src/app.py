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

        h1, h2, h3 {{
            color: #1f2a44;
            font-weight: 600;
        }}

        /* Card containers */
        .card {{
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}

        /* Buttons */
        div.stButton > button {{
            background-color: #1f6feb;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.3rem;
            font-weight: 500;
            border: none;
        }}

        div.stButton > button:hover {{
            background-color: #174ea6;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit Configuration

st.set_page_config(
    page_title="Multimodal Medical Report Analyzer",
    layout="wide"
)

set_background(
    r"C:\Users\kapil\OneDrive\Desktop\Multimodal Medical Report Analyzer\src\bg_image.png"
)

st.title("MedScan")
st.caption(
    "End-to-end analysis of medical PDF reports using OCR, table parsing, "
    "image extraction, and controlled language model summarization."
)

st.markdown("---")

# Session State Initialization

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "parsed_report" not in st.session_state:
    st.session_state.parsed_report = None
if "llm_result" not in st.session_state:
    st.session_state.llm_result = None
if "llm_mode" not in st.session_state:
    st.session_state.llm_mode = None

# Sidebar Controls

st.sidebar.header("Input Controls")

uploaded_file = st.sidebar.file_uploader("Upload Medical PDF", type=["pdf"])
ocr_dpi = st.sidebar.slider("OCR DPI", 200, 500, 300)
scanned_threshold = st.sidebar.slider("Scanned Page Threshold", 10, 200, 50)

st.sidebar.markdown("---")

if st.sidebar.button("Reset Application"):
    st.session_state.clear()
    st.rerun()

def highlight_lab_flags(row):
    if row.get("flag") == "Low":
        return ["background-color: #fdecea"] * len(row)
    if row.get("flag") == "High":
        return ["background-color: #fff4e5"] * len(row)
    return [""] * len(row)


if uploaded_file:

    pdf_path = Path("uploaded_temp.pdf")
    pdf_path.write_bytes(uploaded_file.read())

    col_left, col_right = st.columns([2.2, 1])

    # RIGHT COLUMN 

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("PDF Preview")

        try:
            images = convert_from_path(pdf_path, dpi=150)

            if len(images) == 1:
                st.image(images[0], width="stretch")
            else:
                page = st.slider("Select Page", 1, len(images), 1)
                st.image(images[page - 1], width="stretch")

        except Exception as e:
            st.error(f"Preview error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # LEFT COLUMN 

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.success("PDF uploaded successfully.")

        if st.button("Analyze Report") and not st.session_state.pipeline_done:

            with st.spinner("Extracting text and tables..."):
                text_result = extract_pdf_text_smart(
                    pdf_path=pdf_path,
                    scanned_threshold=scanned_threshold,
                    dpi=ocr_dpi,
                    lang="eng"
                )

            st.subheader("Extracted Text")
            st.text_area("", text_result["full_text"], height=250)

            table_results = None
            table_df = extract_pdf_table(pdf_path)

            if table_df is not None and not table_df.empty:
                st.subheader("Detected Table")
                st.dataframe(table_df)

                table_type = classify_table_type(table_df)
                st.caption(f"Detected table type: {table_type}")

                if table_type == "lab":
                    cleaned_df = clean_lab_table(table_df)
                    if not cleaned_df.empty:
                        st.subheader("Cleaned Laboratory Table")
                        st.dataframe(
                            cleaned_df.style.apply(highlight_lab_flags, axis=1)
                        )
                        table_results = cleaned_df.to_dict(orient="records")

            parsed_report = parse_universal_report(
                text_result["full_text"],
                table_results=table_results
            )

            st.session_state.parsed_report = parsed_report
            st.session_state.pipeline_done = True

        st.markdown("</div>", unsafe_allow_html=True)

        # LLM SUMMARY SECTION
        if st.session_state.pipeline_done and st.session_state.parsed_report:

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.header("Report Interpretation")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Generate Patient Summary"):
                    st.session_state.llm_mode = "patient"
                    st.session_state.llm_result = explain_report(
                        st.session_state.parsed_report,
                        mode="patient"
                    )

            with col2:
                if st.button("Generate Clinical Summary"):
                    st.session_state.llm_mode = "clinician"
                    st.session_state.llm_result = explain_report(
                        st.session_state.parsed_report,
                        mode="clinician"
                    )

            if st.session_state.llm_result:
                result = st.session_state.llm_result

                st.subheader(
                    "Patient Summary"
                    if st.session_state.llm_mode == "patient"
                    else "Clinical Summary"
                )

                st.markdown("Summary")
                st.write(result["summary"])

                st.markdown("Explanation")
                st.write(result["explanation"])

                st.caption(result["disclaimer"])

            st.markdown("</div>", unsafe_allow_html=True)


        # IMAGE EXTRACTION & CLASSIFICATION

        if st.session_state.pipeline_done:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.header("Image Extraction and Classification")

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
                    st.subheader(cls)
                    subset = df[df["type"] == cls]
                    cols = st.columns(3)
                    idx = 0
                    for _, r in subset.iterrows():
                        with cols[idx]:
                            st.image(Image.open(r["path"]), width="stretch")
                            st.caption(
                                f"{r['type']} ({r['confidence']}%)"
                            )
                        idx = (idx + 1) % 3

            st.success("Pipeline completed successfully.")
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload a medical PDF to begin analysis.")
