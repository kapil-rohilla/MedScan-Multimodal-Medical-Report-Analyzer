import streamlit as st
from pdf2image import convert_from_path
import tempfile
import os

# Import 6-model classifier
import sys
sys.path.append(r"C:\Users\kapil\OneDrive\Desktop\Multimodal Medical Report Analyzer\src")
from image_classification import MedicalImageClassifier


st.set_page_config(page_title="Medical Image Classifier", layout="wide")

BASE_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\Multimodal Medical Report Analyzer\image_classification_model"

classifier = MedicalImageClassifier(BASE_MODEL_PATH)

st.title(" Medical Image Classification ")

def classify_and_display(img_path):
    label, conf, all_scores = classifier.predict(img_path)

    st.image(img_path, width=350)
    st.write(f"### 🏷 Final Prediction: **{label}**")
    st.write(f" Confidence: **{conf:.4f}**")

    # Detailed model outputs
    with st.expander(" View Detailed Scores from All Models"):
        for model_key, data in all_scores.items():
            st.write(
                f"**{model_key.upper()}** → {data['prediction']} "
                f"(Confidence: {data['confidence']:.4f})"
            )


uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # IMAGE INPUT
    if file_ext in ["jpg", "jpeg", "png"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.subheader(" Uploaded Image")
        classify_and_display(tmp_path)

    # PDF INPUT
    elif file_ext == "pdf":
        st.subheader(" PDF Detected — Extracting Pages...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_file.read())
            pdf_path = tmp_pdf.name

        pages = convert_from_path(pdf_path)
        st.write(f" Found **{len(pages)} pages**")

        page_num = st.number_input(
            "Select Page", min_value=1, max_value=len(pages), value=1
        )

        # Convert selected page to image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            pages[page_num - 1].save(tmp_img.name, "PNG")
            page_img_path = tmp_img.name

        st.subheader(f" Page {page_num}")
        classify_and_display(page_img_path)