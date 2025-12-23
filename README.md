# MedScan – Multimodal Medical Report Analyzer
MedScan is multimodal pipeline designed to explore how medical reports in PDF format can be systematically analyzed using a combination of classical processing techniques, deep learning, and large language models.
The system focuses on understanding report structure, not on diagnosing diseases. It extracts text, tables, and images from medical PDFs, classifies embedded medical images by modality, reconstructs laboratory tables commonly used in Indian reports, and produces structured JSON outputs. On top of this structured representation, MedScan generates patient-friendly and clinician-oriented summaries to demonstrate explainability workflows.
## --> This project is built as a learning and experimentation platform, not as a clinical product.

# Demo
X-RAY Report:

https://github.com/user-attachments/assets/f0b17373-b9ce-4b72-8698-d4ca2b6c40be

Lab-Report:

https://github.com/user-attachments/assets/8c96c6b8-26f1-4347-8755-8633fa6e787f

# Screenshots

## Image classification

### ECG classification

<img width="1095" height="632" alt="ECG-Classified" src="https://github.com/user-attachments/assets/7498e49b-928e-4f68-9da7-8f66969b5265" />

### X‑ray classification

<img width="1164" height="803" alt="X-RAY Classified" src="https://github.com/user-attachments/assets/0daadbce-bddf-4655-bcd6-b4e3e8818679" />

Ultrasound classification

<img width="824" height="394" alt="Ultrasound-Classified" src="https://github.com/user-attachments/assets/0625df79-6648-453e-b8d9-93a8ae8e5b4e" />

## LLM summaries

### X‑ray clinical summary

<img width="1070" height="743" alt="X-RAY(Clinical Summary)" src="https://github.com/user-attachments/assets/a4857b4b-0692-4968-842a-9985a2db5305" />

### X‑ray patient summary

<img width="1084" height="816" alt="X-RAY(Patient Summary)" src="https://github.com/user-attachments/assets/845b1bdb-7ea7-4de0-8283-64ebdbefe522" />

### Lab report clinical summary
<img width="1102" height="799" alt="Lab(Clinical Summary)" src="https://github.com/user-attachments/assets/08301a62-47fa-405f-a4f5-6de752ce0cfd" />

### Lab report patient summary

<img width="1181" height="824" alt="Lab(Patient Summary)" src="https://github.com/user-attachments/assets/dcd53084-2469-4700-b2f0-b66a6e05ae2d" />

## Lab table extraction

### Cleaned lab table with low and normal flags

<img width="1904" height="537" alt="Lab Table Extracted" src="https://github.com/user-attachments/assets/482b7faf-8679-419f-9118-d170e59cfe58" />

# What MedScan Can Do

1) Accept medical PDFs such as lab reports and imaging reports
2) Detect whether a PDF is digitally generated or scanned

3) Extract:
- Free text
- Tabular data
- Embedded medical images
  
4) Classify extracted images into:
- CT
- MRI
- X-ray
- Ultrasound
- ECG
- Non-medical

5) Parse Indian laboratory reports including CBC and Widal panels. Convert unstructured content into a clean, structured JSON schema

6) Generate:
A patient-friendly explanation
A clinician-focused technical summary

# Pipeline

**At a high level, the system follows this flow:**

1) PDF ingestion
The input PDF is analyzed to determine whether pages are scanned or digitally generated.
- Text, table, and image extraction
- Digital PDFs are parsed directly
- Scanned PDFs use OCR-based fallbacks
- Tables are extracted using structure-aware methods or reconstructed from raw text

2) Image modality classification
Embedded images are normalized and passed through dedicated binary classifiers to identify the most likely medical modality.

3) Hybrid report parsing
Text content is segmented into logical sections such as patient information, findings, diagnosis, and recommendations.

4) Structured JSON generation
All extracted signals are unified into a single machine-readable representation.

5) LLM-based explanation layer
The structured JSON is transformed into readable summaries without allowing the language model to invent medical facts.

## Project Structure

MedScan-Multimodal-Medical-Report-Analyzer/
├── src/
│   ├── app.py
│   ├── pdf_extraction.py
│   ├── table_extraction.py
│   ├── table_cleaner.py
│   ├── table_type_classifier.py
│   ├── hybrid_report_parser.py
│   ├── lab_text_row_parser.py
│   ├── image_extraction_0.py
│   ├── image_classification.py
│   └── llm_explainer.py
│
├── image_classification_model/
│   └── *.pth
│
├── requirements.txt
└── README.md

**The src directory contains all production logic, while trained model weights are stored separately to keep responsibilities clear.**

# External Software & System Dependencies

In addition to Python libraries listed in requirements.txt, MedScan relies on several external tools that must be installed at the system level for full functionality.
These tools are required mainly for handling scanned PDFs, image-based text extraction, and table reconstruction.
1) Tesseract OCR  -> Tesseract is used for optical character recognition when processing scanned PDFs or image-based lab reports. After installation, ensure the tesseract executable is available in your system PATH so it can be called by pytesseract.
2) Poppler -> Poppler provides command-line utilities used by pdf2image to convert PDF pages into images. On Windows, Poppler must be downloaded separately and added to the PATH.
3) Python Environment -> Python version: 3.9 or newer recommended. Virtual environments are strongly encouraged to isolate dependencies.

# -> Running the Application

cd src
streamlit run app.py
(-> The interface allows users to upload a medical PDF and observe each stage of the pipeline, including intermediate outputs and final summaries <-)

# Future Work

- Replacing the current rule-based sentence classification with transformer-based models fine-tuned for medical report section tagging
- Adding confidence calibration, uncertainty estimation, and error analysis tools across the pipeline
- Extending the image pipeline beyond modality classification by exploring image-level finding detection and disease pattern recognition as a separate module.

---

## Final Notes
MedScan is an ongoing learning and research project developed to explore multimodal medical report analysis using a combination of traditional processing techniques, deep learning, and large language models.
The repository is shared to document design decisions, experimentation, and system-level thinking rather than to present a finished clinical product. Feedback, discussions, and research-oriented contributions are welcome.
If you are reviewing this project as part of an academic, learning, or portfolio context, feel free to reach out for clarification on design choices or future directions.








