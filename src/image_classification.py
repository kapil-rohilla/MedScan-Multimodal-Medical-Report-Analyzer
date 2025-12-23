import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import timm

# DETECT IMAGE TYPE (grayscale / colored)
def detect_image_type(image):
    if image.mode == "L":
        return "grayscale"

    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image)
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean()
    diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2]).mean()
    diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2]).mean()

    if diff_rg < 5 and diff_gb < 5 and diff_rb < 5:
        return "grayscale"
    return "colored"

# per-model formatting rules
def convert_for_model(image, model_key):
    model_formats = {
        "ct": "grayscale",
        "mri": "grayscale",
        "xray": "grayscale",
        "ecg": "grayscale",
        "ultrasound": "mixed",
        "nonmedical": "colored",
    }

    expected = model_formats[model_key]
    detected = detect_image_type(image)

    if expected == detected:
        if detected == "grayscale":
            gray = image.convert("L")
            rgb = Image.new("RGB", gray.size)
            for _ in range(3):
                rgb.paste(gray, (0, 0))
            return rgb
        else:
            return image.convert("RGB")

    if expected == "grayscale" and detected == "colored":
        gray = image.convert("L")
        rgb = Image.new("RGB", gray.size)
        for _ in range(3):
            rgb.paste(gray, (0, 0))
        return rgb

    if expected == "colored" and detected == "grayscale":
        gray = image.convert("L")
        rgb = Image.new("RGB", gray.size)
        for _ in range(3):
            rgb.paste(gray, (0, 0))
        return rgb

    return image.convert("RGB")


# EfficientNet-B0 Binary Model Class
class BinaryMedicalClassifier(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False)
        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.backbone(x)

# MAIN CLASS: Loads All 6 Models + Applies Decision Logic

class MedicalImageClassifier:
    def __init__(self, base_path):
        
        self.model_paths = {
            "ct": os.path.join(base_path, "ct_classifier_final.pth"),
            "mri": os.path.join(base_path, "mri_classifier_final.pth"),
            "xray": os.path.join(base_path, "xray_classifier_final.pth"),
            "ultrasound": os.path.join(base_path, "ultrasound_classifier_final.pth"),
            "ecg": os.path.join(base_path, "ecg_classifier_final.pth"),
            "nonmedical": os.path.join(base_path, "nonmedical_classifier_final.pth"),
        }

        self.model_names = {
            "ct": "CT",
            "mri": "MRI",
            "xray": "X-ray",
            "ultrasound": "Ultrasound",
            "ecg": "ECG",
            "nonmedical": "Non-Medical",
        }

        self.device = torch.device("cpu")

        self.models = {k: self._load_model(v) for k, v in self.model_paths.items()}

        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def _load_model(self, path):
        model = BinaryMedicalClassifier()
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model
    
    # Predict using ALL MODELS + Smart Preprocessing
    def predict(self, image_path):
        image = Image.open(image_path)

        results = {}
        medical_models = ["ct", "mri", "xray", "ecg", "ultrasound"]

        with torch.no_grad():
            for key, model in self.models.items():

                img_conv = convert_for_model(image, key)
                img_tensor = self.tfms(img_conv).unsqueeze(0).to(self.device)

                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = probs.max(1)

                is_positive = pred_idx.item() == 1

                results[key] = {
                    "prediction": self.model_names[key] if is_positive else f"Not-{self.model_names[key]}",
                    "confidence": float(conf.item()),
                    "positive": is_positive
                }

        medical_hits = {k: v["confidence"] for k, v in results.items()
                        if v["positive"] and k in medical_models}

        nonmedical_hit = results["nonmedical"]["confidence"] if results["nonmedical"]["positive"] else None

        # If any medical model is positive --> chooses highest confidence
        if medical_hits:
            best_key = max(medical_hits, key=medical_hits.get)
            return (
                self.model_names[best_key],
                medical_hits[best_key],
                results
            )

        # Only Non-medical is positive
        if nonmedical_hit:
            return (
                "Non-Medical",
                nonmedical_hit,
                results
            )

        # No positive predictions --> fallback to highest confidence
        best_key = max(results, key=lambda k: results[k]["confidence"])
        best_result = results[best_key]

        return (
            "Unknown",
            best_result["confidence"],
            results
        )