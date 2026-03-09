from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 224


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def format_english_label(class_name: str) -> str:
    return class_name.replace("_", " ")


def build_default_labels(class_names: list[str]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for class_name in class_names:
        en = format_english_label(class_name)
        labels[class_name] = {
            "ko": en,
            "en": en,
            "display": f"{en} ({en})",
        }
    return labels


def load_artifacts(
    artifacts_dir: str, device: torch.device
) -> tuple[torch.nn.Module, list[str], dict[str, dict[str, str]]]:
    artifacts_path = Path(artifacts_dir)
    class_names_path = artifacts_path / "class_names.json"
    checkpoint_path = artifacts_path / "best_model.pt"
    labels_path = artifacts_path / "class_labels.json"

    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class names file: {class_names_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    with class_names_path.open("r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    model = build_model(len(class_names))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    labels = build_default_labels(class_names)
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        for class_name in class_names:
            entry = raw.get(class_name)
            if not isinstance(entry, dict):
                continue
            ko = str(entry.get("ko") or format_english_label(class_name))
            en = str(entry.get("en") or format_english_label(class_name))
            display = str(entry.get("display") or f"{ko} ({en})")
            labels[class_name] = {"ko": ko, "en": en, "display": display}

    return model, class_names, labels


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(
    *,
    image: Image.Image,
    model: torch.nn.Module,
    class_names: list[str],
    labels: dict[str, dict[str, str]],
    device: torch.device,
    top_k: int = 5,
) -> dict[str, Any]:
    transform = build_eval_transform()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    scores, indices = torch.topk(probs, k=k)

    predictions = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        class_name = class_names[idx]
        label = labels.get(class_name) or {
            "ko": format_english_label(class_name),
            "en": format_english_label(class_name),
            "display": f"{format_english_label(class_name)} ({format_english_label(class_name)})",
        }
        predictions.append(
            {
                "class_name": class_name,
                "display_name": label["display"],
                "ko_name": label["ko"],
                "en_name": label["en"],
                "probability": round(float(score), 6),
            }
        )

    return {
        "top_prediction": predictions[0],
        "top_k": predictions,
    }
