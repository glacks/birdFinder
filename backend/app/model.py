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


def load_artifacts(artifacts_dir: str, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    artifacts_path = Path(artifacts_dir)
    class_names_path = artifacts_path / "class_names.json"
    checkpoint_path = artifacts_path / "best_model.pt"

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

    return model, class_names


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
        predictions.append(
            {
                "class_name": class_names[idx],
                "probability": round(float(score), 6),
            }
        )

    return {
        "top_prediction": predictions[0],
        "top_k": predictions,
    }
