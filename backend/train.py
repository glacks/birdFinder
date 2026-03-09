from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

IMAGE_SIZE = 224


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def make_loaders(
    dataset_dir: str,
    batch_size: int,
    val_ratio: float,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf, val_tf = make_transforms()
    ds_for_classes = datasets.ImageFolder(root=dataset_dir)
    class_names = ds_for_classes.classes

    train_ds_full = datasets.ImageFolder(root=dataset_dir, transform=train_tf)
    val_ds_full = datasets.ImageFolder(root=dataset_dir, transform=val_tf)

    total = len(train_ds_full)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    val_count = max(1, int(total * val_ratio))
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_names


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def format_english_label(class_name: str) -> str:
    return class_name.replace("_", " ")


def save_default_labels(artifacts_dir: Path, class_names: list[str]) -> None:
    labels = {}
    for class_name in class_names:
        en = format_english_label(class_name)
        labels[class_name] = {
            "ko": en,
            "en": en,
            "display": f"{en} ({en})",
        }
    with (artifacts_dir / "class_labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = loss_fn(logits, y)
            else:
                logits = model(x)
                loss = loss_fn(logits, y)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        batch_size = y.size(0)

        total_loss += float(loss.item()) * batch_size
        total_correct += correct
        total_count += batch_size

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = make_loaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        workers=args.workers,
        seed=args.seed,
    )

    model = build_model(len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=None,
            device=device,
            scaler=None,
        )

        print(
            f"[{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), artifacts_dir / "best_model.pt")
            with (artifacts_dir / "class_names.json").open("w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)
            save_default_labels(artifacts_dir, class_names)
            print(f"Saved best model. val_acc={val_acc:.4f}")

    print(f"Training complete. best_val_acc={best_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bird classifier with PyTorch")
    parser.add_argument("--dataset-dir", default="../data_raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
