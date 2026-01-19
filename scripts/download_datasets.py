#!/usr/bin/env python3
"""
Dataset Download Script

Downloads and organizes plant disease datasets for training.

Supported datasets:
- PlantVillage (Kaggle)
- PlantDoc (Real-world images)
- iNaturalist (via HF Hub)

Usage:
    python scripts/download_datasets.py --dataset plantvillage --output data/raw
    python scripts/download_datasets.py --dataset all --output data/raw
"""

import argparse
import os
from pathlib import Path
import shutil
from tqdm import tqdm


def download_plantvillage(output_dir: Path) -> Path:
    """
    Download PlantVillage dataset.

    PlantVillage contains 54,306 images of 38 crop-disease combinations.
    Source: https://www.kaggle.com/datasets/emmarex/plantdisease

    Note: Requires kaggle CLI configured with API credentials.
    """
    from datasets import load_dataset

    print("Downloading PlantVillage dataset...")

    # Try loading from HF Hub first (easier)
    try:
        dataset = load_dataset("plantvillage", split="train")
        save_path = output_dir / "plantvillage"
        dataset.save_to_disk(str(save_path))
        print(f"PlantVillage saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to load from HF Hub: {e}")

    # Fallback to Kaggle
    print("Attempting Kaggle download...")
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            "emmarex/plantdisease",
            path=str(output_dir),
            unzip=True
        )
        print(f"PlantVillage downloaded to {output_dir}")
        return output_dir / "plantdisease"
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Please download manually from:")
        print("https://www.kaggle.com/datasets/emmarex/plantdisease")
        return None


def download_plantdoc(output_dir: Path) -> Path:
    """
    Download PlantDoc dataset.

    PlantDoc contains real-world plant disease images with more variety
    than the controlled PlantVillage images.
    """
    from datasets import load_dataset

    print("Downloading PlantDoc dataset...")

    try:
        dataset = load_dataset("jxie/plantdoc", split="train")
        save_path = output_dir / "plantdoc"
        dataset.save_to_disk(str(save_path))
        print(f"PlantDoc saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download PlantDoc: {e}")
        return None


def download_inaturalist_plants(output_dir: Path, num_samples: int = 10000) -> Path:
    """
    Download plant observations from iNaturalist.

    iNaturalist has millions of plant observations with expert labels.
    We download a subset for training.
    """
    from datasets import load_dataset

    print(f"Downloading iNaturalist plants (up to {num_samples} samples)...")

    try:
        # Load iNaturalist dataset (plants subset)
        dataset = load_dataset(
            "inaturalist",
            split=f"train[:{num_samples}]",
            # Filter for plants
        )
        save_path = output_dir / "inaturalist"
        dataset.save_to_disk(str(save_path))
        print(f"iNaturalist saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download iNaturalist: {e}")
        print("Note: iNaturalist requires specific configuration")
        return None


def create_unified_label_mapping(datasets: list[Path]) -> dict:
    """
    Create a unified label mapping across all datasets.

    This ensures consistent class indices across different data sources.
    """
    all_labels = set()

    for dataset_path in datasets:
        if dataset_path is None:
            continue

        # Load and extract labels
        from datasets import load_from_disk
        try:
            dataset = load_from_disk(str(dataset_path))
            if hasattr(dataset, "features") and "label" in dataset.features:
                labels = dataset.features["label"].names
                all_labels.update(labels)
        except Exception as e:
            print(f"Could not extract labels from {dataset_path}: {e}")

    # Create mapping
    sorted_labels = sorted(all_labels)
    mapping = {label: idx for idx, label in enumerate(sorted_labels)}

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Download plant disease datasets")
    parser.add_argument(
        "--dataset",
        choices=["plantvillage", "plantdoc", "inaturalist", "all"],
        default="plantvillage",
        help="Dataset to download"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory"
    )
    parser.add_argument(
        "--inaturalist-samples",
        type=int,
        default=10000,
        help="Number of iNaturalist samples to download"
    )
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    downloaded = []

    if args.dataset in ["plantvillage", "all"]:
        path = download_plantvillage(args.output)
        if path:
            downloaded.append(path)

    if args.dataset in ["plantdoc", "all"]:
        path = download_plantdoc(args.output)
        if path:
            downloaded.append(path)

    if args.dataset in ["inaturalist", "all"]:
        path = download_inaturalist_plants(args.output, args.inaturalist_samples)
        if path:
            downloaded.append(path)

    # Create unified label mapping
    if downloaded:
        print("\nCreating unified label mapping...")
        mapping = create_unified_label_mapping(downloaded)

        import json
        mapping_path = args.output / "label_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Label mapping saved to {mapping_path}")
        print(f"Total classes: {len(mapping)}")

    print("\nDownload complete!")
    print(f"Downloaded datasets: {[str(p) for p in downloaded]}")


if __name__ == "__main__":
    main()
