#!/usr/bin/env python3
"""
Upload Dataset to Hugging Face Hub

Uploads a prepared dataset to HF Hub for use with Modal training.

Usage:
    python scripts/upload_to_hub.py --input data/balanced --repo your-username/plant-disease-dataset
"""

import argparse
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict, Image
from huggingface_hub import HfApi, create_repo
import os


def load_image_folder_dataset(input_dir: Path) -> DatasetDict:
    """
    Load dataset from image folder structure.

    Expected structure:
    input_dir/
        class_name_1/
            image1.jpg
            image2.jpg
        class_name_2/
            image3.jpg
            ...
    """
    from datasets import load_dataset

    print(f"Loading dataset from {input_dir}...")

    # Load using imagefolder
    dataset = load_dataset("imagefolder", data_dir=str(input_dir))

    print(f"Loaded {len(dataset['train'])} training images")

    return dataset


def add_metadata(dataset: DatasetDict, label_mapping: dict = None) -> DatasetDict:
    """Add metadata columns to dataset."""

    def extract_metadata(example):
        """Extract crop and disease from label."""
        label_name = example.get("label_name", "")

        # Parse label format: "Crop___Disease" or "Crop___healthy"
        parts = label_name.split("___")
        if len(parts) == 2:
            crop = parts[0].replace("_", " ")
            disease = parts[1].replace("_", " ")
            is_healthy = disease.lower() == "healthy"
        else:
            crop = label_name
            disease = "unknown"
            is_healthy = False

        return {
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
        }

    # Add label names if not present
    if "label_name" not in dataset["train"].column_names:
        label_names = dataset["train"].features["label"].names

        def add_label_name(example):
            return {"label_name": label_names[example["label"]]}

        dataset = dataset.map(add_label_name)

    # Add metadata
    dataset = dataset.map(extract_metadata)

    return dataset


def create_train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Create train/validation/test splits."""

    # First split: separate test set
    train_test = dataset.train_test_split(test_size=test_ratio, seed=seed)

    # Second split: separate validation from training
    val_size = val_ratio / (1 - test_ratio)
    train_val = train_test["train"].train_test_split(test_size=val_size, seed=seed)

    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_test["test"],
    })


def upload_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    private: bool = False,
):
    """Upload dataset to Hugging Face Hub."""

    print(f"Uploading to {repo_id}...")

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Push to hub
    dataset.push_to_hub(repo_id, private=private)

    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HF Hub")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset directory"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HF Hub repository (e.g., your-username/plant-disease-dataset)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    args = parser.parse_args()

    # Check for HF token
    if not os.getenv("HUGGING_FACE_HUB_TOKEN") and not os.getenv("HF_TOKEN"):
        print("Warning: No HF Hub token found. Set HUGGING_FACE_HUB_TOKEN environment variable.")
        print("Get your token from: https://huggingface.co/settings/tokens")

    # Load dataset
    if (args.input / "dataset_dict.json").exists():
        # Already a HF dataset
        dataset = load_from_disk(str(args.input))
    else:
        # Image folder structure
        dataset = load_image_folder_dataset(args.input)

    # Add metadata
    dataset = add_metadata(dataset)

    # Create splits if only train exists
    if isinstance(dataset, DatasetDict) and "validation" not in dataset:
        print("Creating train/validation/test splits...")
        dataset = create_train_val_test_split(
            dataset["train"],
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    # Print dataset info
    print("\nDataset info:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")

    if "label" in dataset["train"].features:
        num_classes = len(dataset["train"].features["label"].names)
        print(f"  Classes: {num_classes}")

    # Upload
    upload_to_hub(dataset, args.repo, args.private)


if __name__ == "__main__":
    main()
