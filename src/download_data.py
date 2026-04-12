"""
=============================================================================
download_data.py  –  Dataset Download Helper
=============================================================================
Downloads the Construction Site Safety dataset from Roboflow Universe
(public, free).  Also prints links to larger alternatives.
=============================================================================
Usage:
    python src/download_data.py --dest data/raw
=============================================================================
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

DATASET_INFO = {
    "name"       : "Construction Site Safety (Roboflow)",
    "classes"    : ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
                    "NO-Safety Vest", "Person", "Safety Cone",
                    "Safety Vest", "machinery", "vehicle"],
    "images_approx": 2605,
    "license"    : "CC BY 4.0",
    "roboflow_url": (
        "https://universe.roboflow.com/roboflow-universe-projects/"
        "construction-site-safety"
    ),
    # Pre-exported YOLOv8-format zip (change this to your own export link)
    "zip_url"    : None,   # set via --url arg or Roboflow API
}

ALTERNATIVE_DATASETS = [
    {
        "name"   : "PPE Detection Dataset (Kaggle)",
        "url"    : "https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow",
        "format" : "YOLO",
    },
    {
        "name"   : "COCO 2017 (persons subset)",
        "url"    : "https://cocodataset.org/#download",
        "format" : "COCO JSON",
    },
    {
        "name"   : "Open Images v7 – safety equipment",
        "url"    : "https://storage.googleapis.com/openimages/web/index.html",
        "format" : "CSV",
    },
]


def download_file(url: str, dest_path: str) -> None:
    print(f"  Downloading → {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path,
        reporthook=lambda b, bs, total: print(
            f"\r  {min(b*bs, total)}/{total} bytes", end="", flush=True
        )
    )
    print()


def unzip(zip_path: str, dest_dir: str) -> None:
    print(f"  Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)
    print(f"  Extracted to {dest_dir}")


def create_placeholder_structure(dest: str) -> None:
    """Create README placeholders so the repo structure is clear."""
    for split in ["train", "valid", "test"]:
        for sub in ["images", "labels"]:
            p = Path(dest) / split / sub
            p.mkdir(parents=True, exist_ok=True)
            (p / ".gitkeep").touch()

    readme = Path(dest) / "README.md"
    readme.write_text(
        "# Dataset Folder\n\n"
        "Place your YOLOv8-format dataset here.\n\n"
        "## Expected structure\n"
        "```\n"
        "data/raw/\n"
        "├── train/images/   ← training images\n"
        "├── train/labels/   ← YOLO .txt annotations\n"
        "├── valid/images/\n"
        "├── valid/labels/\n"
        "├── test/images/\n"
        "├── test/labels/\n"
        "└── data.yaml       ← class names + paths config\n"
        "```\n\n"
        "## Recommended dataset\n"
        f"{DATASET_INFO['roboflow_url']}\n\n"
        "## Alternatives\n"
        + "\n".join(f"- [{d['name']}]({d['url']})" for d in ALTERNATIVE_DATASETS)
    )
    print(f"  Placeholder structure created at {dest}/")


def write_data_yaml(dest: str) -> None:
    yaml_content = (
        f"path: {os.path.abspath(dest)}\n"
        "train: train/images\n"
        "val:   valid/images\n"
        "test:  test/images\n\n"
        "nc: 10\n"
        "names:\n"
        "  0: Hardhat\n"
        "  1: Mask\n"
        "  2: NO-Hardhat\n"
        "  3: NO-Mask\n"
        "  4: NO-Safety Vest\n"
        "  5: Person\n"
        "  6: Safety Cone\n"
        "  7: Safety Vest\n"
        "  8: machinery\n"
        "  9: vehicle\n"
    )
    yaml_path = os.path.join(dest, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  data.yaml written → {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset download helper")
    parser.add_argument("--dest", default="data/raw",
                        help="Destination directory for the dataset")
    parser.add_argument("--url", default=None,
                        help="Direct URL to the dataset ZIP (optional)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Construction Site Safety – Dataset Setup")
    print("=" * 60)

    if args.url:
        zip_path = os.path.join(args.dest, "dataset.zip")
        download_file(args.url, zip_path)
        unzip(zip_path, args.dest)
    else:
        print("\n[INFO] No direct URL provided.")
        print("       Creating placeholder directory structure …\n")
        create_placeholder_structure(args.dest)

    write_data_yaml(args.dest)

    print("\n[INFO] Alternative datasets you can use:")
    for d in ALTERNATIVE_DATASETS:
        print(f"  • {d['name']}")
        print(f"    {d['url']}")

    print("\n[DONE] Dataset folder is ready. See data/raw/README.md for details.")


if __name__ == "__main__":
    main()
