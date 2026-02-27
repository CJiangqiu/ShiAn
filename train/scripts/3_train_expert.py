"""Train expert YOLO models from processed datasets."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def train_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = train_root()
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "models" / "output"
CONFIG_DIR = ROOT / "config"


def load_yaml(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: dict[str, Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def infer_scenes(expected_classes: list[str], global_classes: dict[str, Any]) -> list[str]:
    scenes_mapping = global_classes.get("scenes", {}) or {}
    scene_votes: dict[str, int] = {}

    class_to_scenes: dict[str, list[str]] = {}
    for scene_name, scene_classes in scenes_mapping.items():
        for cls_name in scene_classes:
            class_to_scenes.setdefault(cls_name, []).append(scene_name)

    for cls_name in expected_classes:
        for scene in class_to_scenes.get(cls_name, []):
            if scene != "common":
                scene_votes[scene] = scene_votes.get(scene, 0) + 1

    sorted_scenes = sorted(scene_votes.items(), key=lambda x: x[1], reverse=True)
    result = [s for s, votes in sorted_scenes if votes >= 2]
    if not result and sorted_scenes:
        result = [sorted_scenes[0][0]]

    return result if result else ["general"]


def load_dataset_config(expert_id: str) -> tuple[Path, list[str]] | None:
    data_yaml_path = PROCESSED_DIR / expert_id / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[WARN] dataset config missing: {data_yaml_path}")
        return None

    data_config = load_yaml(data_yaml_path)

    names = data_config.get("names")
    if isinstance(names, dict):
        expected_classes = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        expected_classes = names
    else:
        print(f"[WARN] unable to parse class names: {data_yaml_path}")
        return None

    return data_yaml_path, expected_classes


def load_expert_config(expert_id: str) -> dict[str, Any]:
    expert_config_path = CONFIG_DIR / "experts" / f"{expert_id}.yaml"
    if not expert_config_path.exists():
        print(f"[WARN] expert config not found: {expert_config_path}")
        return {}
    return load_yaml(expert_config_path)


def execute_yolo_training(
    expert_id: str,
    data_yaml_path: Path,
    base_model: Path,
    train_config: dict[str, Any],
) -> bool:
    try:
        from ultralytics import YOLO

        print("\n[Training params]")
        print(f"  epochs: {train_config['epochs']}")
        print(f"  batch: {train_config['batch_size']}")
        print(f"  imgsz: {train_config['imgsz']}")
        print(f"  device: {train_config['device']}")
        print(f"  lr0: {train_config.get('lr0', 0.01)}")

        model = YOLO(str(base_model))
        model.train(
            data=str(data_yaml_path.absolute()),
            epochs=train_config["epochs"],
            batch=train_config["batch_size"],
            imgsz=train_config["imgsz"],
            device=train_config["device"],
            patience=train_config.get("patience", 50),
            workers=train_config.get("workers", 8),
            optimizer=train_config.get("optimizer", "AdamW"),
            lr0=train_config.get("lr0", 0.01),
            lrf=train_config.get("lrf", 0.01),
            momentum=train_config.get("momentum", 0.937),
            weight_decay=train_config.get("weight_decay", 0.0005),
            hsv_h=train_config.get("hsv_h", 0.015),
            hsv_s=train_config.get("hsv_s", 0.7),
            hsv_v=train_config.get("hsv_v", 0.4),
            degrees=train_config.get("degrees", 10.0),
            translate=train_config.get("translate", 0.1),
            scale=train_config.get("scale", 0.5),
            shear=train_config.get("shear", 0.0),
            perspective=train_config.get("perspective", 0.0),
            flipud=train_config.get("flipud", 0.5),
            fliplr=train_config.get("fliplr", 0.5),
            mosaic=train_config.get("mosaic", 1.0),
            save=True,
            save_period=-1,
            plots=True,
            val=True,
            project=str(OUTPUT_DIR),
            name=expert_id,
            exist_ok=True,
        )

        print("[OK] training finished")
        return True

    except Exception as exc:
        print(f"[ERROR] training failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def print_training_metrics(expert_id: str) -> None:
    results_csv = OUTPUT_DIR / expert_id / "results.csv"
    if not results_csv.exists():
        print("[WARN] results.csv not found")
        return

    with open(results_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    last_epoch = rows[-1]
    print("\n[Metrics]")
    print(f"  epochs: {len(rows)}")

    metrics_map = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@0.5",
        "metrics/mAP50-95(B)": "mAP@0.5:0.95",
    }

    for key, label in metrics_map.items():
        value = last_epoch.get(key, last_epoch.get(key.strip()))
        if value:
            print(f"    {label}: {float(value):.4f}")


def export_expert_model(
    expert_id: str,
    scene_ids: list[str],
    expected_classes: list[str],
    expert_config: dict[str, Any],
    version: str,
) -> bool:
    output_dir = OUTPUT_DIR / expert_id

    best_pt = output_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, output_dir / "best.pt")
        print("[OK] exported best.pt")
    else:
        print(f"[WARN] best.pt not found: {best_pt}")
        return False

    expert_info: dict[str, Any] = {
        "expert_id": expert_id,
        "version": version,
        "scene_ids": scene_ids,
        "expected_classes": expected_classes,
    }

    if "description" in expert_config:
        expert_info["description"] = expert_config["description"]

    for key in ("detection_mode", "primary_object", "alert_rules", "zone"):
        if key in expert_config:
            expert_info[key] = expert_config[key]

    if "vlm_prompt_template" in expert_config:
        expert_info["vlm_prompt_template"] = expert_config["vlm_prompt_template"]

    if "alert_behavior" in expert_config:
        expert_info["alert_behavior"] = expert_config["alert_behavior"]

    expert_info_path = output_dir / "expert_info.json"
    save_json(expert_info, expert_info_path)
    print("[OK] exported expert_info.json")
    return True


def cleanup_expert_output(output_dir: Path) -> int:
    keep_files = {"best.pt", "expert_info.json"}
    cleaned = 0
    for item in output_dir.iterdir():
        if item.name in keep_files:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
            cleaned += 1
        else:
            item.unlink(missing_ok=True)
            cleaned += 1
    return cleaned


def train_expert(
    expert_id: str,
    train_config: dict[str, Any],
    global_classes: dict[str, Any],
) -> bool:
    print("\n" + "=" * 60)
    print(f"Training expert: {expert_id}")
    print("=" * 60 + "\n")

    dataset_result = load_dataset_config(expert_id)
    if dataset_result is None:
        return False

    data_yaml_path, expected_classes = dataset_result
    print(f"[1/5] classes: {', '.join(expected_classes)}")

    scene_ids = infer_scenes(expected_classes, global_classes)
    print(f"[2/5] scenes: {', '.join(scene_ids)}")

    expert_config = load_expert_config(expert_id)
    if expert_config:
        print("[3/5] expert config loaded")
    else:
        print("[3/5] expert config missing (VLM prompt disabled)")

    base_model = ROOT / train_config["base_model"].replace("train/", "")
    if not base_model.exists():
        print(f"[ERROR] base model missing: {base_model}")
        print("        run: python train/scripts/1_download_models.py")
        return False

    print("\n[4/5] start training...")
    success = execute_yolo_training(expert_id, data_yaml_path, base_model, train_config)
    if not success:
        return False

    print_training_metrics(expert_id)

    print("\n[5/5] export expert files...")
    export_ok = export_expert_model(
        expert_id,
        scene_ids,
        expected_classes,
        expert_config,
        train_config.get("version", "v1.0.0"),
    )
    if export_ok:
        cleaned = cleanup_expert_output(OUTPUT_DIR / expert_id)
        if cleaned > 0:
            print(f"  cleaned artifacts: {cleaned}")

    print("\n" + "=" * 60)
    print(f"[OK] expert {expert_id} training finished")
    print("=" * 60 + "\n")

    return True


def main() -> None:
    print("=" * 60)
    print("Train YOLO expert models")
    print("=" * 60)

    train_config_path = CONFIG_DIR / "train_config.yaml"
    if not train_config_path.exists():
        print(f"[ERROR] missing train_config.yaml: {train_config_path}")
        return

    config = load_yaml(train_config_path)
    global_config = config.get("global", {}) or {}
    exclude_experts = global_config.get("exclude_experts", []) or []

    train_config = {**global_config, **(config.get("expert", {}) or {})}

    global_classes_path = CONFIG_DIR / "global_classes.yaml"
    if not global_classes_path.exists():
        print(f"[ERROR] missing global_classes.yaml: {global_classes_path}")
        return

    global_classes = load_yaml(global_classes_path)
    print("[OK] loaded global_classes.yaml")

    if not PROCESSED_DIR.exists():
        print(f"[ERROR] processed data dir missing: {PROCESSED_DIR}")
        print("        run: python train/scripts/2_preprocess_dataset.py")
        return

    all_datasets = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    if not all_datasets:
        print("[ERROR] no processed datasets found")
        return

    datasets = [d for d in all_datasets if d not in exclude_experts]
    if not datasets:
        print("[ERROR] all datasets excluded")
        return

    print(f"\nfound {len(all_datasets)} datasets:")
    for dataset in all_datasets:
        status = "skip" if dataset in exclude_experts else "train"
        print(f"  - {dataset} [{status}]")

    print(f"\ntraining {len(datasets)} experts\n")

    success_count = 0
    failed_count = 0

    for idx, expert_id in enumerate(datasets, 1):
        print("=" * 60)
        print(f"progress: {idx}/{len(datasets)}")
        print("=" * 60)

        if train_expert(expert_id, train_config.copy(), global_classes):
            success_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 60)
    print("Training summary")
    print("=" * 60)
    print(f"datasets: {len(all_datasets)}")
    print(f"success: {success_count}")
    print(f"failed: {failed_count}")
    print(f"skipped: {len(exclude_experts)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
