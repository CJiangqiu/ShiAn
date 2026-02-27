"""统一数据预处理脚本：自动扫描原始数据集，过滤空标注并做类名映射"""
from __future__ import annotations
import argparse
import ast
import shutil
from pathlib import Path
from typing import Iterable, Optional
import yaml


def train_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = train_root()
GLOBAL_CLASSES_PATH = ROOT / "config" / "global_classes.yaml"
MODEL_ORIGINAL_DIR = ROOT / "models" / "original"
DATA_ORIGINAL_DIR = ROOT / "data" / "original"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
QWEN_TEXT_MODEL_DIR = MODEL_ORIGINAL_DIR / "qwen3-0.6b"  # Qwen3-0.6B文本模型（类名匹配）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 全局类别表（按场景分类）
DEFAULT_GLOBAL_CLASSES = {
    "common": ["person"],
    "worksite": [
        "helmet",
        "vest",
        "gloves",
        "goggles",
        "scaffold",
        "crane",
        "ladder",
        "barrier",
        "boots",
    ],
    "kitchen": ["fire", "smoke", "stove", "knife", "pot", "pan"],
    "vehicle": ["vehicle", "forklift", "truck"],
    "zone": ["restricted_zone", "safe_zone"],
}


def normalize_class_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def get_all_global_classes(scene_classes: dict[str, list[str]]) -> list[str]:
    """获取所有全局类别（扁平化）"""
    all_classes = []
    seen = set()
    for classes in scene_classes.values():
        for cls_name in classes:
            if cls_name not in seen:
                all_classes.append(cls_name)
                seen.add(cls_name)
    return all_classes


def create_global_classes_yaml() -> None:
    """创建全局类别表文件"""
    GLOBAL_CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "scenes": DEFAULT_GLOBAL_CLASSES,
        "all_classes": get_all_global_classes(DEFAULT_GLOBAL_CLASSES),
    }

    with open(GLOBAL_CLASSES_PATH, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, default_flow_style=False)

    print(f"已创建全局类别表: {GLOBAL_CLASSES_PATH}")


def check_global_classes() -> None:
    """检查全局类别表，不存在则创建"""
    if GLOBAL_CLASSES_PATH.exists():
        print(f"全局类别表已存在: {GLOBAL_CLASSES_PATH}")
    else:
        create_global_classes_yaml()


def load_global_classes() -> dict[str, list[str]]:
    """加载全局类别表"""
    check_global_classes()
    data = yaml.safe_load(GLOBAL_CLASSES_PATH.read_text(encoding="utf-8")) or {}
    scenes = data.get("scenes", {}) or {}
    all_classes = data.get("all_classes")
    if not all_classes:
        all_classes = get_all_global_classes(scenes)
    return {"scenes": scenes, "all_classes": all_classes}


def load_qwen_model():
    """加载 Qwen3-0.6B 文本模型（用于类名语义匹配）"""
    if not QWEN_TEXT_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Qwen3-0.6B 模型不存在: {QWEN_TEXT_MODEL_DIR}\n"
            "请先运行: python train/scripts/1_download_models.py"
        )

    print("加载 Qwen3-0.6B 模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_TEXT_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_TEXT_MODEL_DIR), device_map="auto"
    )

    print("Qwen3-0.6B 模型加载完成")
    return tokenizer, model


def match_class_with_llm(
    src_class: str, global_classes: list[str], tokenizer, model
) -> str | None:
    """使用 LLM 进行语义匹配"""
    prompt = f"""你是一个标签匹配助手。给定一个源标签和一个目标标签列表，找出语义最匹配的目标标签。

源标签: "{src_class}"
目标标签列表: {global_classes}

规则:
1. 如果源标签与某个目标标签是同义词或近义词，返回那个目标标签
2. 如果源标签是目标标签的变体（如复数、大小写不同），返回对应的目标标签
3. 如果找不到任何匹配，返回 "NO_MATCH"

只返回一个词，不要解释。
匹配结果:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = response[len(prompt) :].strip().split()[0] if response[len(prompt) :].strip() else "NO_MATCH"

    if result in global_classes:
        return result
    if result == "NO_MATCH":
        return None

    result_lower = result.lower()
    for gc in global_classes:
        if gc.lower() == result_lower:
            return gc
    return None


def load_data_config(data_yaml: Path) -> dict:
    """加载数据集 data.yaml"""
    try:
        return yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        return {}


def read_dataset_classes(data_yaml: Path, data_config: Optional[dict] = None) -> list[str]:
    """从数据集的 data.yaml 读取类别"""
    if data_config:
        names = data_config.get("names")
        if isinstance(names, list):
            return names
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]

    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_yaml}")

    content = data_yaml.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    for line in lines:
        s = line.strip()
        if s.startswith("names:") and "[" in s and "]" in s:
            rhs = s.split("names:", 1)[1].strip()
            v = ast.literal_eval(rhs)
            if isinstance(v, list):
                return v

    names = {}
    in_names = False
    for line in lines:
        if line.strip() == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        s = line.strip()
        if not s or ":" not in s:
            break
        k, v = s.split(":", 1)
        try:
            idx = int(k.strip())
            names[idx] = v.strip().strip("'\"")
        except ValueError:
            break

    if names:
        return [names[i] for i in sorted(names.keys())]

    raise RuntimeError(f"无法解析类别: {data_yaml}")


def resolve_dataset_root(data_config: dict, data_yaml: Path) -> Path:
    root = None
    if isinstance(data_config, dict):
        root = data_config.get("path")
    if root:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = (data_yaml.parent / root_path).resolve()
        if root_path.exists():
            return root_path
    return data_yaml.parent


def normalize_split_paths(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def get_split_paths(data_config: dict, split: str) -> list[str]:
    if not isinstance(data_config, dict):
        return []
    key = split
    if split == "val" and "val" not in data_config:
        for alt in ("valid", "validation"):
            if alt in data_config:
                key = alt
                break
    return normalize_split_paths(data_config.get(key))


def resolve_split_path(dataset_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (dataset_root / path).resolve()
    if path.exists():
        return path
    # Fallback for moved data.yaml (e.g., Roboflow exports with "../train/images")
    if not Path(raw_path).is_absolute():
        alt = raw_path
        while alt.startswith("../") or alt.startswith("..\\"):
            alt = alt[3:]
        if alt.startswith("./") or alt.startswith(".\\"):
            alt = alt[2:]
        if alt:
            alt_path = (dataset_root / alt).resolve()
            if alt_path.exists():
                return alt_path
    return path


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_dir = Path(*parts[:-1])
        return label_dir / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def infer_labels_dir(img_dir: Path) -> Optional[Path]:
    parts = list(img_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        candidate = Path(*parts)
        if candidate.exists():
            return candidate
    candidate = img_dir.parent / "labels"
    if candidate.exists():
        return candidate
    return None


def iter_image_label_pairs(split_path: Path) -> Iterable[tuple[Path, Optional[Path]]]:
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            img = line.strip()
            if not img:
                continue
            img_path = Path(img)
            if not img_path.is_absolute():
                img_path = (split_path.parent / img_path).resolve()
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            yield img_path, infer_label_path(img_path)
        return

    if split_path.is_dir():
        img_dir = split_path
        if split_path.name != "images" and (split_path / "images").exists():
            img_dir = split_path / "images"
        labels_dir = infer_labels_dir(img_dir)
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = (
                labels_dir / f"{img_path.stem}.txt" if labels_dir else infer_label_path(img_path)
            )
            yield img_path, lbl_path


def can_match_all(src_classes: list[str], global_classes: list[str]) -> bool:
    global_map = {normalize_class_name(gc): gc for gc in global_classes}
    for src in src_classes:
        if normalize_class_name(src) not in global_map:
            return False
    return True


def build_class_mapping(
    src_classes: list[str],
    global_classes: list[str],
    tokenizer=None,
    model=None,
) -> dict[str, str]:
    """构建类别映射表"""
    mapping = {}
    failed = []

    print("\n类别映射:")
    for src in src_classes:
        src_norm = normalize_class_name(src)
        matched = None

        for gc in global_classes:
            if normalize_class_name(gc) == src_norm:
                matched = gc
                break

        if matched is None and tokenizer is not None and model is not None:
            matched = match_class_with_llm(src, global_classes, tokenizer, model)

        if matched:
            mapping[src] = matched
            status = "精确" if normalize_class_name(src) == normalize_class_name(matched) else "LLM"
            print(f"  [{status}] \"{src}\" -> \"{matched}\"")
        else:
            failed.append(src)
            print(f"  [失败] \"{src}\" -> 无匹配")

    if failed:
        raise RuntimeError(
            "以下类别无法匹配到全局表:\n"
            + "\n".join(f"  - {c}" for c in failed)
            + "\n\n请更新全局类别表或检查数据集"
        )

    return mapping


def process_dataset(
    dataset_name: str,
    data_config: dict,
    dataset_root: Path,
    output_dir: Path,
    target_classes: list[str],
    class_mapping: dict[str, str],
    src_classes: list[str],
    clear_output: bool = True,
) -> dict:
    """处理数据集：过滤空标注、重新映射"""
    target_id_map = {c: i for i, c in enumerate(target_classes)}

    src_to_target_id = {}
    for i, src in enumerate(src_classes):
        mapped = class_mapping.get(src)
        if mapped and mapped in target_id_map:
            src_to_target_id[i] = target_id_map[mapped]

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "empty_filtered": 0, "boxes": 0}

    for split in ["train", "val", "test"]:
        split_paths = get_split_paths(data_config, split)
        if not split_paths:
            continue

        for raw_path in split_paths:
            split_path = resolve_split_path(dataset_root, raw_path)
            for img_path, lbl_path in iter_image_label_pairs(split_path):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue

                stats["total"] += 1

                new_lines = []
                if lbl_path and lbl_path.exists():
                    for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            old_id = int(float(parts[0]))
                            new_id = src_to_target_id.get(old_id)
                            if new_id is not None:
                                parts[0] = str(new_id)
                                new_lines.append(" ".join(parts))
                        except ValueError:
                            continue

                if not new_lines:
                    stats["empty_filtered"] += 1
                    continue

                out_stem = f"{dataset_name}__{img_path.stem}"
                out_img = output_dir / "images" / split / f"{out_stem}{img_path.suffix.lower()}"
                out_lbl = output_dir / "labels" / split / f"{out_stem}.txt"

                shutil.copy2(img_path, out_img)
                out_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

                stats["kept"] += 1
                stats["boxes"] += len(new_lines)

    yaml_content = (
        f"path: {output_dir.absolute()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(target_classes)}\n"
        "names:\n"
    )
    for i, c in enumerate(target_classes):
        yaml_content += f"  {i}: {c}\n"

    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    return stats


def find_dataset_dirs(base_dir: Path) -> list[Path]:
    """查找包含 data.yaml 的数据集目录"""
    datasets = []
    if not base_dir.exists():
        return []

    for data_yaml in base_dir.rglob("data.yaml"):
        if data_yaml.is_file():
            datasets.append(data_yaml.parent)

    return sorted(set(datasets))


def get_expert_id(dataset_dir: Path) -> str:
    """从数据集目录推断专家ID（取原始数据目录的第一层）"""
    try:
        rel = dataset_dir.relative_to(DATA_ORIGINAL_DIR)
    except ValueError:
        return dataset_dir.name
    return rel.parts[0] if rel.parts else dataset_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一数据预处理")
    parser.add_argument(
        "--dataset",
        action="append",
        help="仅处理指定数据集目录名（可重复）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("统一数据预处理")
    print("=" * 60)

    dataset_dirs = find_dataset_dirs(DATA_ORIGINAL_DIR)
    if not dataset_dirs:
        raise FileNotFoundError(
            f"在 {DATA_ORIGINAL_DIR} 下未找到包含 data.yaml 的数据集"
        )

    if args.dataset:
        wanted = set(args.dataset)
        dataset_dirs = [
            d for d in dataset_dirs
            if d.name in wanted or get_expert_id(d) in wanted
        ]
        if not dataset_dirs:
            raise FileNotFoundError("未找到匹配的数据集目录")

    global_config = load_global_classes()
    global_classes = global_config["all_classes"]

    tokenizer = None
    model = None

    datasets_by_expert: dict[str, list[Path]] = {}
    for ds in dataset_dirs:
        datasets_by_expert.setdefault(get_expert_id(ds), []).append(ds)

    print(f"\n找到 {len(dataset_dirs)} 个数据集（{len(datasets_by_expert)} 个专家）:")
    for expert_id, dirs in datasets_by_expert.items():
        names = ", ".join(d.name for d in dirs)
        print(f"  - {expert_id}: {names}")

    for expert_id, dirs in datasets_by_expert.items():
        print("\n" + "-" * 60)
        print(f"处理专家: {expert_id}")
        output_dir = DATA_PROCESSED_DIR / expert_id

        dataset_items = []
        mapped_union = set()

        for dataset_dir in dirs:
            data_yaml = dataset_dir / "data.yaml"
            data_config = load_data_config(data_yaml)
            src_classes = read_dataset_classes(data_yaml, data_config)
            print(f"  数据集: {dataset_dir.name} | 类别: {src_classes}")

            if not can_match_all(src_classes, global_classes):
                if tokenizer is None or model is None:
                    tokenizer, model = load_qwen_model()

            class_mapping = build_class_mapping(
                src_classes, global_classes, tokenizer, model
            )
            mapped_union.update(class_mapping.values())
            dataset_root = resolve_dataset_root(data_config, data_yaml)
            dataset_items.append(
                (dataset_dir.name, data_config, dataset_root, class_mapping, src_classes)
            )

        target_classes = [c for c in global_classes if c in mapped_union]
        if not target_classes:
            raise RuntimeError(f"专家 {expert_id} 未匹配到任何目标类别")

        print(f"输出目录: {output_dir}")
        for idx, (dataset_name, data_config, dataset_root, class_mapping, src_classes) in enumerate(dataset_items):
            stats = process_dataset(
                dataset_name,
                data_config,
                dataset_root,
                output_dir,
                target_classes,
                class_mapping,
                src_classes,
                clear_output=(idx == 0),
            )

            print(f"  图片: {stats['kept']}/{stats['total']} (过滤空标注: {stats['empty_filtered']})")
            print(f"  标注框: {stats['boxes']}")

    print("\n" + "=" * 60)
    print("预处理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
