"""下载原始底模"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def train_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = train_root()
ORIGINAL_DIR = ROOT / "models" / "original"

YOLO_MODELS = ["yolo12s.pt", "yolo12n.pt"]

# Qwen3-0.6B: 数据预处理类名匹配（轻量级文本模型）
QWEN_TEXT_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN_TEXT_LOCAL_DIR = ORIGINAL_DIR / "qwen3-0.6b"

# Qwen3-VL-2B: 推理阶段本地VLM危险判断（视觉语言模型）
QWEN_VL_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_VL_LOCAL_DIR = ORIGINAL_DIR / "qwen3-vl-2b-instruct"


def ensure_deps() -> None:
    """确保依赖已安装"""
    try:
        import ultralytics  # type: ignore
    except ImportError:
        print("安装 ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

    try:
        import huggingface_hub  # type: ignore
    except ImportError:
        print("安装 huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

    try:
        import sklearn  # type: ignore
    except ImportError:
        print("安装 scikit-learn...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

    # 首先尝试导入torch，如果不存在则先安装CPU版本
    try:
        import torch  # type: ignore
    except ImportError:
        print("安装 torch (CPU版本)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])

    # 现在导入torch并检查CUDA
    import torch  # type: ignore
    if torch.cuda.is_available():
        print("检测到CUDA，确保安装GPU版本的PyTorch...")
        # 检查当前安装的是否已经是GPU版本
        if torch.version.cuda:
            print("GPU版本PyTorch已就绪")
        else:
            # 重新安装GPU版本
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118"])
            print("已安装GPU版本PyTorch")
    else:
        print("CUDA不可用，使用CPU版本的PyTorch")


def download_yolo_models() -> None:
    """下载 YOLO 底模"""
    from ultralytics.utils.downloads import attempt_download_asset  # type: ignore

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

    for model_file in YOLO_MODELS:
        dst = ORIGINAL_DIR / model_file

        if dst.exists():
            print(f"跳过（已存在）: {dst}")
            continue

        print(f"下载 {model_file}...")
        src = Path(attempt_download_asset(model_file, release="v8.3.0"))

        if not src.exists():
            raise FileNotFoundError(f"下载失败: {model_file}")

        dst.write_bytes(src.read_bytes())
        print(f"已保存: {dst}")


def download_qwen_text_model() -> None:
    """下载 Qwen3-0.6B 文本模型（用于数据预处理类名匹配）"""
    # 检查模型权重文件是否存在
    model_file = QWEN_TEXT_LOCAL_DIR / "model.safetensors"
    if model_file.exists():
        print(f"跳过（已存在）: {QWEN_TEXT_LOCAL_DIR}")
        return

    print(f"下载 {QWEN_TEXT_MODEL_ID}...")
    from huggingface_hub import hf_hub_download, list_repo_files

    QWEN_TEXT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id=QWEN_TEXT_MODEL_ID)
    total = len(files)

    for i, filename in enumerate(files, 1):
        print(f"[{i}/{total}] 下载: {filename}", flush=True)
        hf_hub_download(
            repo_id=QWEN_TEXT_MODEL_ID,
            filename=filename,
            local_dir=str(QWEN_TEXT_LOCAL_DIR),
        )

    print(f"已保存: {QWEN_TEXT_LOCAL_DIR}")


def download_qwen_vl_model() -> None:
    """下载 Qwen3-VL-2B 视觉模型（用于推理阶段危险判断）"""
    # 检查模型权重文件是否存在（不只是目录）
    model_file = QWEN_VL_LOCAL_DIR / "model.safetensors.index.json"
    if model_file.exists():
        print(f"跳过（已存在）: {QWEN_VL_LOCAL_DIR}")
        return

    print(f"下载 {QWEN_VL_MODEL_ID}（约5-6GB，需要一定时间）...")
    from huggingface_hub import hf_hub_download, list_repo_files
    import sys

    QWEN_VL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 获取文件列表
    files = list_repo_files(repo_id=QWEN_VL_MODEL_ID)
    total = len(files)

    for i, filename in enumerate(files, 1):
        print(f"[{i}/{total}] 下载: {filename}", flush=True)
        hf_hub_download(
            repo_id=QWEN_VL_MODEL_ID,
            filename=filename,
            local_dir=str(QWEN_VL_LOCAL_DIR),
        )

    print(f"已保存: {QWEN_VL_LOCAL_DIR}")


def main() -> None:
    print("=" * 60)
    print("下载原始底模及依赖")
    print("=" * 60 + "\n")

    ensure_deps()

    print("\n[1/3] 下载 YOLO 模型...")
    download_yolo_models()

    print("\n[2/3] 下载 Qwen3-0.6B 文本模型（数据预处理）...")
    download_qwen_text_model()

    print("\n[3/3] 下载 Qwen3-VL-2B 视觉模型（推理危险判断）...")
    download_qwen_vl_model()

    print("\n" + "=" * 60)
    print("下载完成")
    print("=" * 60)
    print(f"模型目录: {ORIGINAL_DIR}")
    print(f"  - YOLO: yolo12n.pt, yolo12s.pt")
    print(f"  - Qwen3-0.6B: {QWEN_TEXT_LOCAL_DIR.name}")
    print(f"  - Qwen3-VL-2B: {QWEN_VL_LOCAL_DIR.name}")
    print("环境已准备就绪")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
