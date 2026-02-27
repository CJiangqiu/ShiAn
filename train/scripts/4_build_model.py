"""
模型导出脚本
将训练好的专家模型和路由模型从 train/models/output/ 导出到 server/models/
"""

import json
import shutil
from pathlib import Path


def project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
TRAIN_OUTPUT = ROOT / "train" / "models" / "output"
TRAIN_ORIGINAL = ROOT / "train" / "models" / "original"
TRAIN_CONFIG = ROOT / "train" / "config"
SERVER_MODELS = ROOT / "server" / "models"
CLIENT_CONFIG = ROOT / "client" / "config"
TRAIN_SCRIPTS_RUNS = ROOT / "train" / "scripts" / "runs"


def load_json(file_path: Path) -> dict:
    """加载 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"    错误: 无法解析 JSON: {e}")
        return None


def validate_expert_info(info: dict, expert_id: str) -> bool:
    """验证专家元信息"""
    if info.get("expert_id") != expert_id:
        print(f"    错误: expert_id 不匹配: 期望 {expert_id}, 实际 {info.get('expert_id')}")
        return False

    if not isinstance(info.get("version"), str) or not info["version"]:
        print(f"    错误: version 字段无效")
        return False

    if not isinstance(info.get("scene_ids"), list):
        print(f"    错误: scene_ids 字段无效")
        return False

    if not isinstance(info.get("expected_classes"), list):
        print(f"    错误: expected_classes 字段无效")
        return False

    return True




def cleanup_training_files(expert_dir: Path) -> int:
    """清理训练中间文件（保留results.csv和关键图表）"""
    cleanup_items = [
        expert_dir / "weights" / "last.pt",
        expert_dir / "labels.jpg",
        expert_dir / "labels_correlogram.jpg",
        expert_dir / "args.yaml",
        expert_dir / "weights",
    ]

    cleaned_count = 0
    for item in cleanup_items:
        if item.exists():
            try:
                if item.is_file():
                    item.unlink()
                    cleaned_count += 1
                elif item.is_dir() and not any(item.iterdir()):
                    item.rmdir()
                    cleaned_count += 1
            except Exception as e:
                print(f"      警告: 清理失败 {item.name}: {e}")

    return cleaned_count


def cleanup_legacy_runs() -> bool:
    """清理历史遗留的 runs 目录"""
    if not TRAIN_SCRIPTS_RUNS.exists():
        return False

    try:
        shutil.rmtree(TRAIN_SCRIPTS_RUNS)
        print(f"  已清理历史遗留目录: {TRAIN_SCRIPTS_RUNS}")
        return True
    except Exception as e:
        print(f"  警告: 清理 runs 目录失败: {e}")
        return False


def export_expert(expert_dir: Path, cleanup: bool = True) -> bool:
    """导出单个专家模型"""
    expert_id = expert_dir.name

    print(f"\n处理专家: {expert_id}")

    # 1. 检查 YOLO 模型文件
    best_pt = expert_dir / "best.pt"
    if not best_pt.exists():
        # 尝试从 weights 子目录查找
        weights_best = expert_dir / "weights" / "best.pt"
        if weights_best.exists():
            best_pt = weights_best
        else:
            print(f"    跳过: best.pt 不存在")
            return False

    # 2. 检查并验证 expert_info.json
    info_path = expert_dir / "expert_info.json"
    if not info_path.exists():
        print(f"    跳过: expert_info.json 不存在")
        return False

    info = load_json(info_path)
    if info is None:
        return False

    if not validate_expert_info(info, expert_id):
        return False

    # 3. 导出到 server/models/
    dst_dir = SERVER_MODELS / expert_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 复制 YOLO 模型文件
    dst_model = dst_dir / f"{expert_id}.pt"
    shutil.copy2(best_pt, dst_model)

    # 复制专家元信息
    shutil.copy2(info_path, dst_dir / "expert_info.json")

    classes_str = ', '.join(info['expected_classes'])
    print(f"    ✓ 导出完成: {dst_model}")
    print(f"      - 场景: {', '.join(info['scene_ids'])}")
    print(f"      - 类别: [{classes_str}]")

    # 4. 清理训练中间文件
    if cleanup:
        cleaned = cleanup_training_files(expert_dir)
        if cleaned > 0:
            print(f"    已清理 {cleaned} 个训练文件")

    return True




def export_vlm_models() -> int:
    """导出VLM模型"""
    print(f"\n处理VLM模型")

    # 直接复制 qwen3-vl-2b-instruct
    vlm_name = "qwen3-vl-2b-instruct"
    src_dir = TRAIN_ORIGINAL / vlm_name
    dst_dir = SERVER_MODELS / vlm_name

    if not src_dir.exists():
        print(f"    跳过: {vlm_name} 不存在")
        return 0

    if dst_dir.exists():
        print(f"    ○ 已存在: {vlm_name}")
        return 1

    print(f"    复制中: {vlm_name}")
    try:
        shutil.copytree(src_dir, dst_dir)
        print(f"    ✓ 导出完成: {vlm_name}")
        return 1
    except Exception as e:
        print(f"    错误: 复制失败: {e}")
        return 0


def export_client_config() -> int:
    """导出客户端运行时配置"""
    print(f"\n处理客户端配置")

    config_files = ["config.yaml", "vlm_config.yaml"]
    exported = 0

    CLIENT_CONFIG.mkdir(parents=True, exist_ok=True)

    for filename in config_files:
        src = TRAIN_CONFIG / filename
        dst = CLIENT_CONFIG / filename

        if not src.exists():
            print(f"    跳过: {filename} 不存在于 train/config/")
            continue

        try:
            shutil.copy2(src, dst)
            print(f"    ✓ 导出: {filename}")
            exported += 1
        except Exception as e:
            print(f"    错误: 复制 {filename} 失败: {e}")

    return exported


def main():
    """主函数"""

    print("="*60)
    print("模型导出")
    print("="*60)

    # 检查训练输出目录
    if not TRAIN_OUTPUT.exists():
        print(f"错误: 训练输出目录不存在: {TRAIN_OUTPUT}")
        print(f"    请先运行 3_train_expert.py 训练专家模型")
        return

    # 创建服务端模型目录
    SERVER_MODELS.mkdir(parents=True, exist_ok=True)

    # 清理配置（默认开启）
    cleanup_enabled = True
    print(f"\n训练文件清理: {'已启用' if cleanup_enabled else '已禁用'}")
    if cleanup_enabled:
        print("  将自动删除: last.pt, labels.jpg, args.yaml, 空weights目录")
        print("  将保留: results.csv, 各类曲线图（便于查看训练效果）")
        print("  将清理历史遗留: train/scripts/runs/")

    # 清理历史遗留的 runs 目录
    if cleanup_enabled:
        cleanup_legacy_runs()

    # 扫描所有专家模型
    expert_count = 0

    for model_dir in sorted(TRAIN_OUTPUT.iterdir()):
        if not model_dir.is_dir():
            continue

        # 跳过以 _ 开头的目录
        if model_dir.name.startswith('_'):
            continue

        if export_expert(model_dir, cleanup=cleanup_enabled):
            expert_count += 1

    # 导出VLM模型
    vlm_count = export_vlm_models()

    # 导出客户端配置
    config_count = export_client_config()

    # 汇总
    print(f"\n{'='*60}")
    print(f"导出汇总")
    print(f"{'='*60}")
    print(f"专家模型: {expert_count} 个")
    print(f"VLM模型: {vlm_count} 个")
    print(f"客户端配置: {config_count} 个")
    print(f"模型目录: {SERVER_MODELS}")
    print(f"配置目录: {CLIENT_CONFIG}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
