# 视安 (ShiAn)

基于 YOLOv12 + 规则引擎 + 视觉语言模型（VLM）的实时安全监控系统。

## 功能

- **目标检测**：YOLOv12 专家模型，按场景独立训练
- **目标跟踪**：ByteTrack 多目标持续跟踪
- **规则引擎**：自定义 DSL 描述告警触发条件
- **VLM 验证**：规则触发后，由视觉语言模型进行二次确认，降低误报
- **实时推送**：WebSocket 接口实时推送告警事件
- **存储接口**：抽象存储后端，按需对接数据库或第三方服务

## 环境要求

- Python 3.12+
- PyTorch 2.7+ with CUDA
- 8GB+ 显存（如需启用本地 VLM 验证）

## 快速开始

### 安装

```bash
git clone https://github.com/CJiangqiu/ShiAn.git
cd ShiAn
pip install -r requirements.txt
```

### 训练专家模型

训练前需要：
1. 将 YOLO 格式标注的数据集放入 `train/data/original/{expert_id}/`
2. 在 `train/config/experts/` 下创建专家配置（参考 `expert.template.yaml`）

训练脚本位于 `train/scripts/`，按顺序执行：

```bash
python train/scripts/1_download_models.py    # 下载基础模型（仅首次）
python train/scripts/2_preprocess_dataset.py # 数据集预处理
python train/scripts/3_train_expert.py       # 训练专家模型
python train/scripts/4_build_model.py        # 导出到 server/models/
```

### 启动服务

```bash
python -m server.api.main
```

Linux 一键部署（需要 root）：

```bash
python server/run.py start
```

启动后：
- WebSocket 推理接口: `ws://localhost:8000/ws/inference/{expert_id}`
- 健康检查: `http://localhost:8000/health`

### API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/experts` | 列出所有可用专家 |
| GET | `/experts/{expert_id}` | 获取专家详情 |
| POST | `/zones/{source_id}` | 保存区域配置 |
| GET | `/zones/{source_id}` | 获取区域配置 |

## 项目结构

```
server/                          # 推理服务
├─ api/main.py                   # FastAPI + WebSocket 入口
├─ core/                         # 七层推理流水线
├─ config/                       # 运行时配置
├─ models/                       # 专家模型（训练后导出到此）
└─ storage/                      # 存储后端抽象（含示例实现）

train/                           # 训练工具
├─ scripts/                      # 4 步训练脚本
├─ config/experts/               # 专家 YAML 配置
└─ data/                         # 训练数据集
```

## 详细文档

完整的使用指南、规则 DSL 语法、配置说明、架构详解等内容请参考 **《视安技术文档.docx》**。

## 许可证

MIT License
