# 视安 (ShiAn)

基于 YOLOv12 + 规则引擎 + VLM（可选）的实时 AI 安全监控系统。

## 平台支持

- **Linux**（Debian/Ubuntu）：推荐，可使用 `run.py` 一键部署（systemd 守护进程）
- **macOS / Windows**：支持，使用 `python -m server.api.main` 启动

## 快速开始

```bash
# 克隆项目
git clone https://github.com/CJiangqiu/ShiAn.git
cd ShiAn
pip install -r requirements.txt

# 启动服务（Linux 一键部署）
sudo python server/run.py start

# 或跨平台前台启动
python -m server.api.main
```

启动后：
- WebSocket 推理接口: `ws://localhost:8000/ws/inference/{expert_id}`
- 健康检查: `http://localhost:8000/health`

## 项目结构

```
server/                          # 推理服务
├─ api/main.py                   # FastAPI + WebSocket 入口
├─ core/                         # 推理流水线
├─ config/                       # 运行时配置
├─ models/                       # 专家模型（训练后导出到此）
└─ storage/                      # 存储后端抽象（含示例实现）

train/                           # 训练工具
├─ scripts/                      # 训练脚本（按 1-4 顺序执行）
├─ config/experts/               # 专家 YAML 配置
└─ data/                         # 训练数据集
```

## 详细文档

规则 DSL 语法、配置说明、架构详解、API 接口等完整内容请参考 **《视安技术文档》**。

## 许可证

MIT License
