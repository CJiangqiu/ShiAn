"""
视安 (ShiAn) API 服务
WebSocket 推理接口 + 区域配置接口
"""
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from server.core.pipeline import Pipeline
from server.core.types import AlertEvent
from server.core.zone_manager import ZoneManager


# 路径配置
SERVER_DIR = Path(__file__).parent.parent
CONFIG_DIR = SERVER_DIR / "config"
MODELS_DIR = SERVER_DIR / "models"
ZONES_DIR = CONFIG_DIR / "zones"


def load_api_config() -> dict:
    """加载 API 配置"""
    config_path = CONFIG_DIR / "api_config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def load_expert_info(expert_id: str) -> Optional[dict]:
    """加载专家信息"""
    info_path = MODELS_DIR / expert_id / "expert_info.json"
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_expert_zone_requirements(expert_id: str) -> List[str]:
    """获取专家的区域需求"""
    info = load_expert_info(expert_id)
    if not info:
        return []
    zone_cfg = info.get("zone", {}) or {}
    if not zone_cfg.get("required", False):
        return []
    zone_name = zone_cfg.get("zone_name") or f"{expert_id}_zone"
    return [zone_name]


# 全局单例 Pipeline（所有专家共享，避免重复加载VLM）
_pipeline: Optional[Pipeline] = None
_zone_manager = ZoneManager(ZONES_DIR)


def get_pipeline(expert_id: str, source_id: str) -> Pipeline:
    """获取全局 Pipeline 实例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(
            models_dir=MODELS_DIR,
            config_dir=CONFIG_DIR
        )

    # 设置当前 source 的专家
    _pipeline.set_source_expert(source_id, expert_id)
    return _pipeline


def check_zone_requirements(expert_id: str, source_id: str) -> Dict:
    """检查区域配置需求"""
    required = get_expert_zone_requirements(expert_id)
    if not required:
        return {"required": False, "missing": []}

    configured = _zone_manager.load_zones(source_id)
    missing = [z for z in required if z not in configured]

    return {
        "required": True,
        "zones": required,
        "missing": missing,
        "configured": list(configured.keys())
    }


def decode_frame(base64_data: str) -> Optional[np.ndarray]:
    """解码 base64 图像为 numpy 数组"""
    try:
        img_bytes = base64.b64decode(base64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def encode_frame(frame: np.ndarray, quality: int = 75) -> str:
    """编码 numpy 数组为 base64（降低质量提高速度）"""
    # 缩小图片以减少传输大小
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def draw_annotations(
    frame: np.ndarray,
    tracked_objects: list,
    alert_events: list[AlertEvent]
) -> np.ndarray:
    """绘制检测框和告警标注"""
    annotated = frame.copy()

    # 获取告警对象ID
    alert_obj_ids = set()
    for event in alert_events:
        alert_obj_ids.update(event.alert.target_objects)

    # 只绘制车辆类别（避免误画）
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}

    for obj in tracked_objects:
        # 过滤：只绘制车辆
        if obj.cls_name not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, obj.bbox)

        if obj.obj_id in alert_obj_ids:
            color = (0, 0, 255)  # 红色 - 告警
            thickness = 3
        else:
            color = (0, 255, 0)  # 绿色 - 正常
            thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # 标签（包含对象ID）
        label = f"ID:{obj.obj_id} {obj.cls_name} {obj.conf:.2f}"
        # 背景框
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def format_alerts(alert_events: list[AlertEvent]) -> list[dict]:
    """格式化告警信息"""
    alerts = []
    for event in alert_events:
        alert = event.alert
        alerts.append({
            "id": alert.alert_id,
            "type": alert.risk_type,
            "description": alert.risk_desc,
            "severity": alert.severity,
            "confidence": alert.confidence,
            "timestamp": event.timestamp,
            "bbox": list(alert.bbox) if alert.bbox else None,
            "target_objects": alert.target_objects
        })
    return alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    print("=" * 50)
    print("视安 (ShiAn) API 服务启动")
    print(f"模型目录: {MODELS_DIR}")
    print(f"配置目录: {CONFIG_DIR}")
    print("=" * 50)
    yield
    print("视安 (ShiAn) API 服务关闭")
    global _pipeline
    _pipeline = None


# 创建 FastAPI 应用
app = FastAPI(
    title="视安 (ShiAn) API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== REST API ==============

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "pipeline_active": _pipeline is not None
    }


@app.get("/experts")
async def list_experts():
    """列出所有可用专家"""
    experts = []
    for path in MODELS_DIR.iterdir():
        if path.is_dir() and not path.name.startswith("_"):
            info_path = path / "expert_info.json"
            if info_path.exists():
                info = load_expert_info(path.name)
                if info:
                    experts.append({
                        "id": path.name,
                        "description": info.get("description", ""),
                        "scene_ids": info.get("scene_ids", []),
                        "zone_required": bool(info.get("zone", {}).get("required", False))
                    })
    return {"experts": experts}


@app.get("/experts/{expert_id}")
async def get_expert(expert_id: str):
    """获取专家详情（包括区域需求）"""
    info = load_expert_info(expert_id)
    if not info:
        raise HTTPException(status_code=404, detail="专家不存在")

    zone_requirements = get_expert_zone_requirements(expert_id)

    return {
        "id": expert_id,
        "description": info.get("description", ""),
        "scene_ids": info.get("scene_ids", []),
        "expected_classes": info.get("expected_classes", []),
        "zone_required": len(zone_requirements) > 0,
        "zone_names": zone_requirements
    }


class ZoneData(BaseModel):
    """区域数据"""
    zones: Dict[str, List[List[int]]]  # {"safe_zone": [[x1,y1], [x2,y2], ...]}


@app.post("/zones/{source_id}")
async def save_zones(source_id: str, data: ZoneData):
    """保存区域配置"""
    _zone_manager.save_zones(source_id, data.zones)
    return {"status": "ok", "source_id": source_id, "zones": list(data.zones.keys())}


@app.get("/zones/{source_id}")
async def get_zones(source_id: str):
    """获取区域配置"""
    zones = _zone_manager.load_zones(source_id)
    return {"source_id": source_id, "zones": zones}


# ============== WebSocket API ==============

@app.websocket("/ws/inference/{expert_id}")
async def websocket_inference(websocket: WebSocket, expert_id: str, source_id: str = "default"):
    """
    WebSocket 推理接口

    连接参数:
        expert_id: 专家模型ID（路径参数）
        source_id: 视频源ID（查询参数，用于区域配置绑定）

    客户端发送:
    {
        "type": "frame",
        "data": "<base64 JPEG>"
    }

    服务端返回:
    - 正常: {"type": "result", "frame": "...", "alerts": [...], "objects": [...]}
    - 需要区域: {"type": "zone_required", "zones": ["safe_zone"], "missing": ["safe_zone"]}
    - 错误: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    print(f"[WS] 客户端连接，expert_id={expert_id}, source_id={source_id}")

    # 检查专家是否存在
    info = load_expert_info(expert_id)
    if not info:
        await websocket.send_json({
            "type": "error",
            "message": f"专家不存在: {expert_id}"
        })
        await websocket.close()
        return

    # 检查区域配置
    zone_check = check_zone_requirements(expert_id, source_id)
    if zone_check.get("missing"):
        await websocket.send_json({
            "type": "zone_required",
            "zones": zone_check["zones"],
            "missing": zone_check["missing"],
            "message": f"需要配置区域: {', '.join(zone_check['missing'])}"
        })
        # 不关闭连接，等待客户端配置完成后发送 zone_configured 消息
        # 或者客户端可以关闭连接去配置，然后重新连接

    # 获取 Pipeline
    try:
        pipeline = get_pipeline(expert_id, source_id)
        # 重置该视频源的状态（不影响其他源）
        pipeline.reset_source(source_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"加载专家模型失败: {str(e)}"
        })
        await websocket.close()
        return

    zone_configured = not zone_check.get("missing")

    # 如果不需要区域配置，发送 ready 消息让前端开始
    if zone_configured:
        await websocket.send_json({
            "type": "ready",
            "message": "可以开始分析"
        })

    # 最新帧缓冲区（生产者-消费者模式）
    latest_frame_holder = {"frame": None}
    stop_event = asyncio.Event()

    async def receiver():
        """接收帧，只保留最新一帧"""
        nonlocal zone_configured, zone_check
        try:
            while not stop_event.is_set():
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "zone_configured":
                    _zone_manager.clear_cache(source_id)
                    pipeline.zone_manager.clear_cache(source_id)
                    zone_check = check_zone_requirements(expert_id, source_id)
                    if zone_check.get("missing"):
                        await websocket.send_json({
                            "type": "zone_required",
                            "zones": zone_check["zones"],
                            "missing": zone_check["missing"],
                            "message": f"仍缺少区域: {', '.join(zone_check['missing'])}"
                        })
                    else:
                        zone_configured = True
                        pipeline.reset_source(source_id)
                        await websocket.send_json({
                            "type": "zone_ok",
                            "message": "区域配置完成，可以开始分析"
                        })

                elif msg_type == "frame" and zone_configured:
                    frame = decode_frame(data.get("data", ""))
                    if frame is not None:
                        latest_frame_holder["frame"] = frame

        except WebSocketDisconnect:
            stop_event.set()
        except Exception:
            stop_event.set()

    async def processor():
        """持续处理最新帧"""
        frame_count = 0
        try:
            while not stop_event.is_set():
                frame = latest_frame_holder["frame"]
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue

                # 取出并清空，避免重复处理
                latest_frame_holder["frame"] = None

                t_start = time.time()

                # 推理（在线程中执行，不阻塞事件循环）
                t1 = time.time()
                if frame_count == 0:
                    print(f"[帧尺寸] {frame.shape} ({frame.shape[0]*frame.shape[1]/1000000:.1f}MP)")
                alert_events = await asyncio.to_thread(pipeline.process_frame, frame, source_id)
                tracked_objects = pipeline.get_last_tracked_objects(source_id)
                t2 = time.time()

                # 绘制标注
                annotated_frame = draw_annotations(frame, tracked_objects, alert_events)
                t3 = time.time()

                # 编码图片
                encoded_frame = encode_frame(annotated_frame)
                t4 = time.time()

                # 构建响应
                response = {
                    "type": "result",
                    "frame": encoded_frame,
                    "alerts": format_alerts(alert_events),
                    "objects": [
                        {
                            "id": obj.obj_id,
                            "class": obj.cls_name,
                            "confidence": float(obj.conf),
                            "bbox": list(obj.bbox)
                        }
                        for obj in tracked_objects
                    ],
                    "timestamp": time.time()
                }
                t5 = time.time()

                # 发送结果
                if websocket.client_state.value == 1:  # CONNECTED
                    await websocket.send_json(response)
                    t6 = time.time()

                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[性能] 推理:{(t2-t1)*1000:.1f}ms | 绘制:{(t3-t2)*1000:.1f}ms | "
                              f"编码:{(t4-t3)*1000:.1f}ms | 构建:{(t5-t4)*1000:.1f}ms | "
                              f"发送:{(t6-t5)*1000:.1f}ms | 总计:{(t6-t_start)*1000:.1f}ms")

        except RuntimeError as e:
            # WebSocket 连接已关闭，正常退出
            if "websocket" in str(e).lower():
                print(f"[processor] WebSocket 连接已关闭")
            else:
                import traceback
                print(f"[processor] RuntimeError: {e}")
                traceback.print_exc()
            stop_event.set()
        except Exception as e:
            import traceback
            print(f"[processor] 异常: {e}")
            traceback.print_exc()
            stop_event.set()

    try:
        await asyncio.gather(receiver(), processor())
    except Exception:
        pass

    # 清理
    if not stop_event.is_set():
        print(f"[WS] 客户端断开连接，expert_id={expert_id}")
        pending = pipeline.get_pending_vlm_count()
        if pending > 0:
            print(f"[WS] 仍有 {pending} 个VLM任务在后台运行，结果将通过存储后端记录")


@app.get("/vlm/status")
async def vlm_status():
    """获取VLM处理状态"""
    if _pipeline is None:
        return {"pending": 0, "message": "Pipeline未初始化"}

    pending = _pipeline.get_pending_vlm_count()
    return {
        "pending": pending,
        "message": f"有 {pending} 个VLM任务在后台运行" if pending > 0 else "无待处理任务"
    }


@app.post("/vlm/drain")
async def drain_vlm_results(timeout: float = 120.0):
    """
    等待所有VLM任务完成

    适用于视频流结束后，等待后台VLM推理完成并获取所有告警
    """
    if _pipeline is None:
        return {"events": [], "message": "Pipeline未初始化"}

    pending = _pipeline.get_pending_vlm_count()
    if pending == 0:
        return {"events": [], "message": "无待处理任务"}

    print(f"[API] 开始drain VLM结果，待处理: {pending}")

    # 在线程池中执行阻塞操作
    loop = asyncio.get_event_loop()
    events = await loop.run_in_executor(
        None,
        lambda: _pipeline.drain_vlm_results(timeout=timeout)
    )

    # 格式化返回
    return {
        "events": format_alerts(events),
        "count": len(events),
        "message": f"处理完成，产生 {len(events)} 个告警"
    }


def run():
    """启动服务"""
    config = load_api_config()
    server_config = config.get("server", {})

    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    print(f"\n启动服务: http://{host}:{port}")
    print(f"WebSocket: ws://{host}:{port}/ws/inference/{{expert_id}}?source_id={{source_id}}\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
