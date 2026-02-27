#!/usr/bin/env python3
"""
视安 (ShiAn) 服务管理脚本
用法:
    python server/run.py start    # 安装依赖 + 启动服务（守护进程）
    python server/run.py stop     # 停止服务
    python server/run.py status   # 查看状态
    python server/run.py logs     # 查看日志
    python server/run.py restart  # 重启服务
"""
import subprocess
import sys
import os
from pathlib import Path

# 项目根目录（server/ 的上级）
PROJECT_DIR = Path(__file__).parent.parent.resolve()
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
SERVICE_NAME = "shian"
SERVICE_FILE = f"/etc/systemd/system/{SERVICE_NAME}.service"
LOG_FILE = "/var/log/shian.log"
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"

# 系统依赖（apt 包名）
SYSTEM_DEPS = ["libgl1", "libglib2.0-0t64", "libsm6", "libxext6", "libxrender1", "build-essential", "python3-dev"]


def run_cmd(cmd, check=True):
    """执行命令"""
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        if result.stderr.strip():
            print(f"  错误: {result.stderr.strip()}")
        return False
    return True


def check_root():
    if os.geteuid() != 0:
        print("错误: 请使用 root 权限运行")
        sys.exit(1)


def install_system_deps():
    """检查并安装系统依赖"""
    print("\n[1/3] 检查系统依赖...")
    missing = []
    for pkg in SYSTEM_DEPS:
        result = subprocess.run(
            f"dpkg -s {pkg} 2>/dev/null | grep 'Status: install ok'",
            shell=True, capture_output=True
        )
        if result.returncode != 0:
            missing.append(pkg)

    if missing:
        print(f"  安装: {', '.join(missing)}")
        run_cmd(f"apt-get update -qq && apt-get install -y -qq {' '.join(missing)}")
    else:
        print("  已就绪")


def setup_venv():
    """创建虚拟环境并安装 Python 依赖"""
    print("\n[2/3] 检查 Python 环境...")

    if not VENV_PYTHON.exists():
        print("  创建虚拟环境...")
        run_cmd(f"python3 -m venv {VENV_DIR}")

    # 检查关键包是否已安装
    result = subprocess.run(
        f"{VENV_PYTHON} -c \"import fastapi, cv2, ultralytics\" 2>/dev/null",
        shell=True
    )
    if result.returncode != 0:
        print("  安装 Python 依赖（使用阿里云镜像）...")
        req_file = PROJECT_DIR / "requirements.txt"
        run_cmd(f"{VENV_PYTHON} -m pip install -q -r {req_file} -i {PIP_MIRROR} --trusted-host mirrors.aliyun.com")
    else:
        print("  已就绪")


def install_service():
    """安装 systemd 服务"""
    print("\n[3/3] 配置守护服务...")

    service_content = f"""[Unit]
Description=ShiAn AI Safety Monitoring Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={PROJECT_DIR}
ExecStart={VENV_PYTHON} -m server.api.main
Restart=always
RestartSec=5
StandardOutput=append:{LOG_FILE}
StandardError=append:{LOG_FILE}
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""
    with open(SERVICE_FILE, "w") as f:
        f.write(service_content)

    run_cmd("systemctl daemon-reload")
    run_cmd(f"systemctl enable {SERVICE_NAME} -q")
    print("  服务已注册（开机自启 + 崩溃自动重启）")


def start():
    """一键启动"""
    check_root()

    print("=" * 50)
    print("视安 (ShiAn) 一键部署")
    print("=" * 50)

    install_system_deps()
    setup_venv()
    install_service()

    # 启动服务
    print("\n启动服务...")
    run_cmd(f"systemctl start {SERVICE_NAME}")

    # 等待启动
    import time
    time.sleep(2)

    # 检查状态
    result = subprocess.run(
        f"systemctl is-active {SERVICE_NAME}",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip() == "active":
        print("\n" + "=" * 50)
        print("启动成功!")
        print(f"  服务状态: python server/run.py status")
        print(f"  查看日志: python server/run.py logs")
        print(f"  停止服务: python server/run.py stop")
        print("=" * 50)
    else:
        print("\n启动失败，查看日志:")
        os.system(f"tail -20 {LOG_FILE}")


def stop():
    """停止服务"""
    check_root()
    print("停止 ShiAn 服务...")
    run_cmd(f"systemctl stop {SERVICE_NAME}")
    print("已停止")


def restart():
    """重启服务"""
    check_root()
    print("重启 ShiAn 服务...")
    run_cmd(f"systemctl restart {SERVICE_NAME}")
    print("已重启")


def status():
    """查看状态"""
    os.system(f"systemctl status {SERVICE_NAME}")


def logs():
    """查看日志（最近50行）"""
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print("暂无日志")
        return
    os.system(f"tail -50 {LOG_FILE}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    commands = {
        "start": start,
        "stop": stop,
        "restart": restart,
        "status": status,
        "logs": logs,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
