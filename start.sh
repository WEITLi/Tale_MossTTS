#!/bin/bash
set -e

# 路径定义
VENV_PYTHON="./index-tts/.venv/bin/python3"
QWEN_LIBS="./qwen_libs"
INDEX_DIR="."

echo "=================================================="
echo "   Unitale AI后端服务 IndexTTS2 + Qwen3-TTS VoiceDesign"
echo "=================================================="

# 1. 物理清场与解锁
echo "[1/4] 正在清理..."
fuser -k 8300/tcp 2>/dev/null || true
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock
sleep 1

# 2. 系统源换镜像 & 安装编译环境
echo "[2/4] 正在补全系统库..."
sed -i 's@http://.*archive.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list
sed -i 's@http://.*security.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list
apt-get update -qq

# 这里的 sox 是系统软件，必须装
apt-get install -y --no-install-recommends \
    sox libsox-fmt-all psmisc libgomp1 \
    python3-dev build-essential \
    > /dev/null 2>&1

# 3. 依赖库精准对齐
echo "[3/4] 正在检查 Python 依赖..."
ALI_PYPI="https://mirrors.aliyun.com/pypi/simple/"

# (A) 主环境：删除 if 判断，强制执行 uv pip install
# uv 会自动处理已安装的包，速度很快，确保 "sox" 库一定被安装
uv pip install \
    "transformers==4.52.1" \
    "tokenizers==0.21.0" \
    "sox" "onnxruntime" "einops" "soundfile" \
    --python "$VENV_PYTHON" \
    --index-url "$ALI_PYPI"

# (B) 侧载环境 - Qwen3
if [ ! -d "$QWEN_LIBS" ]; then
    mkdir -p "$QWEN_LIBS"
    uv pip install "transformers==4.57.3" "tokenizers==0.22.2" "accelerate==1.12.0" \
        --target "$QWEN_LIBS" --no-deps --python "$VENV_PYTHON" --index-url "$ALI_PYPI"
fi

# (C) 侧载环境 - MOSS-TTS
MOSS_LIBS="./moss_libs"
if [ ! -d "$MOSS_LIBS" ]; then
    echo "   -> 安装 MOSS-TTS 侧载依赖..."
    mkdir -p "$MOSS_LIBS"
    uv pip install "transformers==5.0.0" "tokenizers>=0.22" "accelerate" \
        --target "$MOSS_LIBS" --no-deps --python "$VENV_PYTHON" --index-url "$ALI_PYPI"
fi

# 4. 启动常驻服务
echo "[4/4] 启动服务..."
echo "--------------------------------------------------"

export PYTHONPATH="./Qwen3-TTS:."
# 针对 L40 显卡加速编译
export TORCH_CUDA_ARCH_LIST="8.9"

# 运行
exec "$VENV_PYTHON" api.py