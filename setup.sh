#!/bin/bash

WHL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
WHL_NAME="flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
PACKAGE_DIR="./package"

mkdir -p $PACKAGE_DIR

if [ ! -f "$PACKAGE_DIR/$WHL_NAME" ]; then
    echo "正在从 GitHub 下载 Flash Attention..."
    # -L 是为了处理 GitHub 的重定向，-O 指定输出路径
    curl -L $WHL_URL -o "$PACKAGE_DIR/$WHL_NAME"
    
    if [ $? -eq 0 ]; then
        echo "下载完成。"
    else
        echo "下载失败，请检查网络。"
        exit 1
    fi
else
    echo "$WHL_NAME 已存在，跳过下载。"
fi

# 3. 执行 uv 同步
echo "正在同步环境..."
UV_EXTRA_INDEX_URL="" uv sync