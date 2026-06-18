#!/bin/bash

# ==========================================
# 远程服务器及路径配置
# ==========================================
REMOTE_USER="andyhsueh"
REMOTE_IP="192.168.0.52"
REMOTE_PATH="/home/andyhsueh/Videos/"

# 依赖 SSH Key 免密登录（不再使用明文密码 / sshpass）。
# 请确保运行此脚本的环境已将对应的私钥加入 SSH agent 或 ~/.ssh，
# 且该公钥已加入 ${REMOTE_USER}@${REMOTE_IP} 的 ~/.ssh/authorized_keys。

# ==========================================
# 动态获取脚本参数（Argument）
# ==========================================
# $1 表示运行脚本时输入的第一个参数
TARGET_FILE="$1"

# 1. 检查用户是否输入了文件名参数
if [ -z "$TARGET_FILE" ]; then
    echo -e "\033[31m[错误] 请提供要传输的文件名！\033[0m"
    echo -e "使用方法示例: \033[36m$0 文件名.mp4\033[0m"
    exit 1
fi

# 2. 检查本地目标视频文件是否存在
if [ ! -f "$TARGET_FILE" ]; then
    echo -e "\033[31m[错误] 当前目录下未找到文件: $TARGET_FILE\033[0m"
    exit 1
fi

echo "------------------------------------------------"
echo "待传文件: $TARGET_FILE"
echo "目标服务器: $REMOTE_IP"
echo "目标路径: $REMOTE_PATH"
echo "------------------------------------------------"

# 3. 执行传输（依赖 SSH Key 免密登录）
rsync -avP -e "ssh -o StrictHostKeyChecking=no" --rsync-path="mkdir -p $REMOTE_PATH && rsync" "$TARGET_FILE" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"

# 4. 检查传输返回值
if [ $? -eq 0 ]; then
    echo -e "\033[32m\n[成功] 文件 $TARGET_FILE 已成功传输！\033[0m"
else
    echo -e "\033[31m\n[失败] 传输未完成，请检查网络或权限。\033[0m"
    exit 2
fi
