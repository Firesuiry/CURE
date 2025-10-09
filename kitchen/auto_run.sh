#!/bin/bash

# 自动运行脚本 - 持续运行 run.sh
# 运行结束或异常退出后自动重新启动

SCRIPT_DIR="/root/autodl-tmp/develop/kitchen"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"

echo "开始自动运行模式..."
echo "目标脚本: $RUN_SCRIPT"
echo "按 Ctrl+C 停止自动运行"
echo "================================"

# 计数器
run_count=0

while true; do
    run_count=$((run_count + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 第 $run_count 次运行开始"
    
    # 切换到脚本目录并运行
    cd "$SCRIPT_DIR"
    bash "$RUN_SCRIPT"
    
    exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 第 $run_count 次运行结束，退出码: $exit_code"
    
    # 短暂延迟后重新启动
    echo "等待 3 秒后重新启动..."
    sleep 3
    echo "================================"
done