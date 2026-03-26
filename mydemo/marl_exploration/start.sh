#!/bin/bash
# One-line launcher for Multi-Agent RL Grid Exploration

# Activate conda environment
source ~/miniconda3/bin/activate xuance_env

# Default: run grid exploration with minimal settings for quick test
# Usage: bash start.sh [mode] [device]
#   mode: test|bench|web (default: test)
#   device: cpu|cuda (default: cpu)

MODE=${1:-test}
DEVICE=${2:-cpu}

cd "$(dirname "$0")"

echo "=========================================="
echo "Multi-Agent RL Grid Exploration"
echo "=========================================="

case $MODE in
    test)
        echo "Mode: Quick Test (10000 steps)"
        echo "Device: $DEVICE"
        echo "-------------------------------------------"
        python train_grid.py --device $DEVICE --parallels 4 --running-steps 10000 --eval-interval 2000 --benchmark
        ;;

    bench)
        echo "Mode: Full Benchmark"
        echo "Device: $DEVICE"
        echo "-------------------------------------------"
        python train_grid.py --device $DEVICE --parallels 8 --running-steps 500000 --eval-interval 25000 --benchmark
        ;;

    web)
        echo "Mode: Web Visualization"
        echo "Device: $DEVICE"
        echo "-------------------------------------------"
        echo "Starting web server + training..."
        echo "Open http://localhost:5000 in browser"
        echo "-------------------------------------------"
        cd web_visualizer && python server.py &
        SERVER_PID=$!
        sleep 2
        python train_web.py --device $DEVICE --parallels 4 --server-url http://localhost:5000
        kill $SERVER_PID 2>/dev/null
        ;;

    mpe)
        echo "Mode: MPE Environment (simple_spread)"
        echo "Device: $DEVICE"
        echo "-------------------------------------------"
        python train.py --env-id simple_spread_v3 --device $DEVICE --parallels 4 --benchmark
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash start.sh [test|bench|web|mpe] [cpu|cuda]"
        ;;
esac