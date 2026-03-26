#!/bin/bash
# Unified launcher for Multi-Agent RL Training
#
# Usage: bash start.sh [mode] [device] [options]
#   mode:     test|bench|web (default: test)
#   device:   cpu|cuda (default: cpu)
#   options:  --visualize (enable web visualization)
#
# Examples:
#   bash start.sh test cpu              # Quick test, no visualization
#   bash start.sh test cpu --visualize   # Quick test with web visualization
#   bash start.sh bench cpu              # Full benchmark
#   bash start.sh bench cpu --visualize  # Benchmark with web visualization

# Activate conda environment
source ~/miniconda3/bin/activate xuance_env

# Parse arguments
MODE=${1:-test}
DEVICE=${2:-cpu}
shift 2 || true
VISUALIZE_FLAG=""

# Check for --visualize flag
for arg in "$@"; do
    if [ "$arg" = "--visualize" ]; then
        VISUALIZE_FLAG="--visualize"
    fi
done

cd "$(dirname "$0")"

echo "=========================================="
echo "Multi-Agent RL Training"
echo "=========================================="

case $MODE in
    test)
        echo "Mode: Quick Test (10000 steps)"
        echo "Device: $DEVICE"
        if [ -n "$VISUALIZE_FLAG" ]; then
            echo "Visualization: Enabled"
        fi
        echo "-------------------------------------------"
        python train_grid.py --device $DEVICE --parallels 4 --running-steps 10000 --eval-interval 2000 --benchmark $VISUALIZE_FLAG
        ;;

    bench)
        echo "Mode: Full Benchmark"
        echo "Device: $DEVICE"
        if [ -n "$VISUALIZE_FLAG" ]; then
            echo "Visualization: Enabled"
        fi
        echo "-------------------------------------------"
        python train_grid.py --device $DEVICE --parallels 8 --running-steps 500000 --eval-interval 25000 --benchmark $VISUALIZE_FLAG
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
        cd ..
        python train_grid.py --device $DEVICE --parallels 4 --visualize --benchmark
        kill $SERVER_PID 2>/dev/null
        ;;

    mpe)
        echo "Mode: MPE Environment (simple_spread)"
        echo "Device: $DEVICE"
        if [ -n "$VISUALIZE_FLAG" ]; then
            echo "Visualization: Enabled"
        fi
        echo "-------------------------------------------"
        python train_grid.py --config configs/maddpg_exploration.yaml --device $DEVICE --parallels 4 --benchmark $VISUALIZE_FLAG
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash start.sh [test|bench|web|mpe] [cpu|cuda] [--visualize]"
        ;;
esac
