"""
Launcher for Web Visualization
===============================
Starts both the web server and training client.

Usage:
    python launcher.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import subprocess
import time
import threading
import signal

# Change to web_visualizer directory
WEB_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))


def start_web_server():
    """Start the Flask web server in a subprocess."""
    print("[Launcher] Starting web server on http://localhost:5000")
    server_script = os.path.join(WEB_VIZ_DIR, "server.py")
    proc = subprocess.Popen(
        [sys.executable, server_script],
        cwd=WEB_VIZ_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return proc


def start_training():
    """Start the training script."""
    print("[Launcher] Starting training with visualization...")
    train_script = os.path.join(os.path.dirname(WEB_VIZ_DIR), "train_web.py")
    proc = subprocess.Popen(
        [sys.executable, train_script] + sys.argv[1:],
        cwd=os.path.dirname(WEB_VIZ_DIR)
    )
    return proc


def main():
    print("=" * 60)
    print("Multi-Agent RL Web Visualizer Launcher")
    print("=" * 60)
    print()
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser to view training")
    print()
    print("Press Ctrl+C to stop everything")
    print("=" * 60)

    # Start web server
    server_proc = start_web_server()

    # Wait for server to start
    time.sleep(3)

    # Check if server started successfully
    if server_proc.poll() is not None:
        print("[Launcher] Error: Web server failed to start")
        stdout, stderr = server_proc.communicate()
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return

    print("[Launcher] Web server started successfully")

    # Start training
    try:
        train_proc = start_training()

        # Wait for training to complete
        train_proc.wait()

    except KeyboardInterrupt:
        print("\n[Launcher] Interrupted, stopping processes...")

    finally:
        # Stop web server
        print("[Launcher] Stopping web server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)

        print("[Launcher] All processes stopped")
        print("\nYou can restart with: python launcher.py")


if __name__ == "__main__":
    main()