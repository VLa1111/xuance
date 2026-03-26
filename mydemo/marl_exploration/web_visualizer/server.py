"""
Flask Web Server for Real-time MARL Visualization
=================================================
This server receives training data from the RL agent and broadcasts
it to connected web clients for real-time visualization.

Usage:
    python server.py
    Then open http://localhost:5000 in your browser
"""

import os
import json
import time
import random
import threading
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'marvl-visualization-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
state = {
    "agents": [],           # List of agent positions
    "landmarks": [],        # List of landmark positions
    "coverage_grid": None,  # 2D grid showing exploration coverage
    "grid_size": 20,        # Grid resolution
    "episode_reward": 0,
    "episode": 0,
    "step": 0,
    "coverage_percent": 0,
    "is_training": False,
    "fps": 0,
}


class CoverageGrid:
    """Tracks exploration coverage on a 2D grid."""

    def __init__(self, grid_size=20, world_size=2.0):
        self.grid_size = grid_size
        self.world_size = world_size
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.total_cells = grid_size * grid_size
        self.covered_cells = set()

    def update(self, positions):
        """Update coverage based on agent positions."""
        for x, y in positions:
            # Map world coordinates to grid
            gx = int((x + self.world_size) / (2 * self.world_size) * (self.grid_size - 1))
            gy = int((y + self.world_size) / (2 * self.world_size) * (self.grid_size - 1))
            gx = max(0, min(self.grid_size - 1, gx))
            gy = max(0, min(self.grid_size - 1, gy))

            # Mark cell and neighbors as covered (agent has some radius)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        cell = (nx, ny)
                        if cell not in self.covered_cells:
                            self.covered_cells.add(cell)
                            self.grid[ny][nx] = 1  # Mark as covered

    def get_coverage(self):
        """Get current coverage percentage."""
        return len(self.covered_cells) / self.total_cells * 100

    def reset(self):
        """Reset the grid."""
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.covered_cells = set()


coverage_grid = CoverageGrid(grid_size=state["grid_size"])


@app.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Return current simulation state."""
    return jsonify({
        "agents": state["agents"],
        "landmarks": state["landmarks"],
        "coverage_grid": coverage_grid.grid,
        "grid_size": state["grid_size"],
        "episode_reward": state["episode_reward"],
        "episode": state["episode"],
        "step": state["step"],
        "coverage_percent": coverage_grid.get_coverage(),
        "is_training": state["is_training"],
    })


@app.route('/api/update', methods=['POST'])
def update_state():
    """Receive state update from training process."""
    global state

    data = request.json

    # Update positions
    state["agents"] = data.get("agents", [])
    state["landmarks"] = data.get("landmarks", [])

    # Update coverage
    if state["agents"]:
        coverage_grid.update(state["agents"])

    # Update metrics
    state["episode_reward"] = data.get("episode_reward", 0)
    state["step"] = data.get("step", 0)
    state["is_training"] = data.get("is_training", True)

    # Broadcast to all clients
    socketio.emit('state_update', {
        "agents": state["agents"],
        "landmarks": state["landmarks"],
        "coverage_grid": coverage_grid.grid,
        "coverage_percent": coverage_grid.get_coverage(),
        "episode_reward": state["episode_reward"],
        "step": state["step"],
    })

    return jsonify({"status": "ok"})


@app.route('/api/episode_end', methods=['POST'])
def episode_end():
    """Handle episode end - reset coverage for next episode."""
    global state
    data = request.json

    state["episode"] = data.get("episode", state["episode"] + 1)
    coverage_grid.reset()

    socketio.emit('episode_end', {
        "episode": state["episode"],
        "final_coverage": data.get("coverage_percent", 0),
        "total_reward": data.get("total_reward", 0),
    })

    return jsonify({"status": "ok"})


@app.route('/api/training_status', methods=['GET', 'POST'])
def training_status():
    """Get or set training status."""
    global state
    if request.method == 'POST':
        data = request.json
        state["is_training"] = data.get("is_training", True)
        return jsonify({"status": "ok"})
    return jsonify({"is_training": state["is_training"]})


@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    print(f"Client connected")
    emit('state_update', {
        "agents": state["agents"],
        "landmarks": state["landmarks"],
        "coverage_grid": coverage_grid.grid,
        "coverage_percent": coverage_grid.get_coverage(),
        "episode_reward": state["episode_reward"],
        "step": state["step"],
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected")


if __name__ == '__main__':
    print("=" * 60)
    print("Multi-Agent RL Web Visualizer")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    # Create templates directory
    os.makedirs("templates", exist_ok=True)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)