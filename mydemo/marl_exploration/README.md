# Multi-Agent RL for Robot Swarm Exploration

基于 XuanCe 框架的多智能体强化学习算法实现，用于机器人集群探索任务。

## 项目结构

```
marl_exploration/
├── configs/
│   ├── maddpg_exploration.yaml   # MPE 环境配置
│   ├── maddpg_drones.yaml        # PyBullet Drones 配置
│   └── grid_exploration.yaml     # 网格探索环境配置
├── environments/
│   ├── __init__.py              # 环境注册器
│   └── grid_exploration.py      # 网格探索环境实现
├── train.py                      # 完整训练脚本
├── train_web.py                  # 带 Web 可视化的训练脚本
├── train_grid.py                 # 网格探索训练脚本
├── visualize.py                  # 训练后可视化工具
├── quickstart.py                 # 快速启动示例
├── web_visualizer/               # Web 可视化服务器
│   ├── server.py                 # Flask Web 服务器
│   ├── launcher.py               # 一键启动器
│   └── templates/
│       └── index.html            # 前端页面
├── requirements.txt              # Python 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd mydemo/marl_exploration

# 安装 xuance 基础依赖
pip install xuance pettingzoo

# 安装 Web 可视化依赖
pip install -r requirements.txt
```

### 2. 启动训练和可视化

**方式 A: 一键启动（推荐）**
```bash
cd mydemo/marl_exploration/web_visualizer
python launcher.py
```

**方式 B: 分步启动**
```bash
# 终端 1: 启动 Web 服务器
cd mydemo/marl_exploration/web_visualizer
python server.py

# 终端 2: 启动训练
cd mydemo/marl_exploration
python train_web.py
```

### 3. 查看可视化

在浏览器中打开: **http://localhost:5000**

## 功能特性

### Web 可视化界面
- 实时显示 agents（蓝色圆点）在地图上的位置
- 实时显示 landmarks（红色圆点）的位置
- 显示探索覆盖率热力图（绿色区域表示已探索）
- 显示训练指标：Episode Reward、Coverage %、当前 Episode/Step
- 支持暂停/继续训练显示
- 记录历史 Episode 表现

### 远程访问
- Web 服务器监听 `0.0.0.0:5000`，可从远程访问
- 使用 Socket.IO 实现实时双向通信
- 浏览器打开即用，无需安装额外软件

## 可用算法

| 算法 | 类型 | 适用场景 |
|------|------|----------|
| MADDPG | 连续动作 | 机器人控制、无人机编队 |
| MATD3 | 连续动作 | 协作任务 |
| MAPPO | 离散/连续 | 通用多智能体任务 |
| IPPO | 离散/连续 | 竞争/协作混合 |
| QMIX | 离散动作 | 团队协作任务 |
| IDDPG | 连续动作 | 独立学习 |

## 可用环境

| 环境 | 说明 | 安装 |
|------|------|------|
| `simple_spread_v3` | N 个智能体覆盖 N 个地标 | pettingzoo |
| `simple_push_v3` | 协作推动物体 | pettingzoo |
| `simple_adversary_v3` | 对抗任务 | pettingzoo |
| `MultiHoverAviary` | 无人机悬停控制 | gym-pybullet-drones |

## 配置说明

主要配置项在 `configs/maddpg_exploration.yaml`:

```yaml
# 训练设备 (本地用 cpu，服务器用 cuda:0)
device: "cpu"

# 并行环境数
parallels: 4

# 训练步数
running_steps: 500000

# 网络结构
actor_hidden_size: [256, 256]
critic_hidden_size: [256, 256]

# 学习率
learning_rate_actor: 0.001
learning_rate_critic: 0.001
```

## 奖励函数设计

### 网格探索环境 (`grid_exploration`)

自定义的网格探索环境，专门为"完整覆盖迷宫"任务设计：

```python
def compute_reward(self):
    """
    奖励函数设计:
    1. 覆盖新格子: +10      # 鼓励探索新区域
    2. 重复访问格子: -0.1    # 惩罚无效移动
    3. 完整覆盖奖励: +100    # 达到100%覆盖率时bonus
    4. 速度奖励: coverage/step  # 越快覆盖越好
    """
```

**奖励组成**:
| 奖励项 | 值 | 目的 |
|--------|-----|------|
| 探索新格子 | +10 | 鼓励发现新区域 |
| 重复访问 | -0.1 | 惩罚无效探索 |
| 覆盖率进度 | coverage × 0.1 | 整体进度奖励 |
| 速度奖励 | coverage/step × 10 | 鼓励快速探索 |
| 完成奖励 | +100 | 100%覆盖时一次性奖励 |

### 观测空间

每个智能体的观测是一个 4 维向量:
- `normalized_x`: 智能体 X 坐标 (0-1)
- `normalized_y`: 智能体 Y 坐标 (0-1)
- `coverage`: 当前覆盖率 (0-100) 归一化
- `nearby_agents`: 附近其他智能体数量

### 动作空间

离散动作 (5 个选择):
- 0: 上
- 1: 下
- 2: 左
- 3: 右
- 4: 停留

## 常用命令

```bash
# 快速测试
python quickstart.py

# 完整训练（无可视化）
python train.py --benchmark

# 训练 + Web 可视化
python train_web.py --device cpu --parallels 4

# 自定义参数
python train_web.py \
    --env-id simple_spread_v3 \
    --device cpu \
    --parallels 4 \
    --running-steps 100000

# ============== 网格探索环境 ==============
# 训练网格探索任务
python train_grid.py --device cpu --parallels 8

# 评估训练好的模型
python train_grid.py --benchmark=False --test
```

## 部署到云服务器

1. 上传代码到云服务器
2. 安装依赖:
   ```bash
   pip install xuance pettingzoo flask flask-socketio python-socketio requests
   ```
3. 启动训练:
   ```bash
   python train_web.py --device cuda:0 --parallels 16
   ```
4. 在本地浏览器打开: `http://<服务器IP>:5000`

## 自定义环境

如果你想使用自己的环境，需要:

1. 在 `xuance/environment/multi_agent_env/` 中创建环境类
2. 注册到 `REGISTRY` 字典
3. 在配置文件中指定 `env_name` 和 `env_id`

## 常见问题

Q: `pettingzoo` 导入错误?
A: `pip install pettingzoo`

Q: Web 页面无法连接?
A: 确保服务器已启动，检查防火墙设置

Q: 训练太慢?
A: 减少 `parallels` 数量，或使用 GPU 加速 (`device: cuda:0`)
