# Grid Exploration 训练问题汇总

## 训练日期
2026-03-26

## 问题汇总

### 问题1: Reward记录全为0 (Critical)

**现象描述**:
- TensorBoard记录的episode reward全部为0.0
- 250000条reward记录，无一非零值

**根本原因分析**:

1. **xuance框架的episode_score机制**:
   - xuance框架使用`info["episode_score"]`记录累积reward
   - 这个值在`wrapper.py`的`XuanCeMultiAgentEnvWrapper`中维护
   - `GridExplorationMAEnv.step()`返回的`infos`需要包含`episode_score`

2. **PettingZooWrapper的info传递问题**:
   - `PettingZooWrapper.reset()`返回`(obs)`而不是`(obs, infos)`
   - `PettingZooWrapper.step()`返回`(obs, rewards, terms, truncs, infos)`
   - 但`GridExplorationEnv`的`infos`不包含`episode_score`

3. **Reward计算与记录时机不匹配**:
   - `GridExplorationEnv.step()`每步返回`compute_reward()`（基于coverage的累积值）
   - 但xuance框架期望的`episode_score`是每个step的即时reward累积

**代码位置**:
- `environments/grid_exploration.py` - PettingZooWrapper类和GridExplorationEnv类

---

### 问题2: Action Space定义不一致 (High)

**现象描述**:
- `GridExplorationEnv.action_space = Discrete(5)` 定义为5个动作
- 但xuance框架可能期望`Discrete(5)`对应动作0-4

**根本原因**:
- Discrete(5) 意味着动作空间是 {0, 1, 2, 3, 4}，共5个动作
- 环境代码中正确使用了0-4
- 但需要确认xuance的IQL agent是否正确处理这个action space

---

### 问题3: Observation Space定义有问题 (Medium)

**现象描述**:
- `observation_space = Box(low=-1, high=1, shape=(num_agents * 4,))`
- 每个agent只产生4维特征，但observation_space是所有agent拼接的

**根本原因**:
- `XuanCeMultiAgentEnvWrapper`期望每个agent有自己的observation_space
- 正确的定义应该是每个agent的obs为4维，而不是全局的`num_agents * 4`维

**代码位置**:
- `environments/grid_exploration.py:55-58`

---

### 问题4: 训练超参数不够优化 (Medium)

**当前配置问题**:
```yaml
learning_rate: 0.001      # 偏高，建议0.0005
batch_size: 256            # 偏大，建议128
training_frequency: 25     # 偏频繁，建议15
start_greedy: 1.0
end_greedy: 0.05           # 衰减太快
decay_step_greedy: 1000000 # 衰减步数
```

---

### 问题5: 缺少episode_score初始化 (Critical)

**根本原因**:
- xuance框架期望`info`中包含`episode_score`
- `GridExplorationEnv._get_infos()`只返回`coverage`和`step`
- 没有初始化`episode_score`

---

### 问题6: 训练状态无法验证 (Critical)

**现象描述**:
- 无法通过tensorboard确认训练是否正常
- 没有任何loss或其他训练指标

**根本原因**:
- 可能缺少训练过程中的日志记录
- 需要添加更详细的训练状态输出

---

## 改进方案

### 方案1: 修复PettingZooWrapper和infos传递

**修改文件**: `environments/grid_exploration.py`

1. 修复`PettingZooWrapper.reset()`:
```python
def reset(self):
    obs, infos = self.env.reset()
    # infos已经包含episode_score等
    return obs, infos
```

2. 确保`_get_infos()`返回完整的info:
```python
def _get_infos(self) -> Dict:
    return {
        agent: {
            "coverage": self.get_coverage_percent(),
            "step": self.step_count,
            "episode_score": self.episode_rewards.copy(),  # 关键！
        }
        for agent in self.agents
    }
```

3. 在`step()`中正确更新`episode_rewards`:
```python
# 每个agent的即时reward累加到episode_rewards
for agent in self.agents:
    self.episode_rewards[agent] += rewards[agent]
```

---

### 方案2: 修复Observation Space定义

**修改文件**: `environments/grid_exploration.py`

```python
# 修改前
self.observation_space = Box(
    low=-1, high=1,
    shape=(num_agents * 4,),  # 错误：全局拼接
    dtype=np.float32
)

# 修改后 - 每个agent独立的observation space
self.observation_space = Box(
    low=-1, high=1,
    shape=(4,),  # 每个agent: [x, y, coverage, nearby_agents]
    dtype=np.float32
)
```

---

### 方案3: 优化训练超参数

**修改文件**: `configs/grid_exploration.yaml`

```yaml
# Training
learning_rate: 0.0005      # 从0.001降低
batch_size: 128             # 从256降低
gamma: 0.99                 # 保持
tau: 0.005                  # 从0.001提高

# Exploration
start_greedy: 1.0
end_greedy: 0.05
decay_step_greedy: 1500000  # 从1000000增加，延长探索
start_training: 5000         # 从1000增加，让replay buffer有足够数据
training_frequency: 15       # 从25降低
sync_frequency: 200         # 从100增加，提高稳定性

use_grad_clip: True         # 启用梯度裁剪
grad_clip_norm: 0.5
```

---

### 方案4: 添加训练诊断日志

**修改文件**: `train_grid.py`

添加每1000步打印一次训练状态:
```python
if agent.current_step % 1000 == 0:
    print(f"[Step {agent.current_step}] Loss: {current_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
```

---

### 方案5: 简化reward计算以便调试

**修改文件**: `environments/grid_exploration.py`

暂时使用更简单的reward设计来验证训练流程:
```python
def step(self, actions: Dict):
    # 每个step的即时reward
    step_rewards = {}
    for i, agent in enumerate(self.agents):
        action = actions.get(agent, 4)
        # ... 移动处理 ...

        # 即时reward
        if is_new_cell:
            step_rewards[agent] = 10.0
        else:
            step_rewards[agent] = -0.1

        # 立即更新episode_rewards
        self.episode_rewards[agent] += step_rewards[agent]

    # 更新coverage等
    self._mark_coverage()
    self.step_count += 1

    # infos中包含episode_score
    infos = self._get_infos()

    return observations, step_rewards, terminations, truncations, infos
```

---

## 验证方案

### 验证步骤

1. **单元测试环境**: 运行`python environments/grid_exploration.py`确认环境正常
2. **快速训练测试**: 用10000步快速测试，确认reward有非零值
3. **完整训练**: 200000步训练，检查tensorboard曲线

### 判断修复成功的标准

- TensorBoard中有非零的episode reward曲线
- Reward随着训练进行逐渐增加
- Loss稳定下降或波动
