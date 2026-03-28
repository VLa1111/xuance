import xuance
# 为 IQL 算法创建运行器
runner = xuance.get_runner(method='iql',
                           env='sc2',  # 选择：sc2, mpe, robotic_warehouse, football, magent2.
                           env_id='3m',  # 选择：3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2 等。
                           is_test=False)  # False 用于训练，True 用于测试
runner.run()  # 开始运行（或 runner.benchmark() 用于基准测试）