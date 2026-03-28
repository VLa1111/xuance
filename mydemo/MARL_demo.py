import xuance
runner = xuance.get_runner(algo=["maddpg", "iddpg"],
                           env='mpe',
                           env_id='simple_push_v3')
runner.run(mode='train')