import argparse
from xuance import get_runner

def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--algo", type=str, default="iddpg")
    parser.add_argument("--env", type=str, default="drones")
    parser.add_argument("--env-id", type=str, default="MultiHoverAviary")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=10)
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--test-episode", type=int, default=5)

    return parser.parse_args()

if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser)
    if parser.benchmark:
        runner.run(mode='benchmark')
    else:
        runner.run(mode='train')