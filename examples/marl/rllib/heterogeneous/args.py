import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--env-name", default="buildings", type=str)
parser.add_argument("--system-load-rescale-factor", default=0.6, type=float)
parser.add_argument("--max-episode-steps", default=250, type=int)
parser.add_argument("--local-dir", default="~/ray_result", type=str)
parser.add_argument("--stop-timesteps", default=int(1e10), type=int)
parser.add_argument("--stop-iters", default=int(1e10), type=int)
parser.add_argument("--stop-reward", default=1e10, type=float)
parser.add_argument("--run", default="PPO", type=str)
parser.add_argument("--framework", default="torch", type=str, choices=["torch", "tf2"])
parser.add_argument("--num-gpus", default=0, type=int)
parser.add_argument("--num-cpus", default=1, type=int)
parser.add_argument("--num-samples", default=1, type=int)
parser.add_argument("--log-level", default="WARN", type=str)
parser.add_argument("--node-ip-address", default="127.0.0.1", type=str)

