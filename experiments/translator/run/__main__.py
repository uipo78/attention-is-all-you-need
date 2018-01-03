import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--n_epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--h", type=int, default=6)
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--mask", type=bool, default=None)
parser.add_argument("--d_pw_ffn", type=int, default=2048)
parser.add_argument("--epsilon", type=float, default=1e-6)

parser.add_argument("--logdir", default=None)

options = parser.parse_args()

run(**vars(options))
