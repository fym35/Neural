import argparse
import numpy as np
import csv
import sys
import select
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="Input image to recognize")
parser.add_argument("--type", choices=["raw", "file"], default="raw",
                    help="Image input raw/file, default=raw")
parser.add_argument("--model", type=str, default="weights.npz",
                    help="Model weights file (default: weights.npz")

args = parser.parse_args()

if not os.path.isfile(args.model):
    print(args.model + " does not exist! Please supply a valid model file")
    sys.exit()

node_sizes = [784, 128, 64, 10]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

stdin_fd = sys.stdin.fileno()

# Load weights
data = np.load(args.model)
weights = [data[key] for key in data.files]

layers = [np.zeros(784), np.zeros(128), np.zeros(64), np.zeros(10)]

val_input = args.input
if args.type == "file":
    with open(val_input, "r", encoding="utf-8") as f:
        val_input = f.read()

values = val_input.strip().split(",")

layers[0] = [float(v) / 255.0 for v in values]

for l in range(len(weights)):
    layers[l + 1] = sigmoid(np.dot(layers[l], weights[l]))

max_val = -1
result = -1

for i in range(len(layers[3])):
    if layers[3][i] > max_val:
        max_val = layers[3][i]
        result = i

print("Result: ", result);

