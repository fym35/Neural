import argparse
import numpy as np
import csv
import sys
import select

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=15,
                    help="Number of training iterations (default: 15)")
parser.add_argument("--input", type=str, default="input.csv",
                    help="Location of training data (default: input.csv)")
parser.add_argument("--output", type=str, default="weights.npz",
                    help="Location of model weight data output (default: weights.npz)")
parser.add_argument("--debug", action="store_true",
                    help="Show training as it happens")
args = parser.parse_args()

node_sizes = [784, 128, 64, 10]
weights = [
    np.random.normal(0, 1/(784**0.5), (784, 128)),
    np.random.normal(0, 1/(128**0.5), (128, 64)),
    np.random.normal(0, 1/(64**0.5), (64, 10))
]
learningRate = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_weights():
    np.savez(args.output, *weights)
    print("\nWeights saved to " + args.output)

stdin_fd = sys.stdin.fileno()

for i in range(args.iterations):
    with open(args.input, "r") as f:
        for line in f:
            layers = [np.zeros(784), np.zeros(128), np.zeros(64), np.zeros(10)]
            label = 0

            values = line.split(",")

            label = int(values[0])
            layers[0] = [float(v) / 255.0 for v in values[1:]]

            #TODO: 0 protection
            for l in range(len(weights)):
                layers[l + 1] = sigmoid(np.dot(layers[l], weights[l]))

            max_val = -1
            result = -1

            for i in range(len(layers[3])):
                if layers[3][i] > max_val:
                    max_val = layers[3][i]
                    result = i

            if args.debug:
                print("Correct: ", label, " |  Network Result: ", result)

            target = np.zeros(10)
            target[label] = 1
            error_output = target - layers[3]
            errors = [0] * 3
            errors[2] = error_output
            errors[1] = np.dot(errors[2], weights[2].T) * layers[2] * (1 - layers[2])
            errors[0] = np.dot(errors[1], weights[1].T) * layers[1] * (1 - layers[1])

            deltas = [
               learningRate * np.outer(layers[0], errors[0]),
               learningRate * np.outer(layers[1], errors[1]),
               learningRate * np.outer(layers[2], errors[2])
            ]

            for i in range(len(weights)):
               weights[i] += deltas[i]

save_weights()

