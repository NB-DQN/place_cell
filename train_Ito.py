import math
import random
import argparse

import numpy as np

def generate_sequence(args):
    visited = []
    dataset = []

    shift_x = int(math.floor((args.maze_size_x - 1) / 2))
    shift_y = int(math.floor((args.maze_size_y - 1) / 2))

    current = (0, 0)
    for i in range(0, args.sequence_length):
        while True:
            direction = random.randint(1, 4)
            if   direction == 1:
                current = (current[0] - 1, current[1]    )
            elif direction == 2:
                current = (current[0]    , current[1] + 1)
            elif direction == 3:
                current = (current[0] + 1, current[1]    )
            elif direction == 4:
                current = (current[0]    , current[1] - 1)

            # may be some bugs around here; entering infinity loop
            if current[0] in range(-shift_x, args.maze_size_x - shift_x - 1) and \
               current[1] in range(-shift_y, args.maze_size_y - shift_y - 1):
                break

        supervisor = np.zeros((args.maze_size_x * args.maze_size_y))

        if current in visited:
            supervisor[args.maze_size_y * (current[0] + shift_x) + current[1] + shift_y] = 0.3
        else:
            supervisor[args.maze_size_y * (current[0] + shift_x) + current[1] + shift_y] = 1
            visited.append(current)

        dataset.append((direction, supervisor))

    return dataset, visited

parser = argparse.ArgumentParser()
parser.add_argument('--sequence_length', type=int, default=100)
parser.add_argument('--maze_size_x',     type=int, default=9)
parser.add_argument('--maze_size_y',     type=int, default=9)
args = parser.parse_args()

sequence, visited = generate_sequence(args)

print(sequence)
print(visited)
