#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import sys

import argparse
import numpy as np
import os
import vizdoom

import vizdoomgym_duel as vgd

parser = argparse.ArgumentParser(description='Randomly explore map and save observations')
parser.add_argument('--map', metavar='[0-5]', type=int, required=True)
parser.add_argument('--output', metavar='DIR', type=str, required=True)
parser.add_argument('--part', metavar='N', type=int, default=os.environ['SGE_TASK_ID'])
parser.add_argument('--episodes', metavar='N', type=int, required=True)
parser.add_argument('--path', metavar='DIR', type=str, required=True)
parser.add_argument('--hostile', action='store_true')

args = parser.parse_args()

sys.path.append(args.path)

from src.behaviours.exploration_behaviour import ExplorationBehaviour
from src.behaviours.fighting_behaviour import FightingBehaviour


def explore():
    map_name = None
    if 0 <= args.map <= 5:
        map_name = "map0{}".format(args.map)
        print('Map name: {}'.format(map_name))
    else:
        print('ERROR - Incorrect map number: {}'.format(args.map))
        exit(1)

    file_name = 'explore_{}_part_{:04d}'.format(map_name, args.part)

    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
            print('Output directory created: {}'.format(args.output))
        except FileExistsError:
            if not os.path.isdir(args.output):
                print('Output path exists and is not a directory')
                exit(1)

    output = os.path.join(args.output, file_name)

    print('Output file: {}'.format(output))

    if args.hostile:
        behaviour = FightingBehaviour()
        print('Bot added to exploration')
    else:
        behaviour = ExplorationBehaviour()
        print('No bot in exploration')

    env = vgd.ViZDoomGymDuel(behaviour, map_name=map_name, server=True, players=1)

    np.random.seed(args.part)
    done = False

    episodes_observations = []

    print("Exploration starting...")
    sys.stdout.flush()
    for ep in range(args.episodes):

        observed_frames = []
        print("Episode {}    starting...".format(ep + 1))
        for step in range(2100):
            # action = env.behaviour.sample_nonuniform()
            action = env.behaviour.action_space.sample()

            observation = None
            for _ in range(10):
                observation, reward, done, info = env.step(action)
                if done:
                    break

            if done:
                break

            observed_frames.append(observation)
            if step % 100 == 0:
                print("Step {}".format(step))
                sys.stdout.flush()

        print("Episode {} done".format(ep + 1))
        sys.stdout.flush()
        env.reset()

        episodes_observations.append(np.stack(observed_frames))

    print('Exploration done, saving')

    np.savez_compressed(output, *episodes_observations)

    print('Saved to {}'.format(output))


if __name__ == '__main__':
    explore_passed = False
    while not explore_passed:
        try:
            explore()
            explore_passed = True
        except vizdoom.ViZDoomErrorException:
            print('VizDoomError caught, retrying')
