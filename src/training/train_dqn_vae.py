#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import math
import sys

import argparse
import os
import torch
import torch.optim as optim
from functools import partial
from torchvision.transforms import transforms

from src.agents.drqn_agent import DRQNAgent
from src.agents.replay_memory import ReplayMemory, ContextReplayMemory
from src.behaviours.exploration_behaviour import ExplorationBehaviour
from src.behaviours.running_behaviour import RunningBehaviour
from src.datasets.data_transformer import *
from src.models.dqn import DQN
from src.models.dqn import DQNLatent
from src.models.drqn import DRQN
# new naming
from src.models.vae import DoomVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

our_maps = ["map01", "map02", "map03", "map04", "map05"]
used_maps = ["map01", "map02", "map05"]
doom_input_w = 64
doom_input_h = 64


def train_dqn(envs, optimizer, agent, num_episodes,
              timesteps_per_simulation, actions_per_episode,
              output_path):
    agent.update_target_model()
    num_env = len(envs)
    done = False

    print('[INFO] (train_dqn) Number of environments', num_env)
    for i_episode in range(num_episodes):
        # env = envs[i_episode % len(envs)]
        # env_idx= i_episode % len(envs)
        env_idx = np.random.randint(num_env)
        # env_idx=0
        env = envs[env_idx]
        print('[INFO] Environment used for this episode= ', used_maps[env_idx])

        # Initialize the environment and state
        # get first state
        state = env.reset()  # Get init/current state.

        # preprocess the state with transformation and vae coding
        # if transform is not None:
        #    state= transform(state)
        state = agent.preprocess_state(state)
        next_state = None

        running_loss = 0
        running_reward = 0
        # every time step in the episodes
        for t in range(0, actions_per_episode):
            # Select a (random) action. 
            action = agent.select_action(state)

            # Convert action number to gym action.
            gym_action = action.item()

            summed_reward = 0
            # execute action in emulator and observe new state and reward.
            # done several times to compensate for high frame rates
            for _ in range(timesteps_per_simulation):
                next_state, reward, done, _ = env.step(gym_action)
                summed_reward += reward
                if done:
                    break
            if done:
                break

            running_reward += summed_reward

            # if i_episode > 0:
            # # if i_episode != 0 and (i_episode + 1) % 10 == 0:
            #     env.behaviour.render_exploration_map()
            #     env.render()

            # reward to tensor
            reward = torch.tensor([summed_reward], device=device)

            # preproces next state with transformations and vae coding
            # if transform is not None:
            #    next_state= transform(next_state)
            next_state = agent.preprocess_state(next_state)

            # store experience tuple in the replay memory
            agent.memory.push(state, action, next_state, reward)

            #  
            agent.update_epsilon()

            # compute loss in the agent
            loss = agent.optimize_model(optimizer)
            print("Episode [{}/{}], t=[{}/{}], loss={:.6f}, eps={:.4f}".format(i_episode,
                                                                               num_episodes, t, actions_per_episode,
                                                                               loss, agent.epsilon))

            if loss > 0.:
                running_loss += loss
            if math.isnan(loss):
                print('[ERROR] Loss is NAN')
                sys.exit()

            if math.isinf(loss):
                print('[ERROR] Loss is INF')
                sys.exit()

            agent.update_target_model(t)
            state = next_state

        agent.memory.push_episode()
        print('LOSS Episode {} Avg-Loss= {}'.format(i_episode,
                                                    running_loss / actions_per_episode))

        print('REWARD Episode {} Avg-Reward= {}'.format(i_episode,
                                                        running_reward / actions_per_episode))
        agent.save_model(os.path.join(output_path,
                                      'vqn_vae_net_{:08d}.nn'.format(i_episode)))

    print('Complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_vae', action='store_true', help='Use vae coding if not set use cnn dqn')
    parser.add_argument('--vae_model', type=str, help='Full path to vae model', default='')
    parser.add_argument('--output_path', type=str, help='Where to save models')
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--freq_target_update', type=int, default=30)
    parser.add_argument('--replay_memory_size', type=int, default=50)
    parser.add_argument('--decay_epsilon', type=float, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--initial_epsilon', type=float, default=0.1)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--is_exploration', action='store_true')
    parser.add_argument('--map_index', type=int, default=0)
    parser.add_argument('--recurrent_nn', action='store_true')
    parser.add_argument('--latent_dimension', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=4e-04)
    parser.add_argument('--finetuning', type=str, default='')

    args = parser.parse_args()

    import vizdoomgym_duel as vgd

    if args.is_exploration:
        print('[INFO] Training on exploration behaviour.')
        behaviour = ExplorationBehaviour()
    else:
        print('[INFO] Training on running behaviour.')
        behaviour = RunningBehaviour()

    envs = [vgd.ViZDoomGymDuel(behaviour, map_name="map01", server=True, players=1),
            vgd.ViZDoomGymDuel(behaviour, map_name="map02", server=True, players=1),
            # vgd.ViZDoomGymDuel(behaviour, map="map03", server=True, players=1),
            # vgd.ViZDoomGymDuel(behaviour, map="map04", server=True, players=1),
            vgd.ViZDoomGymDuel(behaviour, map_name="map05", server=True, players=1)]

    action_size = behaviour.action_size

    num_episodes = args.num_episodes
    timesteps_per_simulation = 5
    actions_per_episode = 2100  # 2100
    observe = actions_per_episode * len(envs)

    replay_memory_size = args.replay_memory_size

    print('[INFO] Configuration of training')
    print('[INFO] Dimension of action space', action_size)
    print('[INFO] Number of episodes', num_episodes)
    print('[INFO] Simulation time steps', timesteps_per_simulation)
    print('[INFO] Actions per episode', actions_per_episode)
    print('[INFO] Number of total observations', observe)
    print('[INFO] Update target network frequency', args.freq_target_update)
    print('[INFO] Replay memory length', replay_memory_size)
    print('[INFO] Map for this training is', our_maps[args.map_index])
    print('[INFO] The learning rate is', args.learning_rate)

    vae_file_name = args.vae_model
    output_path = args.output_path

    # make dir tree
    os.makedirs(args.output_path, exist_ok=True)

    preprocess_func = None
    trace_gradient = 1
    crop_size = 120

    if args.with_vae:
        # load vae learned with depth and labels
        z_size = args.latent_dimension
        vae = DoomVAE(2, crop_size, crop_size, z_size)
        # vae= torch.load(vae_file_name)['model']
        vae.load_state_dict(torch.load(vae_file_name))  # , map_location='cpu'))
        vae.eval()
        vae.to(device)

        # Transformations for input image for vae
        transform = transforms.Compose([NpCenterCrop((120, 120)),
                                        # NpResize3d((doom_input_w, doom_input_h),
                                        #           interpolation=cv2.INTER_CUBIC),
                                        torch.from_numpy,
                                        partial(torch.div, other=255.)])
        this_transform = transform

        # prerprocess with vae
        @torch.no_grad()
        def vae_preprocess(tensor, this_device):
            tensor = this_transform(tensor).to(this_device)
            tensor = tensor.float().unsqueeze(0)
            mu, log_var = vae.encode(tensor)
            return mu

        # create agent and policy model
        policy_model = DQNLatent(z_size, action_size)
        preprocess_func = vae_preprocess

    else:
        if args.recurrent_nn:
            # do training with DRQN
            print('[INFO] Training with Recurrent DQN!')
            policy_model = DRQN(120, 160, 2, action_size)
            trace_gradient = 6
            print('[INFO] Trace gradient is', trace_gradient)
        else:
            print('[INFO] Training with plain DQN!')
            policy_model = DQN(120, 160, 2, action_size)

    # for debugging
    this_batch_size = args.batch_size

    if args.recurrent_nn:
        trace_length = 20
        print('[INFO] Context replay memory with trace lenght of', trace_length)
        replay_memory = ContextReplayMemory(len(envs), trace_length)
    else:
        print('[INFO] Plain replay memory with size', replay_memory_size)
        replay_memory = ReplayMemory(replay_memory_size)

    if str(args.finetuning) != '':
        print('[INFO] Performing finetuning from', args.finetuning)
        policy_model.load_state_dict(torch.load(str(args.finetuning)))  # , map_location='cpu'))

    # create dqn agent with policy
    dqn_agent = DRQNAgent(action_size,
                          replay_memory,
                          batch_size=this_batch_size,
                          preprocess_function=preprocess_func,
                          network=policy_model,
                          initial_epsilon=args.initial_epsilon,
                          final_epsilon=args.final_epsilon,
                          epsilon_half_life=args.decay_epsilon,
                          update_target_freq=args.freq_target_update,
                          trace_gradient=trace_gradient)

    optimizer = optim.Adam(dqn_agent.policy_model.parameters(), lr=args.learning_rate)

    train_dqn(envs, optimizer, dqn_agent, num_episodes,
              timesteps_per_simulation, actions_per_episode,
              output_path)

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
