#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import argparse
import os
import torch
from functools import partial
from torchvision.transforms import transforms

from src.plotting.env_plotter import EnvPlotter
from src.agents.drqn_agent import DRQNAgent
from src.behaviours.exploration_behaviour import ExplorationBehaviour
from src.behaviours.running_behaviour import RunningBehaviour
from src.datasets.data_transformer import *
from src.models.dqn import DQN
from src.models.dqn import DQNLatent
from src.models.drqn import DRQN
from src.models.vae import DoomVAE

str_device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(str_device)


def test_policy(env, policy, num_episodes, timesteps_per_simulation, timesteps_per_episode=2100,
                render_env=False):
    save_path = './visualizations'
    total_reward = 0

    for i_episode in range(num_episodes):
        plotter = EnvPlotter(env)

        done = False
        print("Testing episode [{}/{}]".format(i_episode + 1, num_episodes))

        # Initialize the environment and state
        state = env.reset()  # Get init/current state.

        next_state = None

        summed_reward_episode = 0
        t = 1
        for t in range(1, timesteps_per_episode + 1):  # count(start=0, step=1) #  0 1 2 3...
            if t % (timesteps_per_episode // 10) == 0:
                print('#', end='', flush=True)

            gym_action = policy.act(state, env.behaviour)

            # Execute action in emulator and observe new state and reward.
            for _ in range(timesteps_per_simulation):
                next_state, reward, done, _ = env.step(gym_action)
                summed_reward_episode += reward
                if done:
                    break
            if done:
                print("\nStopped episode {} early after {} timesteps.".format(i_episode + 1, t))
                break

            if render_env and device.type == "cpu":
                plotter.update()
                plotter.show()
                if hasattr(env.behaviour, "render_exploration_map") and callable(
                        env.behaviour.render_exploration_map):
                    env.behaviour.render_exploration_map(save_path)
                env.render()

            state = next_state

        print(' Avg-reward {}'.format(summed_reward_episode / t))
        total_reward += summed_reward_episode / t  # Save the average reward of the episode.

    uR = total_reward / (i_episode + 1)

    print("Average episode reward after {} episodes with max {} timesteps: {:.3f}.".format(
        num_episodes, timesteps_per_episode, uR))

    return uR


# old get policy
def get_policy_(out_dir, model_name, action_size):
    model_path = os.path.join(out_dir, 'results', model_name)
    policy = DRQNAgent.get_policy(model_path, action_size)

    return policy


def get_cnn_policy(action_size, model_path, is_rnn=False):
    if is_rnn:
        print('Testing with Recurrent Q Net')
        policy_model = DRQN(120, 160, 2, action_size)
    else:
        print('Testing with only CNN Q Net')
        policy_model = DQN(120, 160, 2, action_size)

    policy_model.load_state_dict(torch.load(model_path, map_location='cpu'))

    #  create dqn agent with policy
    policy = DRQNAgent(action_size,
                       None,
                       preprocess_function=None,
                       network=policy_model,
                       observe=0,
                       initial_epsilon=0.0001,
                       final_epsilon=0.0001)

    return policy


def get_vae_policy(action_size, vae_model_path, dqn_model_path, transform, z_dimension=64):
    z_size = z_dimension
    print("Dimension of latent space", z_size)
    vae = DoomVAE(2, 120, 120, z_size)
    vae.load_state_dict(torch.load(vae_model_path, map_location='cpu'))
    vae.eval()
    vae.to(device)
    this_transform = transform

    #  prerprocess with vae crop and stuff
    def vae_preprocess(tensor, this_device):
        tensor = this_transform(tensor)
        tensor = tensor.float().unsqueeze(0).to(this_device)
        mu, log_var = vae.encode(tensor)
        return mu

    # create agent and policy model
    policy_model = DQNLatent(z_size, action_size)
    policy_model.load_state_dict(torch.load(dqn_model_path, map_location='cpu'))

    # create dqn agent with policy
    policy = DRQNAgent(action_size,
                       None,
                       preprocess_function=vae_preprocess,
                       network=policy_model,
                       observe=0,
                       initial_epsilon=0.0001,
                       final_epsilon=0.0001)
    return policy


def get_env_behaviour(map_index=0, do_running=True):
    import vizdoomgym_duel as vgd
    if do_running:
        print("Testing Running behaviour")
        behaviour = RunningBehaviour()
    else:
        print("Testing Exploration behaviour")
        behaviour = ExplorationBehaviour()

    our_maps = ["map01", "map02", "map03", "map04", "map05"]
    print("Testing with map:", our_maps[map_index])
    env = vgd.ViZDoomGymDuel(behaviour, map_name=our_maps[map_index], server=True, players=1)

    return behaviour, env


def get_policy(args, action_size, z_dimension=128):
    # doom_input_w = 120
    # doom_input_h = 120

    if args.with_vae:
        #  test with vae coding preprocess
        transform = transforms.Compose([NpCenterCrop((120, 120)),
                                        # NpResize3d((doom_input_w, doom_input_h),
                                        #           interpolation=cv2.INTER_CUBIC),
                                        torch.from_numpy,
                                        partial(torch.div, other=255.)])
        policy = get_vae_policy(action_size, args.vae_model_path, args.dqn_model, transform, z_dimension)
    else:
        #  test cnn dqn
        policy = get_cnn_policy(action_size, args.dqn_model, is_rnn=args.is_rnn)
        # transform= None

    return policy


def validate_single_policy():
    parser = argparse.ArgumentParser(description='Train VAE on saved VizDoom explorations.')
    parser.add_argument('--episodes', type=int, default=10, metavar='N',
                        help='number of episodes to test (default: 10)')

    parser.add_argument('--path', metavar='DIR', type=str,
                        help="Root of the project. Has to be added to system path to allow imports on grid.")

    parser.add_argument('--with_vae', action='store_true', help='Test model with vae preprocess')

    parser.add_argument('--dqn_model', metavar='Model name', type=str,
                        help="Model name used when saving images.", required=False, default=None)

    parser.add_argument('--vae_model_path', type=str, default='', help='Full path to VAE model (detached)')
    parser.add_argument('--map_index', type=int, default=0)
    parser.add_argument('--is_rnn', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--z_dimension', type=int, default=128)

    args = parser.parse_args()

    #    if args.path is not None:
    #        sys.path.append(args.path)

    behaviour, env = get_env_behaviour(map_index=args.map_index, do_running=True)
    timesteps_per_simulation = 5
    action_size = behaviour.action_size
    # epsilon = 0.2

    # Test trained DQN.
    # policy= get_policy_(args.out_dir, args.model_name)
    policy = get_policy(args, action_size, z_dimension=args.z_dimension)

    ur = test_policy(env, policy, args.episodes,
                     timesteps_per_simulation,
                     timesteps_per_episode=2100,
                     render_env=args.render)

    # Test random policy.
    class Object(object):
        pass

    random_policy = Object()
    # random_policy.act = lambda state, behaviour: behaviour.sample_nonuniform()
    random_policy.act = lambda state, behaviour: behaviour.action_space.sample()
    policy = random_policy

    test_policy(env, policy, args.episodes, timesteps_per_simulation, timesteps_per_episode=2100,
                render_env=args.render)

    # env.render()
    env.close()


def validate_multiple_policy():
    parser = argparse.ArgumentParser(description='Train VAE on saved VizDoom explorations.')
    parser.add_argument('--episodes', type=int, default=10, metavar='N',
                        help='number of episodes to test (default: 10)')
    parser.add_argument('--input_path', metavar='DIR', type=str, help="Where are the models to be tested.")
    parser.add_argument('--output_path', metavar='DIR', type=str, help='Where to save the statistics')
    parser.add_argument('--with_vae', action='store_true', help='Test model with vae preprocess')
    parser.add_argument('--vae_model_path', type=str, default='', help='Full path to VAE model (detached)')
    parser.add_argument('--map_index', type=int, default=0)
    parser.add_argument('--is_running', action='store_true')
    parser.add_argument('--z_dimension', type=int, default=64)
    parser.add_argument('--is_rnn', action='store_true')

    args = parser.parse_args()

    print('Testing path', args.input_path)

    files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.nn')]
    files.sort()

    selected_files = [files[idx] for idx in range(0, len(files), 50)]

    behaviour, env = get_env_behaviour(map_index=args.map_index, do_running=args.is_running)
    timesteps_per_simulation = 5
    action_size = behaviour.action_size
    # epsilon = 0.2

    avg_reward = []
    for f in selected_files:
        # Test trained DQN.
        # policy= get_policy_(args.out_dir, args.model_name)
        print('Processing file:', os.path.basename(f))
        args.dqn_model = f
        policy = get_policy(args, action_size, args.z_dimension)

        ur = test_policy(env, policy, args.episodes,
                         timesteps_per_simulation,
                         timesteps_per_episode=2100,
                         render_env=False)

        avg_reward.append(ur)

        output_name = os.path.join(args.output_path, "val_reward_map_%d.txt" % args.map_index)
        np.savetxt(output_name, np.array(avg_reward), delimiter='t')


if __name__ == "__main__":
    validate_single_policy()
    # validate_multiple_policy()
