#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import torch
import torch.optim as optim

from src.agents.drqn_agent import DRQNAgent
# from src.models.dqn import DQN
from src.agents.replay_memory import ContextReplayMemory
from src.behaviours.exploration_behaviour import ExplorationBehaviour
from src.models.drqn import DRQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_drqn(envs, optimizer, agent, num_episodes, timesteps_per_simulation, actions_per_episode):
    agent.update_target_model()
    done = False
    for i_episode in range(num_episodes):
        env = envs[i_episode % len(envs)]
        # Initialize the environment and state
        state = env.reset()  # Get init/current state.

        state = agent.preprocess_state(state)
        next_state = None

        agent.init_hidden_state()

        for t in range(0, actions_per_episode):  # count(start=0, step=1) #  0 1 2 3...
            # Select a (random) action.
            action = agent.select_action(state)

            # Convert action number to gym action.
            gym_action = action.item()

            summed_reward = 0
            # Execute action in emulator and observe new state and reward.
            for _ in range(timesteps_per_simulation):
                next_state, reward, done, _ = env.step(gym_action)
                summed_reward += reward
                if done:
                    break
            if done:
                break

            # if i_episode > 0:
            # # if i_episode != 0 and (i_episode + 1) % 10 == 0:
            #     env.behaviour.render_exploration_map()
            #     env.render()
            reward = torch.tensor([summed_reward], device=device)

            next_state = agent.preprocess_state(next_state)

            agent.memory.push(state, action, next_state, reward)

            agent.update_epsilon()

            loss = agent.optimize_model(optimizer)
            print("Episode [{}/{}], t=[{}/{}], loss={:.3f}, eps={:.3f}".format(i_episode, num_episodes, t,
                                                                               actions_per_episode, loss,
                                                                               agent.epsilon))

            agent.update_target_model(t)

            state = next_state

        agent.memory.push_episode()

    print('Complete')


def main():
    import vizdoomgym_duel as vgd
    behaviour = ExplorationBehaviour()
    envs = [vgd.ViZDoomGymDuel(behaviour, map_name="map01", server=True, players=1),
            vgd.ViZDoomGymDuel(behaviour, map_name="map02", server=True, players=1),
            vgd.ViZDoomGymDuel(behaviour, map_name="map03", server=True, players=1),
            vgd.ViZDoomGymDuel(behaviour, map_name="map04", server=True, players=1),
            vgd.ViZDoomGymDuel(behaviour, map_name="map05", server=True, players=1)]

    action_size = behaviour.action_size

    num_episodes = 50
    timesteps_per_simulation = 5
    actions_per_episode = 2100  # 2100
    observe = actions_per_episode * len(envs)

    trace_length = 10
    trace_gradient = 3
    replay_memory = ContextReplayMemory(len(envs), trace_length)
    agent = DRQNAgent(action_size, replay_memory, DRQN(120, 160, 2, action_size), trace_gradient=trace_gradient,
                      observe=observe)  # Initialize agent with replay memory.

    # replay_memory = ReplayMemory(actions_per_episode * len(envs))
    # agent = DRQNAgent(action_size, replay_memory, DQN(120, 160, 2, action_size), observe=observe)  # Initialize agent with replay memory.

    optimizer = optim.RMSprop(agent.policy_model.parameters())

    train_drqn(envs, optimizer, agent, num_episodes, timesteps_per_simulation, actions_per_episode)

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
