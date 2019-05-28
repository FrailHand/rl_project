#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import math

import copy
import numpy as np
import random
import torch
import torch.nn.functional as F


class DRQNAgent:

    def __init__(self, action_size, replay_memory, network,
                 preprocess_function=None,
                 gamma=0.99,
                 learning_rate=0.0001,
                 initial_epsilon=1.0,
                 final_epsilon=0.0001,
                 epsilon_half_life=2000,
                 epsilon_decay='LIN',  # 'LIN' or 'EXP'
                 batch_size=32,
                 observe=5000,
                 trace_gradient=1,
                 update_target_freq=3000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get size of state and action
        self.action_size = action_size

        # these is hyper parameters for the DQN.
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_range = self.initial_epsilon - self.final_epsilon
        self.epsilon_half_life = epsilon_half_life
        self.epsilon_decay = epsilon_decay.upper()
        self.batch_size = batch_size
        self.observe = observe  # 5000 # Start decreasing epsilon from 1.0 after this amount of observations.
        self.epsilon_steps = 0
        self.update_target_freq = update_target_freq
        self.trace_gradient = trace_gradient
        self.memory = replay_memory

        self.preprocess_function = preprocess_function

        # Create main model and target model
        self.policy_model = network
        self.policy_model.to(self.device)  # FIXME unclear where device is really handled
        self.target_model = copy.deepcopy(network)  # Initialize target action-value function Q' with weights of Q.
        self.target_model.to(self.device)

        self.hidden_state = None

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.mavg_ammo_left = []  # Moving Average of Ammo used
        self.mavg_kill_counts = []  # Moving Average of Kill Counts

        self.verbose()

    def init_hidden_state(self, value=None):
        self.hidden_state = value

    def act(self, state, behaviour):
        state = self.preprocess_state(state)
        action = self.select_action(state)

        gym_action = action.item()

        return gym_action

    @staticmethod
    def get_policy(model_path, action_size):
        # TODO fix this line not good
        policy = DRQNAgent(action_size, replay_memory_size=10000, observe=0)
        policy.load_model(model_path + ".nn")  # Load already trained model.
        policy.epsilon = policy.final_epsilon

        return policy

    @property
    def target_model(self):
        return self._target_model

    @target_model.setter
    def target_model(self, value):
        self._target_model = value
        self._target_model.eval()

    def update_target_model(self, steps_done=0):
        """
        After some time interval update the target model to be same with policy model.
        """
        if steps_done % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.target_model.eval()

    def select_action(self, state):
        """
        Select action from model using epsilon-greedy policy
        """
        with torch.no_grad():
            self.policy_model.eval()
            q, hidden = self.policy_model(state, self.hidden_state)
            self.policy_model.train()
        self.hidden_state = hidden

        if np.random.rand() <= self.epsilon:
            action_idx = torch.tensor([[random.randrange(self.action_size)]], device=self.device,
                                      dtype=torch.long)
        else:
            action_idx = q.argmax(-1).view(1, 1)
        return action_idx

    def default_preprocess(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        # the array is a pytorch tensor already
        state = state.float().unsqueeze(0).to(self.device)  # Add batch dim
        return state

    def preprocess_state(self, state):
        if self.preprocess_function is None:
            return self.default_preprocess(state)
        else:
            return self.preprocess_function(state, self.device)

    def update_epsilon(self):
        if len(self.memory) > 0:
            self.epsilon_steps += 1
            explore_steps = self.epsilon_steps - self.observe

            if self.epsilon > self.final_epsilon and explore_steps > 0:

                if self.epsilon_decay == 'LIN':
                    self.epsilon = self.initial_epsilon - self.epsilon_range * explore_steps / (
                            self.epsilon_half_life * 2)
                elif self.epsilon_decay == 'EXP':
                    self.epsilon = self.final_epsilon + self.epsilon_range * math.exp(
                        -1. * explore_steps / (self.epsilon_half_life / math.log(2)))

    def optimize_model(self, optimizer):
        """
        Performs a single step of the optimization. It first samples a batch, concatenates all the tensors into a single
        one, computes Q(st,at) and V(st+1)=maxaQ(st+1,a), and combines them into our loss. By definition we set V(s)=0
        if s is a terminal state. We also use a target network to compute V(st+1) for added stability. The target net-
        work has its weights kept frozen most of the time, but is updated with the policy network’s weights every so
        often. This is usually a set number of steps but we shall use episodes for simplicity.

        :return: None
        """

        # start training as soon as we have one episode stored
        # print('[INFO] the size of memory is', len(self.memory))
        if len(self.memory) < self.batch_size:
            return -1

        state_batch, action_batch, next_state_batch, reward_batch = self.memory.sample_batch(
            self.batch_size)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_model.
        policy_output = self.policy_model(state_batch)[0]
        state_action_values = policy_output.gather(-1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        target_output = self.target_model(next_state_batch)[0]
        next_state_values = target_output.max(-1, keepdim=True)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.unsqueeze(-1)

        state_action_values = state_action_values[..., -self.trace_gradient:]
        expected_state_action_values = expected_state_action_values[..., -self.trace_gradient:]

        # Compute Huber loss. Q(s_t, a) != r_a + V(s_{t+1}) == r_a + max_a Q(s_{t+1})
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        del state_batch
        del action_batch
        del next_state_batch
        del reward_batch

        return loss.item()

    def load_model(self, name, copy_to_target_model=True):
        """
        Load the saved model.

        :param name:
        :return:
        """

        # The lambda expression makes it irrelevant if the checkpoint was saved from CPU or GPU.
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)
        self.policy_model = checkpoint['model']
        print("Loaded policy model from {}.".format(name))

        if copy_to_target_model and self.target_model is not None:
            self.update_target_model()

    def save_model(self, name):
        """
        Save the model which is under training.

        :param name:
        :return:
        """
        # torch.save({'model': self.policy_model}, name)
        torch.save(self.policy_model.state_dict(), name)
        print("Saved policy model at {}.".format(name))

    def verbose(self):
        print('[INFO] ({}) Configuration'.format(self.__class__.__name__))
        print('[INFO] ({}) gamma= {}'.format(self.__class__.__name__, self.gamma))
        print('[INFO] ({}) learning_rate= {}'.format(self.__class__.__name__, self.learning_rate))
        print('[INFO] ({}) epsilon= {}'.format(self.__class__.__name__, self.epsilon))
        print('[INFO] ({}) initial_epsilon= {}'.format(self.__class__.__name__, self.initial_epsilon))
        print('[INFO] ({}) final_epsilon= {}'.format(self.__class__.__name__, self.final_epsilon))
        # print('[INFO] ({}) decay_epsilon= {}'.format(self.__class__.__name__, self.decay_epsilon))
        print('[INFO] ({}) batch_size= {}'.format(self.__class__.__name__, self.batch_size))
        print('[INFO] ({}) observe= {}'.format(self.__class__.__name__, self.observe))
        print('[INFO] ({}) epsilon_steps= {}'.format(self.__class__.__name__, self.epsilon_steps))
        print('[INFO] ({}) update_target_freq= {}'.format(self.__class__.__name__, self.update_target_freq))
        print('[INFO] ({}) replay_memory_size= {}'.format(self.__class__.__name__,
                                                          0 if self.memory is None else self.memory.capacity))
