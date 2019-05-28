#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import gym
import numpy as np
import vizdoom


# Only edit this to define your agent
class Behaviour:
    # list of the available actions for the agent

    def __init__(self):
        self.game = None
        self.action_space = None
        self.observation_space = None

        self._available_buttons_binary = [
            vizdoom.Button.ATTACK,

            vizdoom.Button.TURN_LEFT,
            vizdoom.Button.TURN_RIGHT,

            vizdoom.Button.MOVE_LEFT,
            vizdoom.Button.MOVE_RIGHT,
            vizdoom.Button.MOVE_FORWARD,
            vizdoom.Button.MOVE_BACKWARD,
        ]

        self._available_buttons_delta = [
            vizdoom.Button.TURN_LEFT_RIGHT_DELTA,
        ]

    # initialize the game according to your needs
    def init(self, game: vizdoom.DoomGame):
        self.game = game

        self.game.clear_available_buttons()

        for button in self._available_buttons_binary:
            self.game.add_available_button(button)
        for button in self._available_buttons_delta:
            self.game.add_available_button(button)

        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)

        self.game.add_game_args("+name Agent +colorset 0")

        binary_actions = gym.spaces.MultiBinary(len(self._available_buttons_binary))
        value_actions = gym.spaces.Box(low=-180, high=180, shape=(len(self._available_buttons_delta),), dtype=np.int16)

        self.action_space = gym.spaces.Tuple((binary_actions, value_actions))
        self.observation_space = gym.spaces.Box(0, 255, (self.game.get_screen_height(),
                                                         self.game.get_screen_width(),
                                                         self.game.get_screen_channels()),
                                                dtype=np.uint8)

    # translate a gym action into a vizdoom compatible one
    def parse_action(self, action):
        return np.concatenate(action, axis=0).tolist()

    # sample a random action according to this behaviour
    def sample_nonuniform(self):
        return self.action_space.sample()

    # reward shaping
    def get_reward(self):
        reward = 0
        if self.game.is_player_dead():
            reward -= 5

        return reward

    # define your observation
    # if the episode is finished, will return None!
    def get_observation(self):
        state = self.game.get_state()

        if state is None:
            return None

        observation = np.transpose(state.screen_buffer, (1, 2, 0))
        return observation

    # reset the behaviour
    def reset(self):
        return
