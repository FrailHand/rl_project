#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import gym
import numpy as np
import vizdoom

import vizdoomgym_duel as vgd


class HellboyBaseBehaviour(vgd.Behaviour):
    LABELS_DICT = {
        'DoomPlayer': 191,
        'Clip': 127,
    }
    DEFAULT_LABEL = 63

    def __init__(self):
        super().__init__()
        self._available_buttons_binary = [
            vizdoom.Button.MOVE_FORWARD,
            vizdoom.Button.MOVE_BACKWARD,

            vizdoom.Button.MOVE_LEFT,
            vizdoom.Button.MOVE_RIGHT,

            vizdoom.Button.TURN_LEFT,
            vizdoom.Button.TURN_RIGHT,

            vizdoom.Button.ATTACK,
        ]
        self._available_buttons_delta = ()

        self.health = -1
        self.ammo = -1

    def init(self, game: vizdoom.DoomGame):
        super().init(game)
        self.game.set_screen_resolution(vizdoom.RES_160X120)
        self.game.set_render_corpses(False)

        self.game.add_game_args("+name Hellboy +colorset 3")

        self.action_space = gym.spaces.Discrete(len(self._available_buttons_binary) + 1)
        self.observation_space = gym.spaces.Box(0, 255, (self.game.get_screen_height(),
                                                         self.game.get_screen_width(),
                                                         2),
                                                dtype=np.uint8)

    def reset(self):
        super().reset()

        self.health = -1
        self.ammo = -1

    @property
    def action_size(self):
        return len(self._available_buttons_binary)

    # translate a gym action into a vizdoom compatible one
    def parse_action(self, action):
        viz_action = np.zeros(self.action_size)
        if action:
            viz_action[action - 1] = 1
        return viz_action.tolist()

    def sample_nonuniform(self):
        ac = np.arange(6, dtype=np.int8)
        choice = np.random.choice(ac, 1, replace=False, p=[0.5, 0, 0.13, 0.17, 0.12, 0.08])

        return choice + 1

    def get_observation(self):
        state = self.game.get_state()

        if state is None:
            return None

        labels_buffer = self.get_processed_labels(state)
        depth_buffer = state.depth_buffer

        observation = np.stack((labels_buffer, depth_buffer), axis=0)
        return observation

    def get_processed_labels(self, state):

        labels = state.labels
        labels_buffer = state.labels_buffer

        translate_dict = {}

        for l in labels:
            if l.value != 255:
                try:
                    translate_dict[l.value] = self.LABELS_DICT[l.object_name]
                except KeyError:
                    translate_dict[l.value] = self.DEFAULT_LABEL

        processed_buffer = np.copy(labels_buffer)

        for key, val in translate_dict.items():
            processed_buffer[labels_buffer == key] = val

        return processed_buffer

    def ammo_reward(self):
        ammo = self.game.get_game_variable(vizdoom.GameVariable.SELECTED_WEAPON_AMMO)
        if self.ammo == -1:
            self.ammo = ammo
            return 0
        delta = ammo - self.ammo
        #        if delta > 0:
        #            print('+++++++++++++++++++++++++++++++++')
        self.ammo = ammo
        return delta

    def health_reward(self):
        health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        if self.health == -1:
            self.health = health
            return 0
        delta = health - self.health
        self.health = health
        return delta
