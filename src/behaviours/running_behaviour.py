#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import numpy as np
import vizdoom

from .hellboy_base_behaviour import HellboyBaseBehaviour


class RunningBehaviour(HellboyBaseBehaviour):

    def __init__(self, running_weight=0.5, ammo_weight=2):
        super().__init__()

        self.running_weight = running_weight
        self.ammo_weight = ammo_weight
        self.previous_position = None

    def get_reward(self):
        # living reward
        reward = 0  # -1

        pos_x = self.game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(vizdoom.GameVariable.POSITION_Y)

        position = np.array((pos_x, pos_y))
        distance_traveled = 0
        if self.previous_position is not None:
            distance_traveled = np.linalg.norm(self.previous_position - position)
        self.previous_position = position

        reward += distance_traveled * self.running_weight
        reward += self.ammo_reward() * self.ammo_weight

        return reward
