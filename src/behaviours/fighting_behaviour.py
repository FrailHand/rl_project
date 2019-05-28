#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import os
import vizdoom

from .hellboy_base_behaviour import HellboyBaseBehaviour


class FightingBehaviour(HellboyBaseBehaviour):
    def __init__(self, ammo_weight=1, dying_penalty=5, frag_reward=10, health_weight=5e-2, hit_weight=3):
        super().__init__()
        self.ammo_weight = ammo_weight
        self.dying_penalty = dying_penalty
        self.frag_reward = frag_reward
        self.health_weight = health_weight
        self.hit_weight = hit_weight

        self.hits = 0
        self.frags = 0

    def init(self, game):
        FightingBehaviour.config_dumbot()
        super().init(game)

    def reset(self):
        super().reset()
        self.game.send_game_command("removebots")
        self.game.send_game_command("addbot")

        self.hits = 0
        self.frags = 0

    @staticmethod
    def config_dumbot():
        cfg = '''{
                    name        Dumbot
                    aiming      1
                    perfection  1
                    reaction    1
                    isp         1
                    color       "40 cf 00"
                    skin        base
                }
                '''

        bots_file = os.path.join(os.path.dirname(vizdoom.__file__), 'bots.cfg')

        with open(bots_file, 'w') as f:
            f.write(cfg)

    def get_reward(self):
        # No living penalty
        reward = 0

        # Dead penalty
        if self.game.is_player_dead():
            reward -= self.dying_penalty
            self.health = -1
            self.ammo = -1

        else:
            # Losing health penalty~
            reward += self.health_reward() * self.health_weight

            # Picking up ammo reward
            reward += self.ammo_reward() * self.ammo_weight

        # Hitting the ennemy reward
        hits = self.game.get_game_variable(vizdoom.GameVariable.HITCOUNT)
        reward += (hits - self.hits) * self.hit_weight
        self.hits = hits

        # Killing the ennemy reward
        frags = self.game.get_game_variable(vizdoom.GameVariable.FRAGCOUNT)
        reward += (frags - self.frags) * self.frag_reward
        self.frags = frags

        return reward
