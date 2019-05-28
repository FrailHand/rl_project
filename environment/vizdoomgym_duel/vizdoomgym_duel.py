#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import gym
import numpy as np
import os
import vizdoom
import time


class ViZDoomGymDuel(gym.Env):

    def __init__(self, behaviour, server=False, map_name="map01", players=2):
        self.behaviour = behaviour
        self.is_server = server

        # init game
        self.game = vizdoom.DoomGame()

        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, 'duel.cfg'))

        self.game.set_window_visible(False)

        self.map_name = map_name
        self.game.set_doom_map(map_name)

        if server:
            self.game.add_game_args("-host {} "
                                    "-deathmatch "
                                    "+timelimit 15.0 "
                                    "+sv_forcerespawn 1 "
                                    "+sv_noautoaim 1 "
                                    "+sv_respawnprotect 1 "
                                    "+sv_spawnfarthest 1 "
                                    "+sv_nocrouch 1 "
                                    "+viz_respawn_delay 5 ".format(players))
        else:
            self.game.add_game_args("-join 127.0.0.1")

        self.game.set_mode(vizdoom.Mode.PLAYER)

        self.behaviour.init(self.game)

        init_passed = False
        while not init_passed:
            try:
                self.game.init()
                init_passed = True
            except vizdoom.ViZDoomUnexpectedExitException as e:
                print('Exception caught while initializing the game! Retrying...')
                time.sleep(0.5)

        self.action_space = self.behaviour.action_space
        self.observation_space = self.behaviour.observation_space

        self.viewer = None

    def step(self, action):
        act = self.behaviour.parse_action(action)

        # using the state variable would make much more sense
        self.game.make_action(act)

        reward = self.behaviour.get_reward()

        if self.game.is_player_dead():
            self.game.respawn_player()

        done = self.game.is_episode_finished()

        observation = self.behaviour.get_observation()

        info = {}

        return observation, reward, done, info

    def reset(self):
        self.game.new_episode()
        self.behaviour.reset()

        return self.behaviour.get_observation()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        except AttributeError:
            pass
