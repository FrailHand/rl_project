import cv2
import numpy as np
import os
import vizdoom
import vizdoomgym_duel


class EnvPlotter:
    def __init__(self, env: vizdoomgym_duel.ViZDoomGymDuel, scale=0.7):
        self.env = env
        self.scale = scale
        image_name = '.'.join((env.map_name, 'png'))
        filename = os.path.join(os.path.dirname(__file__), 'overlays', image_name)
        self.map_plot = cv2.imread(filename)
        self.last_position = None
        self.rendered = []

    def update(self):
        pos_x = self.env.game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        pos_y = self.env.game.get_game_variable(vizdoom.GameVariable.POSITION_Y)
        position = int(pos_x), int(pos_y)

        if self.last_position is not None:
            # self.map_plot[int(pos_y), int(pos_x), 0] = 255
            cv2.line(self.map_plot, self.last_position, position, (255, 0, 0), thickness=2)

        self.last_position = position

    def render(self):
        render = np.copy(self.map_plot)

        pos_x = self.env.game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        pos_y = self.env.game.get_game_variable(vizdoom.GameVariable.POSITION_Y)

        angle = -np.radians(self.env.game.get_game_variable(vizdoom.GameVariable.ANGLE))
        beta = np.radians(20)
        gamma = np.pi / 2 - angle - beta
        delta = angle - beta
        length = 30

        p1x = pos_x - np.sin(gamma) * length
        p1y = pos_y + np.cos(gamma) * length

        p2x = pos_x - np.cos(delta) * length
        p2y = pos_y + np.sin(delta) * length

        p0 = int(pos_x), int(pos_y)
        p1 = int(p1x), int(p1y)
        p2 = int(p2x), int(p2y)
        points = np.array((p0, p1, p2), dtype=int)
        cv2.fillConvexPoly(render, points, (0, 255, 255))

        shape = int(self.map_plot.shape[1] * self.scale), int(self.map_plot.shape[0] * self.scale)
        scaled = cv2.resize(render, shape)

        self.rendered.append(scaled)

    def save_gif(self, filename, max_frames=100):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.animation as ani
        import matplotlib.pyplot as plt

        modulo = len(self.rendered) // max_frames

        fig = plt.figure()
        matrices = []
        for index, mat in enumerate(self.rendered):
            if index % modulo == 0:
                matrices.append([plt.imshow(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB), animated=True)])

        animate = ani.ArtistAnimation(fig, matrices, interval=10, blit=True, repeat=False)
        gif_name = '.'.join((filename, 'gif'))
        print('Saving gif to: {}'.format(gif_name))
        animate.save(gif_name, writer='imagemagick', fps=20)

    def show(self, render=True, path_to_save=''):
        if render:
            self.render()

        cv2.imshow('Map', self.rendered[-1])
        if path_to_save != '':
            cv2.imwrite(path_to_save, self.rendered[-1])

        cv2.waitKey(1)
