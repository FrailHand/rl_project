#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import numpy as np
import vizdoom

from .running_behaviour import RunningBehaviour


class ExplorationBehaviour(RunningBehaviour):

    def __init__(self, fov=90, fov_r=80, forget_shape=5e-3, forget_speed=0.02,
                 discover_weight=1e-1, ammo_weight=0.5, map_downsampling=5):
        super().__init__(ammo_weight=ammo_weight)
        map_size = int((1200 + 3 * fov_r) / map_downsampling)

        self.exploration_grid = np.full((map_size, map_size), - np.float('Inf'))

        self.forget_shape = forget_shape
        self.forget_speed = forget_speed
        self.fov_r = fov_r / map_downsampling
        self.fov = np.deg2rad(fov)
        self.discover_weight = discover_weight
        self.down_sampling = map_downsampling

    def reset(self):
        super().reset()
        self.exploration_grid.fill(-np.float('Inf'))

    def fov_side(self, angle):
        return self.fov_r * np.array((np.cos(angle), np.sin(angle)))

    @staticmethod
    def y_on_line(p1, p2, x):
        d = p1 - p2
        if d[0] == 0:
            raise Exception('Divide by zero')
        return p1[1] + d[1] / d[0] * (x - p1[0])

    @staticmethod
    def x_coord(x):
        return x[0]

    # set range of indices as explored area
    def scan_y_deltas(self, y1_array, y2_array, x_array, time):
        all_deltas = []
        for index, x in enumerate(x_array):
            y1 = y1_array[index]
            y2 = y2_array[index]

            sign = np.sign(y2 - y1)
            if sign == 0:
                sign = 1
            stop = y2 + sign

            # get section of the map expored
            row = self.exploration_grid[x, y1:stop:sign]
            all_deltas.append(time - row)
            row.fill(time)

        return np.concatenate(all_deltas)

    def render_exploration_map(self, save_path=None, episode=None):
        import cv2
        time = self.game.get_episode_time()
        img = 1 - (np.power(1 + self.forget_shape,
                            time - self.exploration_grid) - 1) * self.forget_speed
        img[img < 0] = 0

        img = img * 255.0
        img = img.astype(np.uint8)
        img = np.transpose(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.applyColorMap(img, cv2.COLORMAP_PARULA)

        if save_path is not None and episode is not None:
            cv2.imwrite(save_path + '/knowledge_at_%d.jpg' % episode, img)

        cv2.imshow('Knowledge', img)
        cv2.waitKey(1)

    def compute_observation_reward(self, deltas):
        local = (np.power(1 + self.forget_shape, deltas) - 1) * self.forget_speed
        local[local > 1] = 1

        return np.sum(local)

    def half_fov_explore(self, point_start, point_end, point_third, time, reverse=False):
        if reverse:
            x = np.arange(np.ceil(point_end[0]), np.floor(point_start[0]), dtype=int)
        else:
            x = np.arange(np.ceil(point_start[0]), np.floor(point_end[0]), dtype=int)
        if x.size == 0:
            return x

        y1 = self.y_on_line(point_start, point_end, x).astype(int)
        y2 = self.y_on_line(point_start, point_third, x).astype(int)

        # set area as explored and return difference with current time
        return self.scan_y_deltas(y1, y2, x, time)

    def get_reward(self):
        # living reward
        reward = 0  # -1

        # retrieve x and y on the map
        pos_x = self.game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(vizdoom.GameVariable.POSITION_Y)
        angle = np.deg2rad(self.game.get_game_variable(vizdoom.GameVariable.ANGLE))

        time = self.game.get_episode_time()

        # transform to grid positions
        pos = np.array((pos_x, pos_y)) / self.down_sampling + 2 * self.fov_r
        pos1 = pos + self.fov_side(angle + self.fov / 2)
        pos2 = pos + self.fov_side(angle - self.fov / 2)

        # get sorted points to set exploration flags
        points = sorted((pos, pos1, pos2), key=self.x_coord)

        fov_deltas = []

        # set covered area of a triangle as explored region
        fov_deltas.append(self.half_fov_explore(points[0], points[1], points[2], time))

        x = np.atleast_1d(np.floor(points[1][0])).astype(int)
        y1 = np.atleast_1d(points[1][1]).astype(int)
        y2 = np.atleast_1d(self.y_on_line(points[2], points[0], x)).astype(int)

        fov_deltas.append(self.scan_y_deltas(y1, y2, x, time))

        fov_deltas.append(self.half_fov_explore(points[2], points[1], points[0], time, True))

        fov_deltas = np.concatenate(fov_deltas)

        reward += self.compute_observation_reward(fov_deltas) * self.discover_weight

        reward += self.ammo_reward() * self.ammo_weight

        return reward
