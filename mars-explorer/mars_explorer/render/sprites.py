import pygame as pg
import numpy as np

class Drone(pg.sprite.Sprite):
    def __init__(self, viewer, env, agent_id):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.env = env
        self.viewer = viewer
        self.agent_id = agent_id
        self.rotate_img()
        self.rect = self.image.get_rect()

    def update(self):
        self.rotate_img()

        x, y = self.env.positions[self.agent_id]

        self.rect.x = x * self.viewer.TILESIZE
        self.rect.y = y * self.viewer.TILESIZE

    def rotate_img(self):
        if self.env.last_actions[self.agent_id] == 0:
            self.image = pg.transform.rotate(self.viewer.drone_img, 90)
        elif self.env.last_actions[self.agent_id] == 1:
            self.image = pg.transform.rotate(self.viewer.drone_img, -90)
        elif self.env.last_actions[self.agent_id] == 2:
            self.image = pg.transform.rotate(self.viewer.drone_img, 0)
        elif self.env.last_actions[self.agent_id] == 3:
            self.image = pg.transform.rotate(self.viewer.drone_img, 180)

class Obstacle(pg.sprite.Sprite):
    def __init__(self, viewer, x, y):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.image = viewer.obstacle_img
        # self.image = pg.Surface((TILESIZE, TILESIZE))
        # self.image.fill(OBSTACLE_IMG)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * self.viewer.TILESIZE
        self.rect.y = y * self.viewer.TILESIZE
