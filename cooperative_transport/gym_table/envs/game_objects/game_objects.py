import math
import os
import pickle
from typing import List, Tuple

import numpy as np
import pygame
from PIL import Image

from cooperative_transport.gym_table.envs.utils import (
    VERBOSE,
    WINDOW_H,
    WINDOW_W,
    I,
    L,
    b,
    d,
    debug_print,
    m,
)


class Obstacle(pygame.sprite.Sprite):
    """Obstacle object."""

    def __init__(self, position: np.ndarray, size=(50, 50)) -> None:
        """Initialize obstacle.

        Parameters
        ----------
        position : np.ndarray, shape=(2)
            Planar position of the obstacle.
        """
        # create sprite
        pygame.sprite.Sprite.__init__(self)

        # relative paths
        dirname = os.path.dirname(__file__)
        obstacle_path = os.path.join(dirname, "images/obstacle.png")

        # visuals
        self.original_img = pygame.image.load(obstacle_path).convert()
        self.original_img = pygame.transform.scale(self.original_img, size)
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2


class Target(pygame.sprite.Sprite):
    """Target object."""

    def __init__(self, position: np.ndarray) -> None:
        """Initialize the target.

        Parameters
        ----------
        position : np.ndarray, shape=(2)
            Planar position of the target.
        """
        # relative paths
        dirname = os.path.dirname(__file__)
        target_path = os.path.join(dirname, "images/target.png")

        # initial conditions
        self.x = position[0]
        self.y = position[1]

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.original_img = pygame.image.load(target_path).convert()
        self.original_img.convert_alpha()
        self.image = self.original_img
        self.rect = self.image.get_rect()

        self.rect.x = self.x - self.rect.size[0] / 2
        self.rect.y = self.y - self.rect.size[1] / 2


class Table(pygame.sprite.Sprite):
    """Table object."""

    def __init__(
        self, x=0.25 * WINDOW_W, y=0.25 * WINDOW_H, angle=0.0, length=L
    ) -> None:
        """Initialize the table.

        Parameters
        ----------
        x : float
            Initial x position of the center of the table sprite.
        y : float
            Initial y position of the center of the table sprite.
        angle : float
            Initial angle of the table sprite.

        """
        # defining relative paths
        dirname = os.path.dirname(__file__)
        table_img_path = os.path.join(dirname, "images/table.png")

        # initial conditions
        self.x_speed = 0.0
        self.y_speed = 0.0
        self.angle_speed = 0.0
        self.x = x
        self.y = y
        self.angle = angle
        # log the previous state
        self.px = 0.25 * WINDOW_W
        self.py = 0.25 * WINDOW_H
        self.pangle = 0.0

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        original_img = Image.open(table_img_path)
        mode = original_img.mode
        sz = original_img.size
        data = original_img.tobytes()
        self.original_img = pygame.image.fromstring(data, sz, mode)
        self.w, self.h = self.original_img.get_size()
        self.length_from_center_to_person = length / 2
        self.table_center_to_player1 = np.array(
            [
                self.x + self.length_from_center_to_person * np.cos(self.angle),
                self.y + self.length_from_center_to_person * np.sin(self.angle),
            ]
        )
        self.table_center_to_player2 = np.array(
            [
                self.x - self.length_from_center_to_person * np.cos(self.angle),
                self.y - self.length_from_center_to_person * np.sin(self.angle),
            ]
        )
        # get a rotated image
        self.image = pygame.transform.rotate(self.original_img, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)

        # define force limits
        self.force_cap = 1.0
        # define angular acceleration limits
        self.cap_alpha_min = -np.pi / 2
        self.cap_alpha_max = np.pi / 2
        # define angular velocity limits
        self.min_velocity_angle = -np.pi / 4
        self.max_velocity_angle = np.pi / 4

        # scale the forces (range: [-1, 1]) by this factor for the pygame
        self.policy_scaling_factor = 50

    def acceleration(
        self, f1: np.ndarray, f2: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute the acceleration given forces.

        Parameters
        ----------
        f1 : np.ndarray, shape=(2)
            Force 1. Note that *player1* is on the right (blue triangle / arrow keys if using keyboard).
        f2 : np.ndarray, shape=(2)
            Force 2. Note that *player 2* is on left (orange circle / wasd if using keyboard).

        Returns
        -------
        a_x : float
            Linear x acceleration.
        a_y : float
            Linear y acceleration.
        a_angle : float
            Angular acceleration.
        """
        # equations of motion for table

        a_x = -b / m * self.x_speed + 1.0 / m * (f1[0] + f2[0])
        a_y = -b / m * self.y_speed + 1.0 / m * (f1[1] + f2[1])
        M_z = (
            L
            / 2.0
            * (
                math.sin(self.angle) * (f2[0] - f1[0])
                + math.cos(self.angle) * (-f1[1] + f2[1])
            )
        )
        debug_print("Mz", M_z, self.angle_speed, self.angle)
        a_angle = -d / I * self.angle_speed + 1.0 / I * M_z
        a_angle = np.clip(a_angle, self.cap_alpha_min, self.cap_alpha_max)
        return a_x, a_y, a_angle

    def velocity(self, ax, ay, a_angle, dt: float):
        """Compute the velocity given acceleration."""
        vx = self.x_speed + ax * dt
        vy = self.y_speed + ay * dt
        va = self.angle_speed + a_angle * dt
        debug_print("velocity", vx, vy, va)
        return vx, vy, va

    def update(
        self,
        action: np.ndarray,
        delta_t: float,
        update_image=True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Internal updates for the table.

        Parameters
        ----------
        action : np.ndarray, shape=(2)
            Action taken by the RL agent.
        delta_t : float
            Temporal resolution of the dynamics.
        update_image : bool
            Flag indicating whether to update the image.

        Returns
        -------
        f1 : np.ndarray, shape=(2)
            Actions for the non-RL synthetic agent.
        f2 : np.ndarray, shape=(2)
            Actions for the RL agent.
        x : float
            X position of the RL agent.
        y : float
            Y position of the RL agent.
        """
        # take action (RL agent)
        action_clipped = np.clip(action, -self.force_cap, self.force_cap)
        debug_print("Action (clipped)", action_clipped.shape, action_clipped)
        player_1_act = action_clipped[0, :]
        player_2_act = action_clipped[1, :]

        f2s = np.array(player_2_act) * self.policy_scaling_factor
        f1s = np.array(player_1_act) * self.policy_scaling_factor

        # store previous step's values
        self.px = self.x
        self.py = self.y
        self.pangle = self.angle

        # update position using acceleration. Note that we set position first, using a set delta_t
        a_x, a_y, a_angle = self.acceleration(f1s, f2s)
        self.x = self.x + self.x_speed * delta_t + 0.5 * a_x * delta_t * delta_t
        self.y = self.y + self.y_speed * delta_t + 0.5 * a_y * delta_t * delta_t
        self.angle = (
            self.angle + self.angle_speed * delta_t + 0.5 * a_angle * delta_t * delta_t
        )
        self.angle = self.angle % (2 * np.pi)

        # update velocity
        self.x_speed, self.y_speed, self.angle_speed = self.velocity(
            a_x, a_y, a_angle, delta_t
        )

        debug_print("Updated x, y, angle", self.x, self.y, self.angle)

        # get a rotated image
        angle = math.degrees(self.angle)
        if update_image:
            self.image = pygame.transform.rotate(self.original_img, angle)
            self.rect = self.image.get_rect(center=(self.x, self.y))
            self.mask = pygame.mask.from_surface(self.image)

        return (
            f1s,
            f2s,
            self.x,
            self.y,
            self.angle,
            self.x_speed,
            self.y_speed,
            self.angle_speed,
        )


class Agent(object):
    def __init__(self) -> None:
        super(Agent, self).__init__()
        # Agents have fx and fy
        self.f = np.array([0, 0])
        # Limit the action magnitude
        self.cap = 1.0
        self.policy_scaling_factor = 50.0
