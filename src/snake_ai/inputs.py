# function for line generation
import math

import numpy as np
import pygame
from numba import njit

from fast_snake.fast_snake import TileTypes


@njit
def bresenham(x1, y1, x2, y2):
    """
    bresenham algorithm for all cases
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: the points
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    if dx > dy:
        err = dx / 2.0
        while abs(x - x2) > 0.5:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while abs(y - y2) > 0.5:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return np.array(points)


def draw_ai_inputs(game_surface, game_array, food_pos, snake_head_pos, inputs, n_directions=8, start_angle=0,
                   end_angle=2 * math.pi):
    """
    Renders the lines which represent the distance between the snake_data head and the 4 walls
    """
    grid_size = game_surface.get_width() / game_array.shape[1]
    food_pos = pygame.Vector2(food_pos[1], food_pos[0])
    snake_head_to_food = food_pos - snake_head_pos
    # draw vectors
    pygame.draw.line(game_surface, (255, 255, 0),
                     (snake_head_pos.x * grid_size + grid_size / 2, snake_head_pos.y * grid_size + grid_size / 2),
                     ((snake_head_pos.x + snake_head_to_food.x) * grid_size + grid_size / 2,
                      (snake_head_pos.y + snake_head_to_food.y) * grid_size + grid_size / 2), int(grid_size / 5))
    angle = 0
    # angles = None
    # if start_angle is not None:
    #     angles = np.linspace(start_angle, end_angle, n_directions)
    # angles = np.linspace(start_angle, end_angle, n_directions)
    for index, distance in enumerate(inputs[:n_directions]):
        # if angles is not None:
        #     angle = angles[index]
        # angle = angles[index]
        vector = pygame.Vector2(math.cos(angle), math.sin(angle))
        pygame.draw.line(game_surface, (255, 255, 0),
                         (snake_head_pos.x * grid_size + grid_size / 2, snake_head_pos.y * grid_size + grid_size / 2),
                         ((snake_head_pos.x + vector.x * distance) * grid_size + grid_size / 2,
                          (snake_head_pos.y + vector.y * distance) * grid_size + grid_size / 2), int(grid_size / 5))
        angle += 2 * math.pi / n_directions


@njit
def get_body_distance_in_n_directions(snake, game_array, n=8, angles=None):
    """
    Using bresenham navigate in 8 directions from the snake head and check if a body part is in the way
    rotate the rays in the given direction
    :param angles: if None, the angles are calculated using the n_directions
    :param snake: snake_data
    :param game_array: game_array board
    :param n: number of directions to check
    """
    grid_size = game_array.shape[1]
    snake_head_pos = snake[-1][1], snake[-1][0]
    snake_head_to_body_distances = np.zeros(n)
    angle_index = 0
    angle = 0
    for i in range(n):
        if angles is not None:
            angle = angles[i]
        # make a ray from the snake_data head in the direction of the angle to the end of the grid
        vector = math.cos(angle) * grid_size * 2, math.sin(angle) * grid_size * 2
        ray = bresenham(snake_head_pos[0], snake_head_pos[1], snake_head_pos[0] + vector[0],
                        snake_head_pos[1] + vector[1])
        # check if the ray intersects with a body part
        has_intersection(angle_index, game_array, ray, (snake_head_pos[0], snake_head_pos[1]),
                         snake_head_to_body_distances)
        angle_index += 1
        angle += 2 * math.pi / n
    if 0 in snake_head_to_body_distances:
        print(snake_head_to_body_distances)
    return snake_head_to_body_distances


@njit
def has_intersection(angle_index, game_array, ray, snake_head_pos, snake_head_to_body_distances):
    for x, y in ray:
        x, y = int(x), int(y)
        # check if point is in the grid
        if not (0 <= x < game_array.shape[1] and 0 <= y < game_array.shape[0]):
            break

        if game_array[y][x] in [TileTypes.SNAKE_BODY.value, TileTypes.WALL.value]:
            snake_head_to_body_distances[angle_index] = np.sqrt(
                (x - snake_head_pos[0]) ** 2 + (y - snake_head_pos[1]) ** 2)
            break


def get_inputs(game, n_directions=8, start_angle=0, end_angle=2 * math.pi,
               include_wall_distance=False, include_snake_length=False):
    snake = game[2]
    food_pos = game[1]
    game_array = game[0]
    snake_head_pos = pygame.Vector2(snake[-1][1], snake[-1][0])
    food_pos = pygame.Vector2(food_pos[1], food_pos[0])
    snake_head_to_food = food_pos - snake_head_pos
    grid_size = game_array.shape[1]
    # get the distance of the snake head to the 4 walls
    wall_distances = [0, 0, 0, 0]
    for i in range(4):
        if i == 0:
            wall_distances[i] = snake_head_pos.x
        elif i == 1:
            wall_distances[i] = grid_size - snake_head_pos.x
        elif i == 2:
            wall_distances[i] = snake_head_pos.y
        elif i == 3:
            wall_distances[i] = grid_size - snake_head_pos.y
    if not include_wall_distance:
        wall_distances = []

    inputs = [distance for distance in
              get_body_distance_in_n_directions(snake, game_array, n_directions)] + \
             [distance for distance in wall_distances] + \
             [snake_head_to_food.as_polar()[1]] + \
             [snake_head_to_food.length()]
    if include_snake_length:
        inputs.append(len(snake))
    return inputs
