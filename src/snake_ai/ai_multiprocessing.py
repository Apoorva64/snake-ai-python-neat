import multiprocessing
import os
import neat
import numpy as np
import pygame
import neat_reporters.visu as visualize
from fast_snake.src.fast_snake.fast_snake import generate_game, move_snake
from neat_reporters import ai_reporter, ai_reporter_gui
from snake_ai.inputs import get_inputs

RUNS_PER_GAME_SIZE = 10
ENABLE_GUI = True
# GAME_SIZE_RANGE = range(5, 100, 10)
GAME_SIZE_RANGE = [40]
SNAKE_GAME_SIZE = (40, 40)
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
OPPOSITE_DIRECTIONS = {
    (0, 1): (0, -1),
    (0, -1): (0, 1),
    (1, 0): (-1, 0),
    (-1, 0): (1, 0)
}
FOOD_TIMER_MAX = SNAKE_GAME_SIZE[0] * np.sqrt(2)
NUMBER_OF_RAYS = 4
ROTATING_DIRECTIONS = False
INCLUDE_LAST_DIRECTION = False
INCLUDE_SNAKE_LENGTH = False
INCLUDE_WALL_DISTANCE = True
TEMPORAL_LENGTH = 1
CORE_COUNT = multiprocessing.cpu_count() - 1
CHECKPOINT_FOLDER = \
    f'checkpoints-ai-multiprocessing-{NUMBER_OF_RAYS}-rays-{ROTATING_DIRECTIONS}-rotating-directions-0-hidden-fixed-game-size-40-{INCLUDE_SNAKE_LENGTH}-snake-length-{INCLUDE_WALL_DISTANCE}-wall-distance-{INCLUDE_LAST_DIRECTION}-last-direction-{TEMPORAL_LENGTH}-temporal-length-2'


# CHECKPOINT_FOLDER = f"checkpoints-ai-multiprocessing-game-array"


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness_list = []
    for game_size in GAME_SIZE_RANGE:
        for _ in range(RUNS_PER_GAME_SIZE):
            is_dead = False
            food_timer = 0
            last_distance_to_food = 0
            last_direction = (0, 0)
            fitness = 0
            game = generate_game(game_size=(game_size, game_size))
            temporal_inputs = []
            for _ in range(TEMPORAL_LENGTH):
                temporal_inputs.append(get_inputs(game, n_directions=NUMBER_OF_RAYS,
                                                  include_snake_length=INCLUDE_SNAKE_LENGTH,
                                                  include_wall_distance=INCLUDE_WALL_DISTANCE))
            while not is_dead:
                inputs = get_inputs(game, n_directions=NUMBER_OF_RAYS,
                                    include_snake_length=INCLUDE_SNAKE_LENGTH,
                                    include_wall_distance=INCLUDE_WALL_DISTANCE)

                if INCLUDE_LAST_DIRECTION:
                    inputs += list(last_direction)
                temporal_inputs.append(inputs)
                temporal_inputs.pop(0)
                outputs = net.activate(np.array(temporal_inputs).flatten())
                # invalid_outputs = not is_outputs_valid(outputs)
                # if invalid_outputs:
                #     fitness -= 1000
                direction = DIRECTIONS[outputs.index(max(outputs))]
                if last_direction != (0, 0) and direction == OPPOSITE_DIRECTIONS[last_direction]:
                    fitness -= 10
                last_direction = direction

                # move snake_data
                snake_data, is_dead, food_pos, eaten = move_snake(game[0], game[2], direction, game[1])
                if food_pos == (-1, -1):
                    return fitness
                game = (game[0], food_pos, snake_data)
                food_timer += 1
                if food_timer > FOOD_TIMER_MAX * len(snake_data):
                    is_dead = True
                else:
                    snake_head_pos = pygame.Vector2(snake_data[-1][1], snake_data[-1][0])
                    food_pos = pygame.Vector2(food_pos[1], food_pos[0])
                    snake_head_to_food = food_pos - snake_head_pos
                    distance_to_food = snake_head_to_food.length()
                    if distance_to_food < last_distance_to_food:
                        fitness += 1
                    else:
                        fitness -= 1
                    last_distance_to_food = distance_to_food
                    fitness += 0.1
                    if eaten:
                        fitness += 100
                        food_timer = 0
            fitness_list.append(fitness)
    return sum(fitness_list) / len(fitness_list)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # check if checkpoint folder exists
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # get the latest population in the checkpoints folder
    if os.listdir(CHECKPOINT_FOLDER):
        checkpoints = sorted(os.listdir(CHECKPOINT_FOLDER), key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = checkpoints[-1]
        print("Loading checkpoint: " + latest_checkpoint)
        p = neat.Checkpointer.restore_checkpoint(f"{CHECKPOINT_FOLDER}/" + latest_checkpoint)
    else:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(ai_reporter.GraphsReporter(stats, config, "graphs/" + CHECKPOINT_FOLDER, p))
    p.add_reporter(neat.Checkpointer(100, filename_prefix=f'{CHECKPOINT_FOLDER}/neat-checkpoint-'))
    if ENABLE_GUI:
        p.add_reporter(
            ai_reporter_gui.PygameReporter(config=config, number_of_rays=NUMBER_OF_RAYS, game_size=SNAKE_GAME_SIZE,
                                           stats=stats
                                           , include_snake_length=INCLUDE_SNAKE_LENGTH,
                                           include_wall_distance=INCLUDE_WALL_DISTANCE,
                                           include_last_direction=INCLUDE_LAST_DIRECTION,

                                           ))

    pe = neat.ParallelEvaluator(CORE_COUNT, eval_genome)
    winner = p.run(pe.evaluate, 50000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, True, )
    visualize.draw_net(config, winner, True, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    config_path = "config"
    run(config_path)
