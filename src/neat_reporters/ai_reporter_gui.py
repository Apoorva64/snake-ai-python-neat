import io
import math
import random
from threading import Thread

import cv2
import neat
import numpy as np
from neat.reporting import BaseReporter
import time

from neat.math_util import mean, stdev
from neat.six_util import itervalues

from fast_snake.src.fast_snake.fast_snake import generate_game, move_snake, render
from neat_reporters.visu import draw_net
from snake_ai.inputs import draw_ai_inputs, get_inputs
from neat_reporters import visu as visualize

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
OPPOSITE_DIRECTIONS = {
    (0, 1): (0, -1),
    (0, -1): (0, 1),
    (1, 0): (-1, 0),
    (-1, 0): (1, 0)
}
html = """<font face='veranda' color='#ffffff' size=4>
Generation: {generation}    fps: {fps}<br>
Best fitness: {best_fitness}<br>
Mean fitness: {mean_fitness}<br>
Std fitness: {std_fitness}<br>
Average time per generation: {mean_time}<br>
Total time: {total_time}<br>
Current game size: {game_size}<br>
Current fitness: {current_score}<br>
Current snake length: {current_length}<br>
Food timer: {food_timer}<br>
Outputs: <br>
{outputs}<br>
Direction: {direction}<br>
Inputs:<br>
    - Distance to food: {distance_to_food}<br>
    - Angle to food: {angle_to_food}<br>
    - Distance to walls: {distance_to_wall}<br>
    - Rays:<br>
{rays}
</font>""".replace('\n', '')


class PygameReporter(BaseReporter):
    def __init__(self, config, stats, number_of_rays=4, game_size=(40, 40), include_last_direction=False,
                 include_wall_distance=False, include_snake_length=False, temporal_length=1):
        self.fitness_std = None
        self.fitness_mean = None
        self.fitnesses = None
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.average_time_per_generation = 0
        self.config = config
        self.best_genome = None
        self.thread = Thread(target=self.pygame_thread)
        self.thread.start()
        self.number_of_rays = number_of_rays
        self.game_size = game_size
        self.stats = stats
        self.fps_limit = 20
        self.include_last_direction = include_last_direction
        self.include_wall_distance = include_wall_distance
        self.include_snake_length = include_snake_length
        self.temporal_length = temporal_length

    def start_generation(self, generation):
        self.generation = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        self.average_time_per_generation = sum(self.generation_times) / len(self.generation_times)

    def post_evaluate(self, config, population, species, best_genome):
        self.fitnesses = [c.fitness for c in itervalues(population)]
        self.fitness_mean = mean(self.fitnesses)
        self.fitness_std = stdev(self.fitnesses)
        self.best_genome = best_genome

    def pygame_thread(self):
        from pathlib import Path

        import pygame
        import os

        import pygame_gui
        from pygame_gui.core import UIContainer

        from neat_reporters.utils import aspect_scale

        ressources_path = Path('../../data')
        pygame.init()
        # load sounds
        sounds = {}
        for sound_file in (ressources_path / 'sfx').iterdir():
            sounds[sound_file.stem] = pygame.mixer.Sound(str(sound_file))

        pygame.display.set_caption('Snake ai')
        window_surface = pygame.display.set_mode()

        screen_size = (window_surface.get_width(), window_surface.get_height())

        # Place title at the top of the screen with an underline
        title_font = pygame.font.Font(ressources_path / 'fonts/Raleway-Regular.ttf', 50)
        sub_title_font = pygame.font.Font(ressources_path / 'fonts/Raleway-Regular.ttf', 30)

        title_text = title_font.render('Snake ai', True, (255, 255, 255))
        title_rect = title_text.get_rect()
        title_rect.center = (screen_size[0] / 2, 40)
        underline_rect = pygame.Rect(screen_size[0] / 4, title_rect.bottom + 5, screen_size[0] / 2, 2)

        game_surface = pygame.Surface((screen_size[0] / 2.5, screen_size[0] / 2.5))
        game_surface_rect = game_surface.get_rect()
        game_surface.fill((127, 127, 127))
        game_surface_rect.center = (screen_size[0] / 4, screen_size[1] / 1.8)

        game_surface_title = sub_title_font.render('Game', True, (255, 255, 255))
        game_surface_title_rect = game_surface_title.get_rect()
        game_surface_title_rect.midbottom = game_surface_rect.midtop
        game_surface_title_rect.bottom -= 10

        stats_surface_size = (screen_size[0] / 2.5, screen_size[0] / 2.5)
        stats_surface = pygame.Surface(stats_surface_size)
        stats_surface_rect = stats_surface.get_rect()
        stats_surface.fill((0, 0, 12))
        stats_surface_rect.center = (screen_size[0] - screen_size[0] / 4, screen_size[1] / 1.8)

        stats_title_surface = sub_title_font.render('Stats', True, (255, 255, 255))
        stats_title_rect = stats_title_surface.get_rect()
        stats_title_rect.midbottom = stats_surface_rect.midtop
        stats_title_rect.bottom -= 10

        ui_manager = pygame_gui.UIManager(screen_size)
        ui_manager.add_font_paths("raleway", regular_path=str(ressources_path / 'fonts/Raleway-Regular.ttf'))
        stats_container = UIContainer(
            relative_rect=pygame.Rect(stats_surface_rect.left, stats_surface_rect.top, stats_surface_size[0],
                                      stats_surface_size[1] + stats_surface_rect.top), manager=ui_manager)
        stats_text_box = pygame_gui.elements.UITextBox(
            html_text="",
            relative_rect=pygame.Rect(0, 0, stats_surface_size[0], stats_surface_size[1]),
            manager=ui_manager,
            container=stats_container,
        )
        show_ai_net_graph_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(stats_surface_rect.left + stats_surface_size[0] - 200, stats_surface_rect.top,
                                      200, 50),
            text='Show graphs',
            manager=ui_manager,
            starting_height=50,
        )

        increase_game_size_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(stats_surface_rect.left + stats_surface_size[0] - 200,
                                      stats_surface_rect.top + stats_surface_size[1] - 50, 200, 50),
            text='Increase game size',
            manager=ui_manager,
            starting_height=50,
        )
        decrease_game_size_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(stats_surface_rect.left,
                                      stats_surface_rect.top + stats_surface_size[1] - 50, 200, 50),
            text='Decrease game size',
            manager=ui_manager,
            starting_height=50,
        )

        graphs_window_rect = pygame.Rect(0, 0, screen_size[0] / 1.2, screen_size[1] / 1.2)
        graphs_window_rect.center = screen_size[0] / 2, screen_size[1] / 2
        graphs_window = pygame_gui.elements.UIWindow(
            rect=graphs_window_rect,
            manager=ui_manager,
            window_display_title='Graphs',
        )
        graphs_window.on_close_window_button_pressed = lambda: graphs_window.hide()
        graphs_window_left_button_rect = pygame.Rect(0, 0, 40, 30)

        graphs_window_left_button = pygame_gui.elements.UIButton(
            relative_rect=graphs_window_left_button_rect,
            text="<-",
            manager=ui_manager,
            container=graphs_window,
            starting_height=30,
        )
        graphs_window_right_button_rect = pygame.Rect(graphs_window.get_abs_rect().width - 72, 0, 40, 30)
        graphs_window_right_button = pygame_gui.elements.UIButton(
            relative_rect=graphs_window_right_button_rect,
            text="->",
            manager=ui_manager,
            container=graphs_window,
            starting_height=30,
        )
        # graphs_window.hide()
        graphs = {
            0: {
                'title': 'Ai network graph',
                'surface': pygame.Surface(
                    (graphs_window.get_abs_rect().width / 2, graphs_window.get_abs_rect().height / 5)),
            },
            1: {
                'title': 'Speciation graph',
                'surface': pygame.Surface(
                    (graphs_window.get_abs_rect().width / 9, graphs_window.get_abs_rect().height / 5)),
            },
            2: {
                'title': 'Fitness graph',
                'surface': pygame.Surface(
                    (graphs_window.get_abs_rect().width / 9, graphs_window.get_abs_rect().height / 2)),
            },
        }
        for key, graph in graphs.items():
            graph['surface'].fill((255, 255, 0))

        selected_graph_index = 0
        graphs_window_image_surface = pygame.Surface(
            (graphs_window.get_abs_rect().width, graphs_window.get_abs_rect().height))
        graphs_window.title_bar.set_text('Graphs - {}'.format(graphs[selected_graph_index]['title']))

        clock = pygame.time.Clock()
        is_running = True

        graphs_window_image_surface.fill((0, 0, 0))
        graphs_window_image_surface.blit(
            aspect_scale(graphs[selected_graph_index]['surface'], graphs_window.get_abs_rect().width,
                         graphs_window.get_abs_rect().height), (0, 0))
        graphs_window_image_element = pygame_gui.elements.UIImage(
            relative_rect=pygame.Rect(0, 0, graphs_window.get_abs_rect().width,
                                      graphs_window.get_abs_rect().height),
            manager=ui_manager,
            container=graphs_window,
            image_surface=graphs_window_image_surface,
        )
        graphs_window.hide()

        game = generate_game(game_size=self.game_size)
        last_direction = (0, 0)
        fitness = 0
        food_timer = 0
        last_distance_to_food = 0
        start_time = time.time()
        draw_inputs = False
        loading_tick = 0

        intro_video = cv2.VideoCapture(str(ressources_path / 'videos/intro.mp4'))
        success, video_image = intro_video.read()
        fps = intro_video.get(cv2.CAP_PROP_FPS)
        temporal_inputs = []
        for _ in range(self.temporal_length):
            temporal_inputs.append(get_inputs(game, n_directions=self.number_of_rays,
                                              include_snake_length=self.include_snake_length,
                                              include_wall_distance=self.include_wall_distance))
        while is_running:
            if success:
                clock.tick(fps)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False

                success, video_image = intro_video.read()
                if success:
                    video_surf = pygame.image.frombuffer(
                        video_image.tobytes(), video_image.shape[1::-1], "BGR")
                    video_surf = pygame.transform.smoothscale(video_surf, screen_size, )
                    window_surface.blit(video_surf, (0, 0))
                    pygame.display.flip()
                    continue
            window_surface.fill((0, 0, 0))
            time_delta = clock.tick(self.fps_limit) / 1000.0
            inputs = get_inputs(game, n_directions=self.number_of_rays, include_wall_distance=self.include_wall_distance
                                , include_snake_length=self.include_snake_length)
            if self.include_last_direction:
                inputs += list(last_direction)
            temporal_inputs.append(inputs)
            temporal_inputs.pop(0)
            # activate the network
            if self.best_genome is not None:
                outputs = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config).activate(
                    np.array(temporal_inputs).flatten())
                # get the direction
                direction = DIRECTIONS[outputs.index(max(outputs))]
                if last_direction != (0, 0) and direction == OPPOSITE_DIRECTIONS[last_direction]:
                    fitness -= 10
                if last_direction != direction:
                    sounds[['up', 'down', 'left', 'right'][outputs.index(max(outputs))]].play()
                last_direction = direction
                food_timer += 1
                # move snake_data
                snake_data, is_dead, food_pos, eaten = move_snake(game[0], game[2], direction, game[1])
                game = (game[0], food_pos, snake_data)
                # render the game_array
                game_surface.fill((0, 0, 0))
                render(game_surface, game[0])
                if snake_data is None:
                    snake_data = game[2]
                snake_head_pos = pygame.Vector2(snake_data[-1][1], snake_data[-1][0])
                if draw_inputs:
                    draw_ai_inputs(game_surface, game[0], game[1], snake_head_pos, inputs,
                                   n_directions=self.number_of_rays)
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
                    fitness += 70
                    food_timer = 0
                    sounds['eat'].play()

                if is_dead:
                    sounds['dead'].play()
                    fitness = 0
                    game = generate_game(game_size=self.game_size)
                    food_timer = 0
                    temporal_inputs = []
                    for _ in range(self.temporal_length):
                        temporal_inputs.append(get_inputs(game, n_directions=self.number_of_rays,
                                                          include_snake_length=self.include_snake_length,
                                                          include_wall_distance=self.include_wall_distance))
                stats_text_box.set_text(html.format(generation=self.generation,
                                                    best_fitness=self.best_genome.fitness if self.best_genome else 0,
                                                    std_fitness=self.fitness_std,
                                                    mean_fitness=self.fitness_mean,
                                                    total_time=time.time() - start_time,
                                                    current_score=fitness,
                                                    current_length=len(snake_data),
                                                    food_timer=food_timer,
                                                    outputs="<br>".join(["  - {}: {}".format(i, o) for i, o in
                                                                         enumerate(outputs)]),
                                                    direction=direction,
                                                    distance_to_food=distance_to_food,
                                                    angle_to_food=math.degrees(
                                                        math.atan2(snake_head_to_food.y, snake_head_to_food.x)),

                                                    distance_to_wall=inputs[
                                                                     self.number_of_rays:self.number_of_rays + 4],
                                                    rays=", ".join([str(round(r)) for i, r in
                                                                    enumerate(inputs[:self.number_of_rays])]),
                                                    mean_time=self.average_time_per_generation,
                                                    game_size=self.game_size,
                                                    fps=round(clock.get_fps()),
                                                    )
                                        )
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False
                    if event.type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == show_ai_net_graph_button:
                            dot = draw_net(genome=self.best_genome, config=self.config)
                            # render image to surface
                            buffer = io.BytesIO()
                            buffer.write(dot.pipe(format='png'))
                            buffer.seek(0)
                            graphs[0]['surface'] = pygame.image.load(buffer, 'png')
                            buffer.close()
                            buffer = io.BytesIO()
                            visualize.plot_stats(self.stats, ylog=False, view=False).savefig(buffer, format='png')
                            buffer.seek(0)
                            graphs[1]['surface'] = pygame.image.load(buffer, 'png')
                            buffer.close()
                            buffer = io.BytesIO()
                            visualize.plot_species(self.stats, view=False).savefig(buffer, format='png')
                            buffer.seek(0)
                            graphs[2]['surface'] = pygame.image.load(buffer, 'png')
                            buffer.close()
                            graphs_window_image_element.kill()
                            graphs_window_image_surface.fill((0, 0, 0))
                            graphs_window_image_surface.blit(
                                aspect_scale(graphs[selected_graph_index]['surface'],
                                             graphs_window.get_abs_rect().width,
                                             graphs_window.get_abs_rect().height), (0, 0))
                            graphs_window_image_element = pygame_gui.elements.UIImage(
                                relative_rect=pygame.Rect(0, 0, graphs_window.get_abs_rect().width,
                                                          graphs_window.get_abs_rect().height),
                                manager=ui_manager,
                                container=graphs_window,
                                image_surface=graphs_window_image_surface,
                            )
                            graphs_window.show()
                        if event.ui_element == graphs_window_left_button:
                            selected_graph_index -= 1
                            if selected_graph_index < 0:
                                selected_graph_index = len(graphs) - 1
                            graphs_window.title_bar.set_text(
                                'Graphs - {}'.format(graphs[selected_graph_index]['title']))
                            graphs_window_image_element.kill()
                            graphs_window_image_surface.fill((0, 0, 0))
                            graphs_window_image_surface.blit(
                                aspect_scale(graphs[selected_graph_index]['surface'],
                                             graphs_window.get_abs_rect().width,
                                             graphs_window.get_abs_rect().height), (0, 0))
                            graphs_window_image_element = pygame_gui.elements.UIImage(
                                relative_rect=pygame.Rect(0, 0, graphs_window.get_abs_rect().width,
                                                          graphs_window.get_abs_rect().height),
                                manager=ui_manager,
                                container=graphs_window,
                                image_surface=graphs_window_image_surface,
                            )
                        if event.ui_element == graphs_window_right_button:
                            selected_graph_index += 1
                            if selected_graph_index >= len(graphs):
                                selected_graph_index = 0
                            graphs_window.title_bar.set_text(
                                'Graphs - {}'.format(graphs[selected_graph_index]['title']))
                            graphs_window_image_element.kill()
                            graphs_window_image_surface.fill((0, 0, 0))
                            graphs_window_image_surface.blit(
                                aspect_scale(graphs[selected_graph_index]['surface'],
                                             graphs_window.get_abs_rect().width,
                                             graphs_window.get_abs_rect().height), (0, 0))
                            graphs_window_image_element = pygame_gui.elements.UIImage(
                                relative_rect=pygame.Rect(0, 0, graphs_window.get_abs_rect().width,
                                                          graphs_window.get_abs_rect().height),
                                manager=ui_manager,
                                container=graphs_window,
                                image_surface=graphs_window_image_surface,
                            )
                        if event.ui_element == increase_game_size_button:
                            self.game_size = (self.game_size[0] + 5, self.game_size[1] + 5)
                            game = generate_game(game_size=self.game_size)
                            food_timer = 0
                        if event.ui_element == decrease_game_size_button:
                            if self.game_size[0] > 5 and self.game_size[1] > 5:
                                self.game_size = (self.game_size[0] - 5, self.game_size[1] - 5)
                                game = generate_game(game_size=self.game_size)
                                food_timer = 0
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_h:
                            draw_inputs = not draw_inputs
                        if event.key == pygame.K_p:
                            self.fps_limit += 1
                        if event.key == pygame.K_o:
                            self.fps_limit -= 1
                    ui_manager.process_events(event)

                window_surface.fill((0, 0, 0))
                window_surface.blit(title_text, title_rect)
                window_surface.blit(game_surface, game_surface_rect)
                window_surface.blit(stats_title_surface, stats_title_rect)
                window_surface.blit(game_surface_title, game_surface_title_rect)
                pygame.draw.rect(window_surface, (255, 255, 255), underline_rect, 1)
                ui_manager.update(time_delta)
                ui_manager.draw_ui(window_surface)
                pygame.display.update()
                snake_length = len(snake_data)
            else:
                loading_tick += 1
                # blit loading animation on screen
                window_surface.fill((0, 0, 0))
                window_surface.blit(title_text, title_rect)
                pygame.draw.rect(window_surface, (255, 255, 255), underline_rect, 1)

                loading_text = pygame.font.SysFont('Arial', 20).render('Loading' + ('.' * int(loading_tick / 4)), True,
                                                                       (255, 255, 255))
                if loading_tick == 18:
                    loading_tick = 0
                window_surface.blit(loading_text, (window_surface.get_width() / 2 - loading_text.get_width() / 2,
                                                   window_surface.get_height() / 2 - loading_text.get_height() / 2))
                pygame.event.get()
                pygame.display.update()
