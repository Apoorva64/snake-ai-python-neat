import neat_reporters.visu as visualize
from neat.reporting import BaseReporter


class GraphsReporter(BaseReporter):
    stats = None
    config = None

    def __init__(self, stats, config, folder, population):
        self.folder = folder
        self.stats = stats
        self.config = config
        self.population = population

    def post_evaluate(self, config, population, species, best_genome):
        if self.population.generation % 200 == 0:
            visualize.draw_net(config, best_genome, False,
                               filename=f'{self.folder}/best_net_{self.population.generation}',
                               )
            visualize.plot_stats(self.stats, ylog=False, view=True)
            visualize.plot_species(self.stats, view=True)
