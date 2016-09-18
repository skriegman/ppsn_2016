from copy import deepcopy
import csv
import operator

from deap.tools import HallOfFame

from gp.algorithms import afpo


class HistoricalHallOfFame(HallOfFame):
    def __init__(self, maxsize, similar=operator.eq):
        super(HistoricalHallOfFame, self).__init__(maxsize, similar)
        self.historical_trees = list()

    def update(self, population):
        super(HistoricalHallOfFame, self).update(population)
        best_tree = max(population, key=operator.attrgetter("fitness"))
        self.historical_trees.append(deepcopy(best_tree))


class BestTreeArchive(object):
    def __init__(self, frequency):
        self.frequency = frequency
        self.generation_counter = 0
        self.generations = []
        self.best_trees = []

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            best_tree = min(population, key=operator.attrgetter("error"))
            self.best_trees.append(deepcopy(best_tree))
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        best_tree_file = "best_tree_" + log_file
        with open(best_tree_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, best_tree in zip(self.generations, self.best_trees):
                writer.writerow([gen, best_tree])


class SizeDistributionArchive(object):
    def __init__(self):
        self.size_distribution = list()

    def update(self, population):
        sizes = [len(ind) for ind in population]
        self.size_distribution.append(sizes)


class AgeDistributionArchive(object):
    def __init__(self):
        self.age_distribution = list()

    def update(self, population):
        ages = [ind.age for ind in population]
        self.age_distribution.append(ages)

    def save(self, log_file):
        semantic_statistics_file = "age_" + log_file
        with open(semantic_statistics_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, ages in enumerate(self.age_distribution):
                writer.writerow([gen, ages])


class FitnessDistributionArchive(object):
    def __init__(self, frequency):
        self.fitness = []
        self.generations = []
        self.frequency = frequency
        self.generation_counter = 0

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            fitnesses = [ind.fitness.values for ind in population]
            self.fitness.append(fitnesses)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        fitness_distribution_file = "fitness_" + log_file
        with open(fitness_distribution_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, ages in zip(self.generations, self.fitness):
                writer.writerow([gen, ages])


class MutationDistributionArchive(object):
    def __init__(self):
        self.semantic_distances = []
        self.fitness_changes = []
        self.size_changes = []
        self.ages = []
        self.generations = []
        self.generation_counter = 0

    def update(self, semantic_distance, fitness_change, size_change, age, gen_count):
        self.semantic_distances.append(semantic_distance)
        self.fitness_changes.append(fitness_change)
        self.size_changes.append(size_change)
        self.ages.append(age)
        self.generation_counter += gen_count
        self.generations.append(self.generation_counter)

    def save(self, log_file):
        mutation_distribution_file = "mutation_" + log_file
        with open(mutation_distribution_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["gen", "semantic_distance", "fitness_change", "size_change", "age"])
            for gen, dist, fit, size, age in zip(self.generations, self.semantic_distances,
                                                 self.fitness_changes, self.size_changes, self.ages):
                writer.writerow([gen, dist, fit, size, age])


class MultiArchive(object):
    def __init__(self, archives):
        self.archives = archives

    def update(self, population):
        for archive in self.archives:
            archive.update(population)

    def save(self, log_file):
        for archive in self.archives:
            archive.save(log_file)


def pick_fitness_size_from_fitness_age_size(ind):
    ind.fitness.values = (ind.error, 0, len(ind))


def pick_fitness_size_from_fitness_age(ind):
    ind.fitness.values = (ind.error, len(ind))


class ParetoFrontSavingArchive(object):
    def __init__(self, frequency, criteria_chooser=None, simplifier=None):
        self.fronts = []
        self.frequency = frequency
        self.generation_counter = 0
        self.criteria_chooser = criteria_chooser
        self.simplifier = simplifier

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            pop_copy = [deepcopy(ind) for ind in population]
            if self.simplifier is not None:
                self.simplifier(pop_copy)
            if self.criteria_chooser is not None:
                map(self.criteria_chooser, pop_copy)

            non_dominated = afpo.find_pareto_front(pop_copy)
            front = [pop_copy[index] for index in non_dominated]
            front.sort(key=operator.attrgetter("fitness.values"))
            self.fronts.append(front)
        self.generation_counter += 1

    def save(self, log_file):
        pareto_front_file = "pareto_" + log_file
        with open(pareto_front_file, 'wb') as f:
            writer = csv.writer(f)
            generation = 0
            for front in self.fronts:
                inds = [(ind.fitness.values, str(ind)) for ind in front]
                writer.writerow([generation, len(inds)] + inds)
                generation += self.frequency


class TestSetPerformanceArchive(object):
    def __init__(self, frequency, evaluate_test_error):
        self.evaluate_test_error = evaluate_test_error
        self.test_errors = []
        self.frequency = frequency
        self.generation_counter = 0
        self.generations = []

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            best_tree = min(population, key=operator.attrgetter("error"))
            test_error = self.evaluate_test_error(best_tree)[0]
            self.test_errors.append(test_error)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        test_error_file = "test_error_" + log_file
        with open(test_error_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "test_error"])
            for gen, test_error in zip(self.generations, self.test_errors):
                writer.writerow([gen, test_error])


class PopulationSavingArchive(object):
    def __init__(self, frequency, simplifier=None):
        self.inds = []
        self.generations = []
        self.frequency = frequency
        self.generation_counter = 0
        self.simplifier = simplifier

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            pop_copy = [deepcopy(ind) for ind in population]
            if self.simplifier is not None:
                self.simplifier(pop_copy)
            self.inds.append(pop_copy)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        inds_file = "inds_" + log_file
        with open(inds_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, inds in zip(self.generations, self.inds):
                tuples = [(ind.fitness.values, str(ind)) for ind in inds]
                writer.writerow([gen, len(inds)] + tuples)
