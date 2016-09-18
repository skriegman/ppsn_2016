import logging
import random

from deap import tools
import numpy
import time


def breed(parents, toolbox, xover_prob, mut_prob):
    offspring = [toolbox.clone(ind) for ind in parents]

    for i in range(1, len(offspring), 2):
        if random.random() < xover_prob:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            max_age = max(offspring[i - 1].age, offspring[i].age)
            offspring[i].age = offspring[i - 1].age = max_age
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mut_prob:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def find_AF_pareto_front(population):
    unique_ages = dict()
    for ind in population:
        if ind.age in unique_ages:
            if unique_ages[ind.age].fitness.values[0] < ind.fitness.values[0]:
                unique_ages[ind.age] = ind
            elif unique_ages[ind.age].fitness.values[0] == ind.fitness.values[0]:
                unique_ages[ind.age] = random.choice((ind, unique_ages[ind.age]))
        else:
            unique_ages[ind.age] = ind

    front = []
    best_fitness = 0.0
    for age in sorted(unique_ages):
        if unique_ages[age].fitness.values[0] > best_fitness:
            best_fitness = unique_ages[age].fitness.values[0]
            front.append(unique_ages[age])

    return front


def list_dominates_errors(list1, list2):
    not_equal = False
    for list1_value, list2_value in zip(list1, list2):
        if list1_value < list2_value:
            not_equal = True
        elif list1_value > list2_value:
            return False
    return not_equal


def find_pareto_front_fitnesses(fitnesses):
    pareto_front = set(range(len(fitnesses)))

    for i in range(len(fitnesses)):
        if i not in pareto_front:
            continue

        fitness1 = fitnesses[i]
        for j in range(i + 1, len(fitnesses)):
            fitness2 = fitnesses[j]

            if fitness2.dominates(fitness1) or fitness1 == fitness2:
                pareto_front.discard(i)
            if fitness1.dominates(fitness2):
                pareto_front.discard(j)

    return pareto_front


def find_pareto_front(population):
    """Finds a subset of nondominated indivuals in a given list

    :param population: a list of individuals
    :return: a set of indices corresponding to nondominated individuals
    """

    pareto_front = set(range(len(population)))

    for i in range(len(population)):
        if i not in pareto_front:
            continue

        ind1 = population[i]
        for j in range(i + 1, len(population)):
            ind2 = population[j]

            # if individuals are equal on all objectives, mark one of them (the first encountered one) as dominated
            # to prevent excessive growth of the Pareto front
            if ind2.fitness.dominates(ind1.fitness) or ind1.fitness == ind2.fitness:
                pareto_front.discard(i)

            if ind1.fitness.dominates(ind2.fitness):
                pareto_front.discard(j)

    return pareto_front


def reduce_population(population, tournament_size, target_popsize, nondominated_size):
    num_iterations = 0
    new_population_indices = list(range(len(population)))
    while len(new_population_indices) > target_popsize and len(new_population_indices) > nondominated_size:
        if num_iterations > 10e6:
            logging.info("Pareto front size may be exceeding the size of population! Stopping the execution!")
            # random.sample(new_population_indices, len(new_population_indices) - target_popsize)
            exit()
        num_iterations += 1
        tournament_indices = random.sample(new_population_indices, tournament_size)
        tournament = [population[index] for index in tournament_indices]
        nondominated_tournament = find_pareto_front(tournament)
        for i in range(len(tournament)):
            if i not in nondominated_tournament:
                new_population_indices.remove(tournament_indices[i])
    population[:] = [population[i] for i in new_population_indices]


def reduce_population_pairwise(population, target_popsize, nondominated_size, deterministic=True):
    while len(population) > target_popsize and len(population) > nondominated_size:
        inds = random.sample(range(len(population)), 2)
        ind0 = population[inds[0]]
        ind1 = population[inds[1]]

        if ind0.fitness.values == ind1.fitness.values:
            if deterministic:
                population.pop(inds[0])
            else:
                population.pop(random.choice(inds))
        elif ind0.fitness.dominates(ind1.fitness):
            population.pop(inds[1])
        elif ind1.fitness.dominates(ind0.fitness):
            population.pop(inds[0])


def evaluate_age_fitness(ind, error_func):
    ind.error = error_func(ind)[0]
    return ind.error, ind.age


def evaluate_age_fitness_size(ind, error_func):
    ind.size = len(ind)
    return evaluate_age_fitness(ind, error_func) + (ind.size,)


def evaluate_fitness_size(ind, error_func):
    ind.error = error_func(ind)[0]
    ind.size = len(ind)
    return ind.error, ind.size


def afpo(population, toolbox, xover_prob, mut_prob, ngen, tournament_size, num_randoms=1, hall_of_fame=None,
         stats=None, reduce_pairwise=False):
    target_popsize = len(population)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for ind in population:
        ind.age = 0
        objectives = toolbox.evaluate(ind)
        ind.fitness.values = objectives

    with numpy.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
        record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)

    if hall_of_fame is not None:
        hall_of_fame.update(population)

    for gen in range(1, ngen + 1):
        # extend the population by adding offspring
        parents = toolbox.select(population, len(population) - num_randoms)
        offspring = breed(parents, toolbox, xover_prob, mut_prob)
        population.extend(offspring)

        # add a few random individuals with age of 0
        new_random_inds = [toolbox.individual() for _ in range(num_randoms)]
        for new_ind in new_random_inds:
            new_ind.age = 0
        population.extend(new_random_inds)

        # increase the age and evaluate the individuals
        for ind in population:
            ind.age += 1
            objectives = toolbox.evaluate(ind)
            ind.fitness.values = objectives

        # calculate the size of the global pareto front
        nondominated = find_pareto_front(population)
        logging.debug("Generation: %5d - Pareto Front Size: %5d", gen, len(nondominated))

        # perform Pareto tournament selection until the size of the population is reduced to the target value
        if reduce_pairwise:
            reduce_population_pairwise(population, target_popsize, len(nondominated))
        else:
            reduce_population(population, tournament_size, target_popsize, len(nondominated))

        # Append the current generation statistics to the logbook
        with numpy.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if hall_of_fame is not None:
            hall_of_fame.update(population)

    return population, logbook


def assign_pure_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error,)


def assign_age_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age)


def assign_age_fitness_size(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age, len(ind))


def assign_size_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error, len(ind))


def pareto_optimization(population, toolbox, xover_prob, mut_prob, ngen, tournament_size, num_randoms=1, archive=None,
                        stats=None, calc_pareto_front=True, verbose=False, reevaluate_population=False):
    start = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'cpu_time'] + (stats.fields if stats else [])

    target_popsize = len(population)
    for ind in population:
        ind.error = toolbox.evaluate_error(ind)[0]
    toolbox.assign_fitness(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), cpu_time=time.time() - start, **record)
    if archive is not None:
        archive.update(population)
    if verbose:
        print logbook.stream

    for gen in range(1, ngen + 1):
        if reevaluate_population:
            for ind in population:
                ind.error = toolbox.evaluate_error(ind)[0]

        parents = toolbox.select(population, len(population) - num_randoms)
        offspring = breed(parents, toolbox, xover_prob, mut_prob)
        offspring += [toolbox.individual() for _ in range(num_randoms)]
        for ind in offspring:
            ind.error = toolbox.evaluate_error(ind)[0]

        population.extend(offspring)
        toolbox.assign_fitness(population)

        if calc_pareto_front:
            pareto_front_size = len(find_pareto_front(population))
            logging.debug("Generation: %5d - Pareto Front Size: %5d", gen, pareto_front_size)
            if pareto_front_size > target_popsize:
                logging.info("Pareto front size exceeds the size of population")
                break
        else:
            pareto_front_size = 0

        reduce_population(population, tournament_size, target_popsize, pareto_front_size)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), cpu_time=time.time() - start, **record)
        if archive is not None:
            archive.update(population)
        if verbose:
            print logbook.stream

        for ind in population:
            ind.age += 1

    return population, logbook
