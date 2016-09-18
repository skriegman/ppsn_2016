import logging
import os
import random
import sys
import cachetools
import operator
import numpy
from sklearn import preprocessing
from deap import creator, base, tools, gp

from gp.experiments import runner
from gp.algorithms import afpo, archive
from gp.experiments import symbreg, reports, fast_evaluate
from gp.semantic import simplify, semantics


def get_toolbox(predictors, response):
    creator.create("ErrorAgeSize", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.ErrorAgeSize)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_polynomial_explog_trig_pset(len(predictors[0]))
    pset.addEphemeralConstant("gaussian", lambda: random.gauss(0.0, 1.0))

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom)

    # Crossover
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=300))

    # Mutation
    toolbox.register("expr_mutation", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutation, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=300))

    # Fast evaluation configuration
    numpy_response = numpy.array(response)
    numpy_predictors = numpy.array(predictors)
    expression_dict = cachetools.LRUCache(maxsize=2000)

    # toolbox.register("error_func", fast_evaluate.mean_absolute_percentage_error, response=numpy_response)
    toolbox.register("error_func", fast_evaluate.anti_correlation, response=numpy_response)
    toolbox.register("evaluate_error", fast_evaluate.fast_numpy_evaluate, context=pset.context,
                     predictors=numpy_predictors, error_function=toolbox.error_func, expression_dict=expression_dict)
    toolbox.register("evaluate", afpo.evaluate_age_fitness_size, error_func=toolbox.evaluate_error)

    random_data_points = numpy.random.choice(len(predictors), 1000, replace=False)
    subset_predictors = numpy_predictors[random_data_points, :]

    toolbox.register("calc_semantics", semantics.calculate_semantics, context=pset.context,
                     predictors=subset_predictors)
    toolbox.register("simplify_front", simplify.simplify_all, toolbox=toolbox, size_threshold=0,
                     semantics_threshold=10e-5, precompute_semantics=True)

    pop = toolbox.population(n=1000)
    mstats = reports.configure_inf_protected_stats()
    pareto_archive = archive.ParetoFrontSavingArchive(frequency=1,
                                                      criteria_chooser=archive.pick_fitness_size_from_fitness_age_size,
                                                      simplifier=toolbox.simplify_front)

    toolbox.register("run", afpo.afpo, population=pop, toolbox=toolbox, xover_prob=0.75, mut_prob=0.01, ngen=1000,
                     tournament_size=2, num_randoms=1, stats=mstats, hall_of_fame=pareto_archive)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(pareto_archive))

    return toolbox


if __name__ == "__main__":
    YEAR = int(sys.argv[2])
    resolution = int(sys.argv[3])
    training_data_dir = "../data/SweData/train/"

    #
    # STANDARD: No binning: GP selects from all possible pixels
    # predictors_name = "pixel_level/predictors_pixel_level_leave_out_{}.npy".format(YEAR)
    #
    #
    # FILTERS: upsampling to overlapping circles grid
    predictors_name = "grid/predictors_leave_out_{0}_resolution_{1}.npy".format(YEAR, resolution)
    #
    #

    response_name = "daily_total/response_daily_total_leave_out_{}.npy".format(YEAR)
    predictors_file = os.path.join(training_data_dir, predictors_name)
    response_file = os.path.join(training_data_dir, response_name)
    p, r = numpy.load(predictors_file), numpy.load(response_file)
    scale = preprocessing.StandardScaler()
    p = scale.fit_transform(p)
    # print numpy.sum(numpy.std(p, axis=0))
    r = symbreg.numpy_protected_log_abs(r)

    exp_name = "year_{0}_resolution_{1}".format(YEAR, resolution)
    random_seed = int(sys.argv[1])
    runner.run_data(random_seed, p, r, [get_toolbox], ["afpo"], exp_name, logging_level=logging.INFO)
