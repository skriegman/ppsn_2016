import logging
import os
import random
import sys
import cachetools
import operator
import numpy as np
import glob
import csv
import ast

from deap import creator, base, tools, gp

from gp.experiments import runner
from gp.algorithms import afpo, archive
from gp.experiments import symbreg, reports, fast_evaluate

from gp.features.parametrized_terminals import ParametrizedPrimitiveSet,  RadiusMeanTerminal
from gp.features.parametrized_evaluation import fast_numpy_evaluate_metadata, calculate_semantics, \
    AggregationPrimitiveTree
from gp.features import mutation
from gp.semantic import simplify, semantics


def get_parametrized_pset():
    pset = ParametrizedPrimitiveSet("MAIN", 0)

    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(symbreg.numpy_protected_div_dividend, 2)
    pset.addPrimitive(symbreg.numpy_protected_log_abs, 1)
    pset.addPrimitive(symbreg.numpy_protected_exponential, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addEphemeralConstant("gaussian", lambda: random.gauss(0.0, 1.0))
    return pset


def configure_toolbox(primitive_tree_class, *terminal_classes):
    creator.create("ErrorAgeSize", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", primitive_tree_class, fitness=creator.ErrorAgeSize)

    toolbox = base.Toolbox()
    pset = get_parametrized_pset()
    for terminal in terminal_classes:
        pset.addParametrizedTerminal(terminal)
    toolbox.pset = pset

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    return toolbox


def get_toolbox_base(predictors, response, toolbox, param_mut_prob):
    metadata_dict = dict()
    latitude_longitude = np.load('../data/SweData/metadata/latlon.npy')
    elevation = np.load('../data/SweData/metadata/elevation.npy')
    aspect = np.load('../data/SweData/metadata/aspect.npy')
    metadata_dict["LatLon"] = latitude_longitude
    metadata_dict["Elevation"] = np.repeat(elevation, 3)
    metadata_dict["Aspect"] = np.repeat(aspect, 3)
    metadata_dict["Response"] = response
    predictors_dict = [None, None, None]
    predictors_indices = np.arange(predictors.shape[1])
    predictors_dict[0] = predictors[:, predictors_indices % 3 == 0]
    predictors_dict[1] = predictors[:, predictors_indices % 3 == 1]
    predictors_dict[2] = predictors[:, predictors_indices % 3 == 2]
    metadata_dict["Predictors"] = predictors_dict

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom)

    # Crossover
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=300))

    # Mutation
    toolbox.register("expr_mutation", gp.genFull, min_=0, max_=2)
    toolbox.register("subtree_mutate", gp.mutUniform, expr=toolbox.expr_mutation, pset=toolbox.pset)
    toolbox.decorate("subtree_mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("subtree_mutate", gp.staticLimit(key=len, max_value=300))

    toolbox.register("parameter_mutation", mutation.one_point_parameter_mutation,
                     toolbox=toolbox, metadata=metadata_dict, two_point_scale=0.005, radius_scale=0.25, iterations=20)
    toolbox.register("mutate", mutation.multi_mutation,
                     mutations=[toolbox.subtree_mutate, toolbox.parameter_mutation], probs=[0.05, param_mut_prob])

    # Fast evaluation configuration
    numpy_response = np.array(response)
    numpy_predictors = np.array(predictors)
    expression_dict = cachetools.LRUCache(maxsize=2000)
    toolbox.register("error_func", fast_evaluate.anti_correlation, response=numpy_response)
    toolbox.register("evaluate_error", fast_numpy_evaluate_metadata, context=toolbox.pset.context,
                     predictors=numpy_predictors, metadata=metadata_dict, error_function=toolbox.error_func,
                     expression_dict=expression_dict, arg_prefix="ARG")
    toolbox.register("evaluate", afpo.evaluate_age_fitness_size, error_func=toolbox.evaluate_error)

    random_data_points = np.random.choice(len(predictors), 1000, replace=False)
    subset_predictors = numpy_predictors[random_data_points, :]
    toolbox.register("calc_semantics", calculate_semantics, context=toolbox.pset.context,
                     predictors=subset_predictors, metadata=metadata_dict)
    toolbox.register("simplify_front", simplify.simplify_all, toolbox=toolbox, size_threshold=0,
                     semantics_threshold=10e-5, precompute_semantics=True)

    pop = toolbox.population(n=1000)
    mstats = reports.configure_inf_protected_stats()
    pareto_archive = archive.ParetoFrontSavingArchive(frequency=1,
                                                      criteria_chooser=archive.pick_fitness_size_from_fitness_age_size,
                                                      simplifier=toolbox.simplify_front)

    toolbox.register("run", afpo.afpo, population=pop, toolbox=toolbox, xover_prob=0.75, mut_prob=0.20, ngen=1000,
                     tournament_size=2, num_randoms=1, stats=mstats,
                     mut_archive=None, hall_of_fame=pareto_archive)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(pareto_archive))

    return toolbox


def get_toolbox_circle_aggregation(predictors, response):
    toolbox = configure_toolbox(AggregationPrimitiveTree, RadiusMeanTerminal)
    return get_toolbox_base(predictors, response, toolbox, 0.50)


def get_toolbox_gpesa(predictors, response):
    file_patterns = glob.glob("../data/SweData/hill_climber/*results*_{}_*".format(YEAR))
    centroids =[]
    radiuses = []
    features = []
    for file_name in file_patterns:
        with open(file_name, "rb") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                centroids.append(ast.literal_eval(row[0]))
                radiuses.append(ast.literal_eval(row[1]))
                features.append(ast.literal_eval(row[2]))

    class GPESARadiusMeanTerminal(RadiusMeanTerminal):
        parameters_dict = {"centroids": centroids,
                           "radiuses": radiuses,
                           "features": features}

        def __init__(self):
            RadiusMeanTerminal.__init__(self)
            index = random.choice(range(len(self.parameters_dict["centroids"])))
            self.centroid = self.parameters_dict["centroids"][index]
            self.radius = self.parameters_dict["radiuses"][index]
            self.feature = self.parameters_dict["features"][index]

    toolbox = configure_toolbox(AggregationPrimitiveTree, GPESARadiusMeanTerminal)
    return get_toolbox_base(predictors, response, toolbox, 0.50)


if __name__ == "__main__":
    toolbox_func = get_toolbox_gpesa

    YEAR = int(sys.argv[2])
    training_data_dir = "../data/SweData/train/"
    predictors_name = "pixel_level/predictors_pixel_level_leave_out_{}.npy".format(YEAR)
    response_name = "daily_total/response_daily_total_leave_out_{}.npy".format(YEAR)
    predictors_file = os.path.join(training_data_dir, predictors_name)
    response_file = os.path.join(training_data_dir, response_name)
    p, r = np.load(predictors_file), np.load(response_file)
    r = symbreg.numpy_protected_log_abs(r)

    exp_name = "aggregation_{}".format(YEAR)
    random_seed = int(sys.argv[1])
    runner.run_data(random_seed, p, r, [toolbox_func], ["afpo"], exp_name, logging_level=logging.INFO)
