import numpy as np
import csv
import glob
import itertools
from collections import deque
import re
from sklearn import preprocessing, linear_model
from deap import base, tools, creator, gp
import cachetools
from functools import partial
from gather_hill_climber_results import earth_to_square

from gp.features import importance
from gp.algorithms import afpo
from gp.experiments import symbreg, fast_evaluate
from gp.semantic import semantics
from gp.features.parametrized_terminals import ParametrizedPrimitiveSet, ParametrizedTerminal, RadiusMeanTerminal
from gp.features.parametrized_evaluation import AggregationPrimitiveTree
from gp.features import parametrized_evaluation


def pipeline(year, predictor_level, response_level, scale=False):
    if predictor_level != "pixel_level":
        predictors_train = glob.glob("./train/grid/predictors*_{1}_*{0}.npy".format(predictor_level, year))
        response_train = glob.glob("./train/{0}/response*{1}.npy".format(response_level, year))
        predictors_test = glob.glob("./test/grid/predictors*_{1}_*{0}.npy".format(predictor_level, year))
        response_test = glob.glob("./test/{0}/response*{1}.npy".format(response_level, year))
    else:
        predictors_train = glob.glob("./train/{0}/predictors*{1}.npy".format(predictor_level, year))
        response_train = glob.glob("./train/{0}/response*{1}.npy".format(response_level, year))
        predictors_test = glob.glob("./test/{0}/predictors*{1}.npy".format(predictor_level, year))
        response_test = glob.glob("./test/{0}/response*{1}.npy".format(response_level, year))
    p_train, r_train = np.load(predictors_train[0]), np.load(response_train[0])
    p_test, r_test = np.load(predictors_test[0]), np.load(response_test[0])

    r_train, r_test = symbreg.numpy_protected_log_abs(r_train), symbreg.numpy_protected_log_abs(r_test)

    if scale:
        scale = preprocessing.StandardScaler()
        p_train = scale.fit_transform(p_train)
        p_test = scale.transform(p_test)
    return p_train, r_train, p_test, r_test


def fast_numpy_evaluate_metadata(ind, context, train_predictors, test_predictors,
                                 metadata, error_function=None, expression_dict=None, arg_prefix="ARG"):
    semantics_stack = []
    expressions_stack = []
    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=2000)
    for node in reversed(ind):
        expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
        subtree_semantics = [semantics_stack.pop() for _ in range(node.arity)]
        if expression in expression_dict:
            vector = expression_dict[expression]
        else:
            vector = get_node_semantics(node, subtree_semantics, train_predictors, test_predictors, context, metadata, ind, arg_prefix)
            expression_dict[expression] = vector
        expressions_stack.append(expression)
        semantics_stack.append(vector)
    if error_function is None:
        return semantics_stack.pop()
    else:
        return error_function(semantics_stack.pop())


def get_node_semantics(node, subtree_semantics, train_predictors, test_predictors, context, metadata, ind, arg_prefix="ARG"):

    if isinstance(node, ParametrizedTerminal):
        # this is the main difference in validation: we use the scaling from training

        train_vector = node.get_validation_input_vector(train_predictors, metadata, ind, terminal_dict=None)
        test_vector = node.get_validation_input_vector(test_predictors, metadata, ind, terminal_dict=None)
        vector = symbreg.numpy_protected_div_zero(test_vector - np.mean(train_vector), np.std(train_vector))

    elif isinstance(node, gp.Terminal):
        vector = semantics.get_terminal_semantics(node, context, test_predictors, arg_prefix)
    else:
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vector = context[node.name](*subtree_semantics)
    return vector


def get_parametrized_pset():
    pset = ParametrizedPrimitiveSet("MAIN", 0)
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(symbreg.numpy_protected_div_dividend, 2)
    pset.addPrimitive(symbreg.numpy_protected_log_abs, 1)
    pset.addPrimitive(symbreg.numpy_protected_exponential, 1)
    # pset.addPrimitive(np.exp, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    return pset


def get_online_validation_toolbox(train_predictors, test_predictors, train_response, test_response,
                                  primitive_tree_class, *terminal_classes):
    creator.create("ErrorSize", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", primitive_tree_class, fitness=creator.ErrorSize)

    toolbox = base.Toolbox()
    pset = get_parametrized_pset()
    for terminal in terminal_classes:
        pset.addParametrizedTerminal(terminal)
    toolbox.pset = pset

    metadata_dict = dict()
    latitude_longitude = np.load('./metadata/latlon.npy')
    elevation = np.load('./metadata/elevation.npy')
    aspect = np.load('./metadata/aspect.npy')
    metadata_dict["LatLon"] = latitude_longitude
    metadata_dict["Elevation"] = np.repeat(elevation, 3)
    metadata_dict["Aspect"] = np.repeat(aspect, 3)

    toolbox.register("validate_error", linear_model_from_semantics, context=toolbox.pset.context,
                     evaluation_func=fast_numpy_evaluate_metadata,
                     train_predictors=train_predictors, test_predictors=test_predictors,
                     train_response=train_response, test_response=test_response,
                     metadata=metadata_dict)
    toolbox.register("validate", afpo.evaluate_fitness_size, error_func=toolbox.validate_error)
    return toolbox


def linear_model_from_semantics(ind, context, evaluation_func,
                                train_predictors, test_predictors,
                                train_response, test_response, metadata=None):
    if metadata:
        training_semantics = evaluation_func(ind, context, train_predictors, train_predictors,  # scaling factor
                                             metadata, error_function=None)
        testing_semantics = evaluation_func(ind, context, train_predictors, test_predictors,
                                            metadata, error_function=None)
    else:
        training_semantics = evaluation_func(ind, context, train_predictors, error_function=None)
        testing_semantics = evaluation_func(ind, context, test_predictors, error_function=None)
    training_semantics = training_semantics.reshape(-1, 1)

    lr = linear_model.LinearRegression()
    model = lr.fit(training_semantics, train_response)

    testing_semantics = testing_semantics.reshape(-1, 1)
    error = fast_evaluate.mean_absolute_error(model.predict(testing_semantics), test_response)
    return error


def get_offline_validation_toolbox(ptrain, predictors, rtrain, response):
    toolbox = base.Toolbox()
    toolbox.pset = symbreg.get_numpy_pset(len(predictors[0]))
    creator.create("ErrorSize", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.ErrorSize)
    # toolbox.register("validate_func", fast_evaluate.correlation, response=response)
    # toolbox.register("validate_error", fast_evaluate.fast_numpy_evaluate, context=toolbox.pset.context,
    #                  predictors=predictors, error_function=toolbox.validate_func, expression_dict=None)
    toolbox.register("validate_error", linear_model_from_semantics, context=toolbox.pset.context,
                     evaluation_func=fast_evaluate.fast_numpy_evaluate,
                     train_predictors=ptrain, test_predictors=predictors,
                     train_response=rtrain, test_response=response)
    toolbox.register("validate", afpo.evaluate_fitness_size, error_func=toolbox.validate_error)

    return toolbox


def get_last_pareto_front(pareto_file):
    with open(pareto_file) as f:
        line = f.readlines()[-1]
        front_string = ",".join(line.split(",")[2:])
        inds = eval(front_string)
        return [eval(i) for i in inds]


def get_all_pareto_optimal_inds(pareto_files, toolbox, primitive_tree_class, nparams):
    all_inds = []
    all_training = []
    for pareto_file in pareto_files:
        last_front = get_last_pareto_front(pareto_file)
        for ind_tuple in last_front:
            tree_string = ind_tuple[1]
            all_training.append(ind_tuple[0][0])

            if nparams > 0:
                pareto_ind = primitive_tree_class.from_string(tree_string, toolbox.pset, nparams)
            else:
                pareto_ind = gp.PrimitiveTree.from_string(tree_string, toolbox.pset)

            pareto_ind = creator.Individual(pareto_ind)
            pareto_ind.fitness.values = toolbox.validate(pareto_ind)
            all_inds.append(pareto_ind)
    return all_inds, all_training


def save_testing_results(model_type, inds, test_results, name):
    testing_results_file = "./pareto/validation/{0}/validation_{1}.csv".format(model_type, name)
    with open(testing_results_file, 'wb') as f:
        writer = csv.writer(f)
        for i, ind in enumerate(inds):
            rounded_results = "{0:.6f}".format(test_results[i])
            writer.writerow(
                [int(ind.fitness.values[1]), "{0:.6f}".format(ind.fitness.values[0]), rounded_results, str(ind)])


def read_front(file_name):
    sizes, trainning_errors, validation_errors, equations = [], [], [], []
    with open(file_name) as f:
        reader = csv.reader(f)
        if len(sizes) > 0:
            reader.next()
        for row in reader:
            sizes.append(int(row[0]))
            trainning_errors.append(float(row[1]))
            validation_errors.append(float(row[2]))
            equations.append(row[3])

        return sizes, trainning_errors, validation_errors, equations


def save_validation_error_by_run(model_type, year, exp_num, nparams, primitive_tree_class, *terminal_classes):

    # get last pareto front from each run for particular year
    files = glob.glob("./pareto/{0}/*{1}*/pareto*".format(model_type, year, exp_num))

    if nparams > 0:
        p_train, r_train, p_test, r_test = pipeline(year, predictor_level="pixel_level",
                                                    response_level="daily_total", scale=False)
        tool = get_online_validation_toolbox(p_train, p_test, r_train, r_test, primitive_tree_class, *terminal_classes)
    else:
        # when we are using standard GP, we can just standardize the columns a priori based on training scaling
        p_train, r_train, p_test, r_test = pipeline(year, predictor_level=exp_num,
                                                    response_level="daily_total", scale=True)
        tool = get_offline_validation_toolbox(p_train, p_test, r_train, r_test)
    count = 0
    for f in files:
        try:
            iterable_f = [f]
            front, train = get_all_pareto_optimal_inds(iterable_f, tool, primitive_tree_class, nparams)

            test = [ind.fitness.values[0] for ind in front]
            sorted_index = [idx for (error, idx)
                            in sorted(zip(train, range(len(train))))]
            sorted_test = [test[idx] for idx in sorted_index]
            sorted_train = [train[idx] for idx in sorted_index]
            sorted_front = [front[idx] for idx in sorted_index]

            for n, ind in enumerate(sorted_front):
                # train, test
                ind.fitness.values = (sorted_train[n], ind.fitness.values[1])

            non_dominated_index = afpo.find_pareto_front(sorted_front)
            aggregated_sorted_front = [sorted_front[i] for i in non_dominated_index]
            aggregated_sorted_test = [sorted_test[i] for i in non_dominated_index]

            if len(front) > 0:
                # name = "grid_{1}_{0}_run_{2}".format(exp_num, year, count)
                name = "{0}_{1}_{2}".format(exp_num, year, count)
                save_testing_results(model_type, aggregated_sorted_front, aggregated_sorted_test, name)
                count += 1
            else:
                print f
        except SyntaxError:
            print f
            pass


def get_gp_filter_location_relevance(resolution):

    file_patterns = glob.glob("/Users/mecl/gp_mecl/data/SweData/pareto/validation/importance/"
                              "change_in_semantics_filter_{}.csv".format(resolution))
    terminals, change_in_semantics = [], []
    for filename in file_patterns:
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                terminals.append(row[0])
                change_in_semantics.append(abs(float(row[1])))

    log_change_in_semantics = np.array([np.log(u) if (np.isfinite(u) and u != 0.0) else 0.0
                                        for u in change_in_semantics])

    # # rank based approach
    # nonzero_changes = log_change_in_semantics[log_change_in_semantics != 0.0]
    # sorted_index = nonzero_changes.argsort()
    # ranks = np.empty(len(nonzero_changes), int)
    # ranks[sorted_index] = np.arange(len(nonzero_changes))
    # log_change_in_semantics[log_change_in_semantics != 0.0] = ranks + 1  # 0 is blank

    terminal_dict = {}
    for term, change in zip(terminals, log_change_in_semantics):
        terminal_dict[int(term[3:])] = abs(change)

    index = range(resolution ** 2) * 3  # three features with the same circle
    max_importance = [0] * resolution ** 2
    for key, val in terminal_dict.iteritems():
        filter_index = index[key]
        # if val > max_importance[filter_index]:
        #     max_importance[filter_index] = val
        max_importance[filter_index] += val / 3.0

    return max_importance


def get_gpesa_location_relevance(ax, nonzero_frequency_mask):

    file_patterns = glob.glob("/Users/mecl/gp_mecl/data/SweData/pareto/validation/importance/"
                              "change_in_semantics_embedded.csv")
    terminals, change_in_semantics = [], []
    for filename in file_patterns:
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                terminals.extend(terminals_from_string(row[0], 4))
                try:
                    change_in_semantics.append(abs(float(row[1])))
                except ValueError:  # '--' from masked array
                    change_in_semantics.append(0.0)

    xy = []
    r = []
    w = []
    f = []
    for change, term in zip(change_in_semantics, terminals):
        xy.append(term.centroid)
        r.append(term.radius / 10.)
        w.append(change)
        f.append(term.feature)

    log_w = np.array([np.log(u) if (np.isfinite(u) and u != 0.0) else 0.0 for u in w])

    # # rank based approach
    # nonzero_changes = log_w[log_w != 0.0]
    # sorted_index = nonzero_changes.argsort()
    # ranks = np.empty(len(nonzero_changes), int)
    # ranks[sorted_index] = np.arange(len(nonzero_changes))
    # log_w[log_w != 0.0] = ranks + 1

    return earth_to_square(xy, r, log_w, f, nonzero_frequency_mask)


def terminals_from_string(string, nparams):

        tokens = re.split("[ \t\n\r\f\v()\[\],]", string.replace('array', ','))
        expr = []

        def consume(iterator, n):
            deque(itertools.islice(iterator, n), maxlen=0)

        iterator = range(len(tokens)).__iter__()

        for i in iterator:
            token = tokens[i]
            if 'Radius' in token:
                x = eval(token)
                parameters = []
                num_params = nparams
                count = 0
                while len(parameters) < num_params:
                    if tokens[i+1] not in ['', ',', '']:
                        parameters.append(float(tokens[i + 1]))
                    i += 1
                    count += 1
                y = x()
                y.set_params(*parameters)
                expr.append(y)
                consume(iterator, count)
        return expr


def euclidean_distance(a, b, axis=0):
    diff = a - b
    squares = np.multiply(diff, diff)
    return np.sqrt(np.sum(squares, axis=axis))


def save_changes_in_semantics(resolution):
    change_in_semantics_dict = {}
    for year in range(2003, 2012):
        file_patterns = glob.glob("./pareto/validation/filter/validation_resolution_{0}_{1}_*.csv".format(resolution,
                                                                                                          year))
        p_train, r_train, p_test, r_test = pipeline(year, predictor_level="resolution_{}".format(resolution),
                                                    response_level="daily_total", scale=True)
        toolbox = get_offline_validation_toolbox(p_train, p_test, r_train, r_test)
        feature_names = ["ARG{}".format(n) for n in range(p_train.shape[1])]

        for filename in file_patterns:
            _, _, _, equations = read_front(filename)
            for equation in equations:
                tree = gp.PrimitiveTree.from_string(equation, toolbox.pset)
                original_semantics = fast_evaluate.fast_numpy_evaluate(tree, toolbox.pset.context, p_train)

                for key in feature_names:
                    modified_semantics = importance.neutralize_feature(tree, key, toolbox.pset, p_train,
                                                                       fast_evaluate.fast_numpy_evaluate)
                    change_in_semantics = euclidean_distance(modified_semantics, original_semantics)
                    if key not in change_in_semantics_dict:
                        change_in_semantics_dict[key] = [change_in_semantics]
                    else:
                        change_in_semantics_dict[key].append(change_in_semantics)

    change_in_semantics_file = "./pareto/validation/importance/change_in_semantics_filter_{0}.csv".format(resolution)
    with open(change_in_semantics_file, 'wb') as f:
        writer = csv.writer(f)
        for key in change_in_semantics_dict:
            masked_array = np.ma.masked_invalid(change_in_semantics_dict[key])
            count = len(masked_array) - np.ma.count_masked(masked_array)
            change = np.sum(masked_array)
            writer.writerow([key, change / count])


def save_changes_in_embedded_semantics():
    metadata_dict = dict()
    latitude_longitude = np.load('./metadata/latlon.npy')
    metadata_dict["LatLon"] = latitude_longitude
    semantics_func = partial(parametrized_evaluation.fast_numpy_evaluate_metadata, metadata=metadata_dict)

    change_in_semantics_dict = {}
    for year in range(2003, 2012):
        file_patterns = glob.glob("./pareto/validation/embedded/validation*_{0}_*.csv".format(year))
        p_train, r_train, p_test, r_test = pipeline(year, predictor_level="pixel_level",
                                                    response_level="daily_total", scale=False)
        predictors = p_train
        predictors_dict = [None, None, None]  # this is a list not dict
        predictors_indices = np.arange(predictors.shape[1])
        predictors_dict[0] = predictors[:, predictors_indices % 3 == 0]
        predictors_dict[1] = predictors[:, predictors_indices % 3 == 1]
        predictors_dict[2] = predictors[:, predictors_indices % 3 == 2]
        metadata_dict["Predictors"] = predictors_dict

        toolbox = get_online_validation_toolbox(p_train, p_test, r_train, r_test,
                                                AggregationPrimitiveTree, RadiusMeanTerminal)

        for filename in file_patterns:
            _, _, _, equations = read_front(filename)
            for equation in equations:
                feature_names = [terminal.format() for terminal in terminals_from_string(equation, 4)]

                tree = AggregationPrimitiveTree.from_string(equation, toolbox.pset, nparams=4)
                original_semantics = semantics_func(tree, toolbox.pset.context, p_train)

                for key in feature_names:
                    modified_semantics = importance.neutralize_feature(tree, key, toolbox.pset, p_train, semantics_func)
                    change_in_semantics = euclidean_distance(modified_semantics, original_semantics)
                    if key not in change_in_semantics_dict:
                        change_in_semantics_dict[key] = [change_in_semantics]
                    else:
                        change_in_semantics_dict[key].append(change_in_semantics)

    change_in_semantics_file = "./pareto/validation/importance/change_in_semantics_embedded.csv"
    with open(change_in_semantics_file, 'wb') as f:
        writer = csv.writer(f)
        for key in change_in_semantics_dict:
            masked_array = np.ma.masked_invalid(change_in_semantics_dict[key])
            count = len(masked_array) - np.ma.count_masked(masked_array)
            change = np.sum(masked_array)
            writer.writerow([key, change / count])
