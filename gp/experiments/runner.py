import random
import logging
import numpy


def configure_basic_logging(filename, level=logging.DEBUG):
    logging.basicConfig(filename=filename, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=level)


def configure_advanced_logging(filename, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)

    if len(logger.handlers) > 0:
        logger.handlers[0].stream.close()
        logger.removeHandler(logger.handlers[0])

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def run(random_seed, predictors, response, toolbox_functions, algorithm_names, benchmark_name="",
        test_predictors=None, test_response=None):
    """Runs sequentially given regression algorithms on a given dataset

    :param random_seed: seed for the random generator
    :param predictors: a list of tuples, each of which specifies values of predictor variables for a single observation
    :param response: a list of observed values of response variable (corresponding to observations in predictors list)
    :param toolbox_functions: a list of functions each of which produces a configuration of a single learning algorithm
    :param benchmark_name: optional name of the problem being solved, to be included in a produced csv result file
    """

    for toolbox_func, algorithm_name in zip(toolbox_functions, algorithm_names):
        if test_predictors is None or test_response is None:
            toolbox = toolbox_func(predictors=predictors, response=response)
        else:
            toolbox = toolbox_func(predictors=predictors, response=response, test_predictors=test_predictors,
                                   test_response=test_response)

        logging.info("Starting algorithm %s", algorithm_name)
        pop, log = toolbox.run()

        logging.info("Saving results of algorithm %s", algorithm_name)
        log_file_name = "{}_{}_{}.log".format(algorithm_name, benchmark_name, random_seed)
        toolbox.save(pop, log, log_file_name)


def run_data(random_seed, predictors, response, toolbox_functions, algorithm_names, dataset_name,
             test_predictors=None, test_response=None, logging_level=logging.INFO):
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    configure_advanced_logging("debug_{}.log".format(random_seed), level=logging_level)
    run(random_seed, predictors, response, toolbox_functions, algorithm_names, benchmark_name=dataset_name,
        test_predictors=test_predictors, test_response=test_response)
