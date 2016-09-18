import operator
from deap import gp
import numpy


def numpy_protected_div_dividend(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = left[numpy.isinf(x)]
            x[numpy.isnan(x)] = left[numpy.isnan(x)]
        elif numpy.isinf(x) or numpy.isnan(x):
            x = left
    return x


def numpy_protected_div_zero(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 0.0
            x[numpy.isnan(x)] = 0.0
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 0.0
    return x


def numpy_protected_div_one(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1.0
            x[numpy.isnan(x)] = 1.0
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1.0
    return x


def numpy_protected_log_abs(x):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.log(numpy.abs(x))
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = -1e300
            x[numpy.isnan(x)] = 0
        elif numpy.isinf(x):
            x = -1e300
        elif numpy.isnan(x):
            x = 0
    return x


def numpy_protected_exponential(x):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.exp(x)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 0
            x[numpy.isnan(x)] = 0
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 0
    return x


def error_fitness(individual, toolbox, error_measure):
    func = toolbox.compile(expr=individual)
    return error_measure(func),


def normalized_mean_squared_error(func, predictors, response):
    return mean_squared_error(func, predictors, response) / numpy.var(response)


def mean_squared_error(func, predictors, response):
    squared_error = 0
    for instance in xrange(len(predictors)):
        try:
            predicted_val = func(*predictors[instance])
            e = response[instance] - predicted_val
            squared_error += (e * e)
        except (OverflowError, ValueError):
            return float("inf")
    return squared_error / len(predictors)


def get_anu_pset(arity):
    pset = gp.PrimitiveSet("MAIN", arity)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    return pset


def get_numpy_polynomial_explog_trig_pset(arity):
    pset = gp.PrimitiveSet("MAIN", arity)
    pset.addPrimitive(numpy.add, 2)
    pset.addPrimitive(numpy.subtract, 2)
    pset.addPrimitive(numpy.multiply, 2)
    pset.addPrimitive(numpy_protected_log_abs, 1)
    pset.addPrimitive(numpy.exp, 1)
    pset.addPrimitive(numpy.cos, 1)
    pset.addPrimitive(numpy.sin, 1)
    return pset


def get_numpy_pset(arity, div_function=None, log_function=None, prefix="ARG"):
    if div_function is None:
        div_function = numpy_protected_div_dividend

    if log_function is None:
        log_function = numpy_protected_log_abs

    pset = gp.PrimitiveSet("MAIN", arity, prefix=prefix)
    pset.addPrimitive(numpy.add, 2)
    pset.addPrimitive(numpy.subtract, 2)
    pset.addPrimitive(numpy.multiply, 2)
    pset.addPrimitive(div_function, 2)
    pset.addPrimitive(log_function, 1)
    pset.addPrimitive(numpy.cos, 1)
    pset.addPrimitive(numpy.sin, 1)
    pset.addPrimitive(numpy.exp, 1)
    return pset
