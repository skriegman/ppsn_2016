import cachetools
from gp.semantic import semantics
import numpy


def fast_numpy_evaluate(ind, context, predictors, error_function=None, expression_dict=None, arg_prefix="ARG"):
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
            vector = semantics.get_node_semantics(node, subtree_semantics, predictors, context, arg_prefix)
            expression_dict[expression] = vector

        expressions_stack.append(expression)
        semantics_stack.append(vector)

    if error_function is None:
        return semantics_stack.pop()
    else:
        return error_function(semantics_stack.pop())


def mean_absolute_error(vector, response):
    errors = numpy.abs(vector - response)
    mean_error = numpy.mean(errors)
    if not numpy.isfinite(mean_error):
        return numpy.inf,
    return mean_error.item(),


def root_mean_square_error(vector, response):
    with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
        squared_errors = numpy.square(vector - response)
    mse = numpy.mean(squared_errors)
    if not numpy.isfinite(mse):
        return numpy.inf,
    rmse = numpy.sqrt(mse)
    return rmse.item(),


def anti_correlation(vector, response):
    f_corr = 1 - abs(numpy.corrcoef(vector, response).item(1))
    if not numpy.isfinite(f_corr):
        return numpy.inf,
    return f_corr,
