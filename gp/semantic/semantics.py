import copy
import cachetools
import numpy
from deap import gp


class SemanticPrimitiveTree(gp.PrimitiveTree):
    def __init__(self, content):
        gp.PrimitiveTree.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        old_semantics = getattr(self, "semantics", None)
        self.__dict__["semantics"] = None
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        new.__dict__["semantics"] = self.__dict__["semantics"] = old_semantics
        return new


class Semantics(numpy.ndarray):
    def __new__(cls, numpy_array, node_index, expr, ind, tree_size=1, tree_height=0):
        self = numpy_array.view(cls)
        self.node_index = node_index
        self.tree_size = tree_size
        self.tree_height = tree_height
        self.expr = expr
        self.ind = ind
        return self

    def get_nodes(self):
        return self.ind[self.node_index:self.node_index + self.tree_size]

    def __deepcopy__(self, memo):
        copy = numpy.copy(self)
        return type(self)(copy, self.node_index, self.expr, self.ind, self.tree_size, self.tree_height)


def calc_eval_semantics(ind, context, predictors, eval_semantics, expression_dict=None, arg_prefix="ARG"):
    ind.semantics = calculate_semantics(ind, context, predictors, expression_dict, arg_prefix)
    return eval_semantics(ind.semantics[0])


def calculate_semantics(ind, context, predictors, expression_dict=None, arg_prefix="ARG"):
    semantics = []
    sizes_stack = []
    height_stack = []
    semantics_stack = []
    expressions_stack = []

    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=2000)

    for index, node in enumerate(reversed(ind)):
        expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
        subtree_semantics = [semantics_stack.pop() for _ in range(node.arity)]
        subtree_size = sum([sizes_stack.pop() for _ in range(node.arity)]) + 1
        subtree_height = max([height_stack.pop() for _ in range(node.arity)]) + 1 if node.arity > 0 else 0

        if expression in expression_dict:
            vector = expression_dict[expression]
        else:
            vector = get_node_semantics(node, subtree_semantics, predictors, context, arg_prefix)
            expression_dict[expression] = vector

        expressions_stack.append(expression)
        semantics_stack.append(vector)
        sizes_stack.append(subtree_size)
        height_stack.append(subtree_height)

        semantics.append(Semantics(vector, len(ind) - index - 1, expression, ind, subtree_size, subtree_height))

    semantics.reverse()
    return semantics


def get_node_semantics(node, subtree_semantics, predictors, context, arg_prefix="ARG"):
    if isinstance(node, gp.Terminal):
        vector = get_terminal_semantics(node, context, predictors, arg_prefix)
    else:
        with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vector = context[node.name](*subtree_semantics)
    return vector


def get_terminal_semantics(node, context, predictors, arg_prefix="ARG"):
    if isinstance(node, gp.Ephemeral) or isinstance(node.value, float):
        return numpy.ones(len(predictors)) * node.value

    if node.value in context:
        return numpy.ones(len(predictors)) * context[node.value]

    arg_index = node.value[len(arg_prefix):]
    return predictors[:, int(arg_index)]


def get_expressions(ind):
    expressions_stack = []
    expressions = []
    for index, node in enumerate(reversed(ind)):
        if isinstance(node, gp.Terminal):
            expression = node.format()
            expressions.append(expression)
            expressions_stack.append(expression)
        else:
            expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
            expressions.append(expression)
            expressions_stack.append(expression)
    expressions.reverse()
    return expressions


def get_syntactically_unique_semantics(population, ignore_constant=False, ignore_infinite=False, include_subtrees=True):
    """
    :param population: a population of individuals with precomputed semantics
    :return: a list of semantics corresponding to unique expressions/subtrees without constant semantics, i.e.,
    such that output the same (or close) results for each training input
    """

    subtree_expression_set = set()
    syntactically_unique_semantics = []
    for ind in population:
        for subtree_semantics in ind.semantics:
            if ignore_infinite and not numpy.isfinite(numpy.sum(subtree_semantics)):
                continue
            if ignore_constant and is_constant(subtree_semantics):
                continue

            if subtree_semantics.expr not in subtree_expression_set:
                subtree_expression_set.add(subtree_semantics.expr)
                syntactically_unique_semantics.append(subtree_semantics)

            if not include_subtrees:
                break

    return syntactically_unique_semantics


def is_constant(vector):
    return numpy.ptp(vector) <= 10e-8
