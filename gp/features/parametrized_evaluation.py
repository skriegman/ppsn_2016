from copy import deepcopy
import re
from collections import deque
import itertools
import cachetools
import numpy as np
from deap import gp

from gp.experiments import symbreg
from gp.semantic.semantics import Semantics, get_terminal_semantics
from gp.features import parametrized_terminals


def fast_numpy_evaluate_metadata(ind, context, predictors, metadata, error_function=None,
                                 expression_dict=None, terminal_dict=None,
                                 arg_prefix="ARG"):
    semantics_stack = []
    expressions_stack = []

    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=2000)
    if terminal_dict is None:
        terminal_dict = cachetools.LRUCache(maxsize=2000)

    for node in reversed(ind):
        expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
        subtree_semantics = [semantics_stack.pop() for _ in range(node.arity)]

        if expression in expression_dict:
            # print "Retrieving {}".format(expression)
            vector = expression_dict[expression]
        else:
            # print "Evaluating {}".format(expression)
            vector = get_node_semantics(node, subtree_semantics, predictors, context,
                                        metadata, ind, terminal_dict, arg_prefix)
            if not isinstance(node, parametrized_terminals.ParametrizedTerminal):  # otherwise stored in terminal_dict
                expression_dict[expression] = vector

        expressions_stack.append(expression)
        semantics_stack.append(vector)

    if error_function is None:
        return semantics_stack.pop()
    else:
        return error_function(semantics_stack.pop())


def get_node_semantics(node, subtree_semantics, predictors, context, metadata, ind, terminal_dict, arg_prefix="ARG"):
    if isinstance(node, parametrized_terminals.ParametrizedTerminal):
        vector = node.get_input_vector(predictors, metadata, ind, terminal_dict=terminal_dict)
        vector = symbreg.numpy_protected_div_zero(vector - np.mean(vector), np.std(vector))
    elif isinstance(node, gp.Terminal):
        vector = get_terminal_semantics(node, context, predictors, arg_prefix)
    else:
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vector = context[node.name](*subtree_semantics)
    return vector


def calc_eval_semantics(ind, context, predictors, metadata, eval_semantics, expression_dict=None, terminal_dict=None,
                        arg_prefix="ARG"):
    ind.semantics = calculate_semantics(ind, context, predictors, metadata, expression_dict, terminal_dict, arg_prefix)
    return eval_semantics(ind.semantics[0])


def calculate_semantics(ind, context, predictors, metadata, expression_dict=None, terminal_dict=None, arg_prefix="ARG"):
    semantics = []
    sizes_stack = []
    height_stack = []
    semantics_stack = []
    expressions_stack = []

    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=2000)
    if terminal_dict is None:
        terminal_dict = cachetools.LRUCache(maxsize=2000)

    for index, node in enumerate(reversed(ind)):
        expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
        subtree_semantics = [semantics_stack.pop() for _ in range(node.arity)]
        subtree_size = sum([sizes_stack.pop() for _ in range(node.arity)]) + 1
        subtree_height = max([height_stack.pop() for _ in range(node.arity)]) + 1 if node.arity > 0 else 0

        if expression in expression_dict:
            vector = expression_dict[expression]
        else:
            vector = get_node_semantics(node, subtree_semantics, predictors, context, metadata, ind, terminal_dict,
                                        arg_prefix)
            expression_dict[expression] = vector

        expressions_stack.append(expression)
        semantics_stack.append(vector)
        sizes_stack.append(subtree_size)
        height_stack.append(subtree_height)

        semantics.append(Semantics(vector, len(ind) - index - 1, expression, ind, subtree_size, subtree_height))

    semantics.reverse()
    return semantics


######################################
# Primitive Trees                    #
######################################

class MutablePrimitiveTree(gp.PrimitiveTree):
    def __deepcopy__(self, memo):
        new = self.__class__([deepcopy(x) for x in self])
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    @classmethod
    def from_string(cls, string, pset, nparams):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v()\[\],]", string.replace('array', ','))
        expr = []
        ret_types = deque()

        def consume(iterator, n):
            '''Advance the iterator n-steps ahead. If n is none, consume entirely.'''
            deque(itertools.islice(iterator, n), maxlen=0)

        iterator = range(len(tokens)).__iter__()

        for i in iterator:
            token = tokens[i]
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))
                expr.append(primitive)
                if isinstance(primitive, gp.Primitive):
                    ret_types.extendleft(reversed(primitive.args))

            else:
                try:
                    x = eval("parametrized_terminals.{}".format(token))
                    parameters = []
                    num_params = nparams
                    count = 0
                    while len(parameters) < num_params:
                        if tokens[i + 1] not in ['', ',', '']:
                            parameters.append(float(tokens[i + 1]))
                        i += 1
                        count += 1
                    y = x()
                    y.set_params(*parameters)
                    expr.append(y)
                    consume(iterator, count)
                except SyntaxError:
                    try:
                        token = eval(token)
                    except NameError:
                        raise TypeError("Unable to evaluate terminal: {}.".format(token))

                    if type_ is None:
                        type_ = type(token)

                    if not issubclass(type(token), type_):
                        raise TypeError("Terminal {} type {} does not "
                                        "match the expected one: {}."
                                        .format(token, type(token), type_))
                    expr.append(gp.Terminal(token, False, type_))
        return cls(expr)


class AggregationPrimitiveTree(MutablePrimitiveTree):
    def __init__(self, content):
        MutablePrimitiveTree.__init__(self, content)
        self.num_features = 3
