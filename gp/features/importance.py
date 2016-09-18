import numpy as np
from deap import gp


class NeutralNode(gp.Terminal):
    ret = float

    def __init__(self, values):
        gp.Terminal.__init__(self, values, True, float)
        if values == 'ones':
            self.value = 1.0
        elif values == 'zeros':
            self.value = 0.0

    def format(self):
        return 'NeutralNode({})'.format(int(self.value))


class NeutralPrimitive(gp.Primitive):
    ret = float

    def __init__(self, values):
        gp.Primitive.__init__(self, values, [float], float)


def neutralize_feature(ind, name, pset, predictors, semantics_func):
    pset.addPrimitive(np.zeros_like, 1, name='zeros')
    pset.addPrimitive(np.ones_like, 1, name='ones')
    expressions_stack = []
    neutral_node_indices = []
    reversed_ind = ind[::-1]

    for index, node in enumerate(reversed(ind)):
        subtree = [expressions_stack.pop() for _ in range(node.arity)]
        expression = node.format(*subtree)

        if "("+name+")" in expression or expression == name:
            if node.arity == 0:
                # case 1: neutral terminal
                # record index and neutralize node with zeros
                neutral_node_indices.append(index)
                reversed_ind[index] = NeutralNode(values='zeros')

            elif node.arity == 1 and index-1 in neutral_node_indices:
                # case 2: unary operation of neutral node
                # record index and neutralize node with zeros
                neutral_node_indices.append(index)
                reversed_ind[index] = NeutralPrimitive(values='zeros')

            elif (node.arity == 2) and (index-1 in neutral_node_indices) and (index-2 in neutral_node_indices):
                # case 3: binary operation of two neutral nodes
                # record index and neutralize node with zeros
                neutral_node_indices.append(index)
                reversed_ind[index] = NeutralPrimitive(values='zeros')

            elif node.arity == 2 and index-1 in neutral_node_indices:
                # case 4: binary operation with right neutral node
                # neutralize node with ones
                if node.name in ["multiply", "numpy_protected_div_dividend"]:
                    if isinstance(reversed_ind[index-1], NeutralNode):
                        reversed_ind[index-1] = NeutralNode(values='ones')
                    if isinstance(reversed_ind[index-1], NeutralPrimitive):
                        reversed_ind[index-1] = NeutralPrimitive(values='ones')

            elif node.arity == 2 and index-2 in neutral_node_indices:
                # case 5: binary operation with left neutral node
                if node.name in ["multiply", "numpy_protected_div_dividend"]:
                    # neutralize node with ones
                    if isinstance(reversed_ind[index-2], NeutralNode):
                        reversed_ind[index-2] = NeutralNode(values='ones')
                    if isinstance(reversed_ind[index-2], NeutralPrimitive):
                        reversed_ind[index-2] = NeutralPrimitive(values='ones')

        expressions_stack.append(expression)
    # print neutral_node_indices
    ind = reversed_ind[::-1]
    # print gp.PrimitiveTree(ind)

    return semantics_func(ind, pset.context, predictors)
