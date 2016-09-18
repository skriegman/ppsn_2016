import random

from bitarray import bitarray
import numpy
from deap import gp

from gp.semantic import semantics


def simplify_constant_semantics(population, toolbox, semantics_threshold=0.01, size_threshold=0,
                                precompute_semantics=False):
    for ind in population:
        if len(ind) < size_threshold:
            continue

        if precompute_semantics:
            ind.semantics = toolbox.calc_semantics(ind)

        removed = bitarray(len(ind))
        removed.setall(False)
        num_nodes_removed = 0

        for subtree_semantics in ind.semantics:
            subtree_index = subtree_semantics.node_index
            subtree_size = subtree_semantics.tree_size
            if subtree_size > 1 and not removed[subtree_index]:
                min_value = numpy.max(subtree_semantics).item()
                max_value = numpy.min(subtree_semantics).item()
                if abs(max_value - min_value) < semantics_threshold:
                    new_const = random.uniform(min_value, max_value)
                    slice_begin = -num_nodes_removed + subtree_index
                    slice_end = slice_begin + subtree_size
                    ind[slice_begin:slice_end] = [gp.Terminal(new_const, True, float)]
                    removed[subtree_index:subtree_index + subtree_size] = True
                    num_nodes_removed += subtree_size - 1


def simplify_semantics_pairwise(population, toolbox, semantics_threshold=0.01, size_threshold=0,
                                precompute_semantics=False):
    if precompute_semantics:
        for ind in population:
            ind.semantics = toolbox.calc_semantics(ind)

    unique_expressions = semantics.get_syntactically_unique_semantics(population, ignore_constant=True,
                                                                      ignore_infinite=True)
    replacement_dict = semantics.get_expression_replacement_dict(unique_expressions, max_diff=semantics_threshold)

    for ind in population:
        if len(ind) < size_threshold:
            continue

        removed = bitarray(len(ind))
        removed.setall(False)
        num_nodes_removed = 0

        for subtree_semantics in ind.semantics:
            subtree_index = subtree_semantics.node_index
            subtree_size = subtree_semantics.tree_size
            if not removed[subtree_index] and subtree_semantics.expr in replacement_dict:
                slice_begin = -num_nodes_removed + subtree_index
                slice_end = slice_begin + subtree_size
                replacement_nodes = replacement_dict[subtree_semantics.expr]
                ind[slice_begin:slice_end] = replacement_nodes
                removed[subtree_index:subtree_index + subtree_size] = True
                num_nodes_removed += subtree_size - len(replacement_nodes)


def simplify_all(population, toolbox, semantics_threshold=0.01, size_threshold=0, precompute_semantics=False):
    simplify_constant_semantics(population, toolbox, semantics_threshold, size_threshold,
                                precompute_semantics=precompute_semantics)
    simplify_semantics_pairwise(population, toolbox, semantics_threshold, size_threshold, precompute_semantics=True)
