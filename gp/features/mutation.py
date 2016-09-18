import random
import numpy as np

from gp.features import parametrized_terminals


def multi_mutation(ind, mutations, probs):
    for mutation, probability in zip(mutations, probs):
        if random.random() < probability:
            ind, = mutation(ind)
    return ind,


def multi_crossover(ind1, ind2, crossovers, probs):
    for crossover, probability in zip(crossovers, probs):
        if random.random() < probability:
            ind1, ind2 = crossover(ind1, ind2)
    return ind1, ind2


def choose_single_mutation(ind, mutations, weights):
    rnd = random.random() * sum(weights)
    for mutation, w in zip(mutations, weights):
        rnd -= w
        if rnd < 0:
            ind, = mutation(ind)
            return ind,

    return ind,


def bounce_back_mutation(value, scale, lower_bound=0.0, upper_bound=1.0, maxit=5):
    out_of_bounds = True
    count = 0
    while out_of_bounds:
        count += 1
        new = value + np.random.normal(scale=scale)
        if new < lower_bound:
            new = 2 * lower_bound - new
        elif new > upper_bound:
            new = 2 * upper_bound - new

        if count > maxit:
            new = random.uniform(lower_bound, upper_bound)
            out_of_bounds = False
        elif new < lower_bound or new > upper_bound:
            out_of_bounds = True
        else:
            out_of_bounds = False
    return new


def real_parameter_mutation(ind, per_parameter_probability, distribution, lower_bound=-np.inf, upper_bound=np.inf):
    for node in ind:
        if isinstance(node, parametrized_terminals.ParametrizedTerminal):
            for index, value in enumerate(node.parameters):
                if random.random() < per_parameter_probability:
                    node.parameters[index] = value + distribution()
                    node.parameters[index] = np.clip(node.parameters[index], lower_bound, upper_bound)
    return ind,


def get_parametrized_terminal_indices(ind):
    parametrized_terminal_indices = [index
                                     for index, node in enumerate(ind)
                                     if isinstance(node, parametrized_terminals.ParametrizedTerminal)]
    return parametrized_terminal_indices


def select_one_point(ind):
    return random.choice(get_parametrized_terminal_indices(ind))


def one_point_parameter_mutation(ind, toolbox, metadata, radius_scale=0.25, iterations=1):
    try:
        selected_node = select_one_point(ind)
    except IndexError:
        return ind,

    curr_error = toolbox.evaluate_error(ind)

    for repetition in range(iterations):
        selected_node = select_one_point(ind)
        clone = toolbox.clone(ind)
        radius = clone[selected_node].radius
        if random.random() < 0.5:
            sigma = max(50.0, radius * radius_scale)
            clone[selected_node].radius = bounce_back_mutation(radius, scale=sigma,
                                                               lower_bound=10.0, upper_bound=1000.0)
        else:
            latitude, longitude = metadata["LatLon"]
            space = np.array(zip(latitude, longitude))
            if clone[selected_node].centroid is None:
                clone[selected_node].centroid = list(random.choice(zip(latitude, longitude)))
            distances = clone[selected_node].distances_to_centroid(space)
            sigma = max(10.0, radius * radius_scale)
            desired_distance = np.abs(np.random.normal(scale=sigma))
            # there could be multiple points with the same distance to desire
            distance_to_desire = np.abs(distances - desired_distance)
            nearest_distance = distance_to_desire.min()
            potential_positions = np.where(distance_to_desire == nearest_distance)
            new_centroid = random.choice(space[potential_positions, :])
            clone[selected_node].centroid = new_centroid[0].tolist()

        new_error = toolbox.evaluate_error(clone)
        fitness_change = new_error[0] - curr_error[0]
        if fitness_change < 0.0:
            ind = clone
            curr_error = new_error
    return ind,
