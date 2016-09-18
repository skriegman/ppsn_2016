import random
import numpy as np
from copy import deepcopy
from deap import gp
from deap.gp import PrimitiveSet


class ParametrizedPrimitiveSet(PrimitiveSet):
    def __init__(self, name, arity, prefix="ARG"):
        PrimitiveSet.__init__(self, name, arity, prefix)

    def addParametrizedTerminal(self, parametrized_terminal_class):
        self._add(parametrized_terminal_class)
        self.context[parametrized_terminal_class.__name__] = parametrized_terminal_class.call
        self.terms_count += 1


class ParametrizedTerminal(gp.Terminal):
    ret = object

    def __init__(self, name):
        gp.Terminal.__init__(self, name, True, object)
        self.parameters = []

    def __deepcopy__(self, memo):
        new = self.__class__()
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def get_input_vector(self, data, metadata, individual, terminal_dict):
        raise NotImplementedError

    def call(*parameters):
        pass  # implement this method to make the class work with standard gp.compile

    def format(self):
        params = ",".join(str(arg) for arg in self.parameters)
        return "{}({})".format(self.__class__.__name__, params)


######################################
# Drop Pin Aggregation               #
######################################

class RadiusAggregationTerminal(ParametrizedTerminal):
    def __init__(self):
        ParametrizedTerminal.__init__(self, "AggregationTerminal")
        self.feature = random.choice(xrange(3))
        self.statistic = None
        self.metadata_index = "LatLon"
        self.centroid = None
        self.radius = random.uniform(100, 1000)

    def set_params(self, latitude, longitude, radius, feature):
        self.centroid = [latitude, longitude]
        self.radius = radius
        self.feature = feature

    def distances_to_centroid(self, space):
        # adopted from geopy.distance.great_circle
        earth_radius = 6372.795

        lat1, lng1 = np.radians(self.centroid[0]), np.radians(self.centroid[1])
        lat2, lng2 = np.radians(space[:, 0]), np.radians(space[:, 1])

        sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
        sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

        delta_lng = lng2 - lng1
        cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

        d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                               (cos_lat1 * sin_lat2 -
                                sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                       sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
        return earth_radius * d

    def get_input_vector(self, data, metadata_dict, individual, terminal_dict=None):
        data = metadata_dict["Predictors"][int(self.feature)]
        latitude, longitude = metadata_dict["LatLon"]
        if self.centroid is None:
            self.centroid = list(random.choice(zip(latitude, longitude)))

        terminal_string = self.format()
        if terminal_dict and terminal_string in terminal_dict:
            return terminal_dict[terminal_string]

        space = np.array(zip(latitude, longitude))
        distances = self.distances_to_centroid(space)
        output = self.statistic(data[:, distances < self.radius], axis=1)
        if terminal_dict:
            terminal_dict[terminal_string] = output
        return output

    def get_validation_input_vector(self, data, metadata_dict, individual, terminal_dict=None):
        latitude, longitude = metadata_dict["LatLon"]
        if self.centroid is None:
            self.centroid = list(random.choice(zip(latitude, longitude)))
        space = np.array(zip(latitude, longitude))
        distances = self.distances_to_centroid(space)
        metadata = np.repeat(distances, individual.num_features)
        feature_indices = np.arange(len(metadata)) % individual.num_features == self.feature
        choice = np.logical_and(metadata < self.radius, feature_indices)
        return self.statistic(np.compress(choice, data, axis=1), axis=1)

    def plot(self, metadata_dict):
        latitude, longitude = metadata_dict["LatLon"]
        space = np.array(zip(latitude, longitude))
        distances = self.distances_to_centroid(space)
        weights = np.zeros_like(distances)
        weights[distances < self.radius] = 1
        return np.reshape(weights, (113, 113))

    # Euclidean distance (faster but less accurate)
    # def circle_mask(self, n=113):
    #     a, b = self.centroid
    #     y, x = np.ogrid[-a:n-a, -b:n-b]
    #     mask = x*x + y*y <= self.radius*self.radius
    #     array = np.zeros((n, n))
    #     array[mask] = 1
    #     return array
    #
    # def get_input_vector(self, data, metadata_dict, individual):
    #     p = metadata_dict["Predictors"][self.feature]
    #     if self.centroid is None:
    #         self.centroid = [random.randint(0, 112), random.randint(0, 112)]
    #     within_radius_mask = self.circle_mask()
    #     return self.statistic(np.compress(within_radius_mask.flatten(), p, axis=1), axis=1)

    def format(self):
        return "{}({},{},{})".format(self.__class__.__name__, self.centroid, self.radius, self.feature)


class RadiusMeanTerminal(RadiusAggregationTerminal):
    def __init__(self):
        RadiusAggregationTerminal.__init__(self)
        self.statistic = np.mean
