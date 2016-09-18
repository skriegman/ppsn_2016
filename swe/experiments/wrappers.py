import csv
import glob
import sys
import random
import time
import numpy as np
from sklearn import preprocessing, linear_model

from gp.experiments import fast_evaluate, symbreg
from gp.features.mutation import bounce_back_mutation


def pipeline(year, predictor_level, response_level, scale=False):
    predictors_train = glob.glob("../data/SweData/train/{0}/predictors*{1}.npy".format(predictor_level, year))
    response_train = glob.glob("../data/SweData/train/{0}/response*{1}.npy".format(response_level, year))
    predictors_test = glob.glob("../data/SweData/test/{0}/predictors*{1}.npy".format(predictor_level, year))
    response_test = glob.glob("../data/SweData/test/{0}/response*{1}.npy".format(response_level, year))
    p_train, r_train = np.load(predictors_train[0]), np.load(response_train[0])
    p_test, r_test = np.load(predictors_test[0]), np.load(response_test[0])

    r_train, r_test = np.log(r_train), np.log(r_test)

    if scale:  # never scale in preprocessing for circle method
        scale = preprocessing.StandardScaler()
        p_train = scale.fit_transform(p_train)
        p_test = scale.transform(p_test)
    return p_train, r_train, p_test, r_test


def spherical_distance(centroid, space):
    earth_radius = 6372.795
    lat1, lng1 = np.radians(centroid[0]), np.radians(centroid[1])
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


def get_input_vector(data, lat_lon, centroid, radius, feature):
    latitude, longitude = lat_lon
    space = np.array(zip(latitude, longitude))
    distances = spherical_distance(centroid, space)
    metadata = np.repeat(distances, 3)
    feature_indices = np.arange(len(metadata)) % 3 == feature
    choice = np.logical_and(metadata < radius, feature_indices)
    if not np.any(choice):  # there are no points in the selected bin
        return np.zeros(shape=(len(data),))
    vector = np.mean(np.compress(choice, data, axis=1), axis=1)
    return vector  # not scaled yet


def linear_model_with_circles(p_train, r_train, p_test, r_test, lat_lon, centroids, radiuses, features,
                              inner_cross_validation_repeats=10, model=linear_model.LassoCV):

    count = 0
    for centroid, radius, feature in zip(centroids, radiuses, features):
        train_vector = get_input_vector(p_train, lat_lon, centroid, radius, feature)
        pc_train = symbreg.numpy_protected_div_zero(train_vector - np.mean(train_vector), np.std(train_vector))

        test_vector = get_input_vector(p_test, lat_lon, centroid, radius, feature)
        pc_test = symbreg.numpy_protected_div_zero(test_vector - np.mean(train_vector), np.std(train_vector))

        if count > 0:
            p_train_table = np.vstack((p_train_table, pc_train))
            p_test_table = np.vstack((p_test_table, pc_test))
        else:
            p_train_table = pc_train
            p_test_table = pc_test
            count += 1

    p_train_table = p_train_table.T
    p_test_table = p_test_table.T

    validation_errors = []
    for _ in range(inner_cross_validation_repeats):
        training_subset = np.random.choice(len(p_train_table), 1500, replace=False)
        validation_subset = np.ones(len(p_train_table), np.bool)
        validation_subset[training_subset] = 0

        training_predictors = p_train_table[training_subset, :]
        validation_predictors = p_train_table[validation_subset, :]
        training_response = r_train[training_subset]
        validation_response = r_train[validation_subset]

        lr = model()
        model = lr.fit(training_predictors, training_response)
        validation_error = fast_evaluate.root_mean_square_error(model.predict(validation_predictors), validation_response)
        validation_errors.append(validation_error)

    lr = model()
    model = lr.fit(p_train_table, r_train)
    train_error = fast_evaluate.root_mean_square_error(model.predict(p_train_table), r_train)
    test_error = fast_evaluate.root_mean_square_error(model.predict(p_test_table), r_test)
    coef = model.coef_.tolist()

    return train_error, test_error, np.mean(validation_error) if len(validation_errors) > 0 else train_error, coef


def one_point_parameter_mutation(lat_lon, centroids, radiuses, radius_scale=0.25, max_radius=1000.0):
    mutated_ind_index = random.choice(range(len(radiuses)))
    radius = radiuses[mutated_ind_index]
    centroid = centroids[mutated_ind_index]
    if random.random() < 0.5:
        sigma = max(50.0, radius * radius_scale)
        radiuses[mutated_ind_index] = bounce_back_mutation(radius, scale=sigma,
                                                           lower_bound=10.0, upper_bound=max_radius)
    else:
        latitude, longitude = lat_lon
        space = np.array(zip(latitude, longitude))
        distances = spherical_distance(centroid, space)
        sigma = max(10.0, radius * radius_scale)
        desired_distance = np.abs(np.random.normal(scale=sigma))
        # there could be multiple points with the same distance to desire
        distance_to_desire = np.abs(distances - desired_distance)
        nearest_distance = distance_to_desire.min()
        potential_positions = np.where(distance_to_desire == nearest_distance)
        new_centroid = random.choice(space[potential_positions, :])
        centroids[mutated_ind_index] = new_centroid[0].tolist()
    return centroids, radiuses


def get_random_circle(lat_lon):
    latitude, longitude = lat_lon
    centroid = list(random.choice(zip(latitude, longitude)))
    radius = random.uniform(10, 1000)
    feature = random.choice(xrange(3))
    return centroid, radius, feature


def get_random_circles(lat_lon, ncircles):
    centroids = []
    radiuses = []
    features = []
    latitude, longitude = lat_lon
    for n in range(ncircles):
        centroids.append(list(random.choice(zip(latitude, longitude))))
        radiuses.append(random.uniform(100, 1000))
        features.append((n - 1) % 3)
    return centroids, radiuses, features


def hill_climber(year, num_circles, iterations):
    lat_lon = np.load('../data/SweData/metadata/latlon.npy')
    p_train, r_train, p_test, r_test = pipeline(year, predictor_level="pixel_level",
                                                response_level="daily_total", scale=False)

    centroids, radiuses, features = get_random_circles(lat_lon, num_circles)
    curr_circles = zip(centroids, radiuses, features)
    curr_train_error, curr_test_error, curr_validation_error, curr_coefficients = \
        linear_model_with_circles(p_train, r_train, p_test, r_test, lat_lon, centroids, radiuses, features)

    training_errors = [curr_train_error[0]]
    test_errors = [curr_test_error[0]]
    validation_errors = [curr_validation_error]
    coefficients = [curr_coefficients]
    circles = [curr_circles]

    # start = time.time()
    for gen in range(1, iterations):
        mutated_centroids, mutated_radiuses = one_point_parameter_mutation(lat_lon, centroids, radiuses)
        train_error, test_error, validation_error, coef = \
            linear_model_with_circles(p_train, r_train, p_test, r_test, lat_lon, centroids, radiuses, features)

        if validation_error < curr_validation_error:
            curr_test_error = test_error
            curr_train_error = train_error
            curr_validation_error = validation_error
            curr_coefficients = coef
            curr_circles = zip(mutated_centroids, mutated_radiuses, features)

        training_errors.append(curr_train_error[0])
        test_errors.append(curr_test_error[0])
        validation_errors.append(curr_validation_error)
        coefficients.append(curr_coefficients)
        circles.append(curr_circles)
        # print "time: {}".format(time.time() - start)

    return training_errors, test_errors, validation_errors, coefficients, circles


def save_hill_climber(year, num_circles, iterations, seed):
    hill_climber_history = hill_climber(year, num_circles, iterations)

    filename = "./hill_climber_history_{0}_{1}_{2}.csv".format(year, num_circles, seed)
    rows = zip(*hill_climber_history)
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_error", "test_error", "validation_error",
                         "coefficients", "circle_parameters"])
        for iteration, row in enumerate(rows):
            writer.writerow([iteration] + list(row))

    filename = "./hill_climber_results_{0}_{1}_{2}.csv".format(year, num_circles, seed)
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["centroid", "radius", "feature"])
        results = hill_climber_history[4][-1:][0]
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    random_seed = int(sys.argv[1])
    random.seed(random_seed)
    np.random.seed(random_seed)

    YEAR = int(sys.argv[2])
    RESOLUTION = int(sys.argv[3])

    num_circles = RESOLUTION * RESOLUTION * 3
    # start = time.time()
    save_hill_climber(YEAR, num_circles, 1000, random_seed)
    # print "time: {}".format(time.time() - start)

