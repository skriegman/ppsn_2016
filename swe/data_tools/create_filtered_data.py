import sys
import numpy as np
import glob
import csv
from sklearn import preprocessing, linear_model
from gp.experiments import fast_evaluate


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


def distances_to_centroid(p1, p2):
    # adopted from geopy.distance.great_circle
    earth_radius = 6372.795
    lat1, lng1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lng2 = np.radians(p2[0]), np.radians(p2[1])
    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)
    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)
    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                           (cos_lat1 * sin_lat2 -
                            sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                   sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
    return earth_radius * d


def circle_mask(a, b, r, n):
    y, x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array = np.zeros((n, n))
    array[mask] = 1
    return array


def get_circle_grid(new_resolution, old_resolution):
    tile_lengths = np.repeat(old_resolution / float(new_resolution), new_resolution)
    radiuses = tile_lengths * (2 ** 0.5) / 2
    centroids = tile_lengths / 2 + np.pad(np.cumsum(tile_lengths), (1, 0), mode='constant')[:-1]
    return centroids, radiuses


def aggregate_feature(feature, new_resolution, old_resolution=113):
    centroids, radiuses = get_circle_grid(new_resolution, old_resolution)
    dataframe = np.empty((len(feature), new_resolution**2))
    column = 0
    # start = time.time()
    for x in centroids:
        for y, r in zip(centroids, radiuses):
            mask = circle_mask(x, y, r, 113)
            predictors_mask = mask.flatten()
            dataframe[:, column] = np.mean(np.compress(predictors_mask, feature, axis=1), axis=1)
            column += 1

    # print "runtime: {}".format(time.time() - start)
    return dataframe


def save_datasets(year, resolution):
    p_train, r_train, p_test, r_test = pipeline(year, "pixel_level", "daily_total")
    feature_indices = np.arange(np.shape(p_test)[1]) % 3

    p_train0 = p_train[:, feature_indices == 0]
    p_train1 = p_train[:, feature_indices == 1]
    p_train2 = p_train[:, feature_indices == 2]
    p_train0 = aggregate_feature(p_train0, resolution)
    p_train1 = aggregate_feature(p_train1, resolution)
    p_train2 = aggregate_feature(p_train2, resolution)
    aggregated_training_features = np.hstack((p_train0, p_train1, p_train2))

    p_test0 = p_test[:, feature_indices == 0]
    p_test1 = p_test[:, feature_indices == 1]
    p_test2 = p_test[:, feature_indices == 2]
    p_test0 = aggregate_feature(p_test0, resolution)
    p_test1 = aggregate_feature(p_test1, resolution)
    p_test2 = aggregate_feature(p_test2, resolution)
    aggregated_testing_features = np.hstack((p_test0, p_test1, p_test2))

    np.save("../data/SweData/train/grid/predictors_leave_out_{1}_resolution_{0}".format(resolution, year), aggregated_training_features)
    np.save("../data/SweData/test/grid/predictors_validate_{1}_resolution_{0}".format(resolution, year), aggregated_testing_features)


def save_standard_results(model_type, year):
    p_train, r_train, p_test, r_test = pipeline(year, "pixel_level", "daily_total")
    scale = preprocessing.StandardScaler()
    p_train = scale.fit_transform(p_train)
    p_test = scale.transform(p_test)

    if model_type == "Lasso":
        lr = linear_model.LassoCV()
    elif model_type == "Ridge":
        lr = linear_model.RidgeCV()
    elif model_type == "OLS":
        lr = linear_model.LinearRegression()
    elif model_type == "Elastic_Net":
        lr = linear_model.ElasticNetCV()
    else:
        raise TypeError("Invalid model_type")

    model = lr.fit(p_train, r_train)
    predicted_train = model.predict(p_train)
    predicted_test = model.predict(p_test)
    train_correlation = fast_evaluate.anti_correlation(predicted_train, r_train)
    test_correlation = fast_evaluate.anti_correlation(predicted_test, r_test)
    train_rmse = fast_evaluate.root_mean_square_error(predicted_train, r_train)
    test_rmse = fast_evaluate.root_mean_square_error(predicted_test, r_test)
    train_mae = fast_evaluate.mean_absolute_error(predicted_train, r_train)
    test_mae = fast_evaluate.mean_absolute_error(predicted_test, r_test)
    coefficients = model.coef_.tolist()

    filename = "Standard_{0}_year_{1}.csv".format(model_type, year)
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["train_correlation", "test_correlation",
                         "train_rmse", "test_rmse", "train_mae", "test_mae",
                         "coefficients"])
        writer.writerow((train_correlation, test_correlation,
                         train_rmse, test_rmse, train_mae, test_mae,
                         coefficients))


if __name__ == "__main__":
    random_seed = int(sys.argv[1])
    YEAR = int(sys.argv[2])
    RESOLUTION = int(sys.argv[3])

    save_datasets(YEAR, RESOLUTION)

    # save_standard_results("Lasso", YEAR)
    # save_standard_results("Ridge", YEAR)
    # save_standard_results("Elastic_Net", YEAR)




