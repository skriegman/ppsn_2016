import glob
import csv
import ast
import numpy as np
import re
from swe.data_tools.process_raw_data import read_coordinates
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


def get_last_lines(year):
    file_patterns = glob.glob("../data/SweData/hill_climber/*history*{}*".format(year))
    last_lines = []
    for file_name in file_patterns:
        with open(file_name, "rb") as f:
            all_lines = f.readlines()
            last_lines.append(all_lines[-1])
    return last_lines


def save_results(year):
    results = get_last_lines(year)
    filename = "./aggregated_hill_climber_results_{}.csv".format(year)
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "generation", "train_error", "test_error", "validation_error",
                         "coefficients", "circle_parameters"])
        for result in enumerate(results):
            writer.writerow(result)


def read_results(model, year, resolution):
    num_circles = resolution*resolution*3
    file_name = "/Users/mecl/gp_mecl/data/SweData/hill_climber/aggregated_{0}_hill_climber_results_{1}_{2}.csv".format(model, year, num_circles)
    train_error = []
    test_error = []
    coefficients = []
    circle_parameters = []
    with open(file_name, "rb") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            train_error.append(ast.literal_eval(row[1].split(",")[1]))
            test_error.append(ast.literal_eval(row[1].split(",")[2]))
            coefficients.append(row[1].split(",")[4:4+num_circles])
            circle_parameters.append(row[1].split(",")[4+num_circles:])

    rep = {'[': '', ']': '', '(': '', ')': '', '"': ""}
    rep = dict((re.escape(k), v) for k, v in rep.iteritems())
    pattern = re.compile("|".join(rep.keys()))
    for idx, coef in enumerate(coefficients):
        coefficients[idx] = [eval(pattern.sub(lambda m: rep[re.escape(m.group(0))], x)) for x in coef]

    for idx, param in enumerate(circle_parameters):
        circle_parameters[idx] = [eval(pattern.sub(lambda m: rep[re.escape(m.group(0))], x)) for x in param]

    centers = []
    radiuses = []
    features = []
    for param in circle_parameters:
        for key, val in zip(np.arange(len(param)) % 4, param):
            if key == 0 or key == 1:
                centers.append(val)
            elif key == 2:
                radiuses.append(val)
            else:
                features.append(val)
    centers = np.array(centers).reshape(len(centers)/2, 2)
    circle_parameters = zip(centers.tolist(), radiuses, features)

    return np.array(train_error), np.array(test_error), coefficients, circle_parameters


def circle_mask(a, b, r, n):
    y, x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array = np.zeros((n, n))
    array[mask] = 1
    return array


def earth_to_square(xy, radiuses, weights, features, nonzero_frequency_mask):
    # latitude, longitude = np.load('../data/SweData/metadata/latlon.npy')
    latitude, longitude = read_coordinates(directory='/Users/mecl/gp_mecl/data/SweData/raw_data/')
    latitude, longitude = latitude.flatten(), longitude.flatten()
    coordinates = np.array(zip(latitude, longitude)).tolist()
    # found = [any(e == z for e in coordinates) for z in xy]
    xy_indices = [coordinates.index(z) for z in xy]

    grid_index = np.unravel_index(xy_indices, (113, 113))
    # sanity check
    # grid = np.array(coordinates).reshape((113, 113, 2))
    # print np.all(xy == grid[grid_index])

    features = np.array(features).flatten()
    intensities = np.abs(weights).flatten()
    centers = zip(*grid_index)

    usefulness = np.zeros((113, 113))
    count = np.zeros((113, 113))
    max_usefulness = np.zeros((113, 113))
    normalized_nonzero_frequency = [(nonzero_frequency_mask[f] - np.min(nonzero_frequency_mask[f])) /
                                    (np.max(nonzero_frequency_mask[f]) - np.min(nonzero_frequency_mask[f]))
                                    for f in range(3)]

    for (x, y), r, w, f in zip(centers, radiuses, intensities, features):
        count += circle_mask(y, x, r, 113)
        usefulness += circle_mask(y, x, r, 113) * w  # * np.log(normalized_nonzero_frequency[int(f)] + np.e)
        max_usefulness = np.maximum(usefulness, circle_mask(y, x, r, 113) * w)

    heatmap = (usefulness / count) * max_usefulness * np.log(max_usefulness + np.e)
    return heatmap


def plot_wrapper(model, resolution, ax, nonzero_frequency_mask):
    xy = []
    r = []
    w = []
    f = []
    for year in range(2003, 2011):
        _, test, coefficients, circle_parameters = read_results(model, year, resolution)
        latlons, radiuses, features = zip(*circle_parameters)
        radiuses = np.array(radiuses) / 10.
        xy.extend(latlons)
        r.extend(radiuses.tolist())
        w.extend(coefficients)
        f.extend(features)

    w = np.power(w, 2)
    heatmap = earth_to_square(xy, r, w, f, nonzero_frequency_mask)

    # ax = plt.subplot(1, 1, 1, aspect='equal')
    # plot_single_circle_grid(centers, radiuses, ax, intensities, grid=False, alpha=0.2)

    image = ax.pcolormesh(heatmap, cmap='jet')

    # plt.title("Wrapped Ridge (WR)", fontsize=14)
    plt.ylim((0, 112))
    plt.xlim((0, 112))
    plt.xticks([])
    plt.yticks([])

    ax.annotate(int(resolution), xy=(4, 98), fontsize=30,
                path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])

    # plt.savefig("/Users/mecl/gp_mecl/exp/swe/heatmap_ridge.pdf")
    # plt.show()
    return image