import sys
import os
import numpy
import csv
from sklearn import linear_model, preprocessing
from gp.experiments import symbreg, fast_evaluate


def save_linear_fit(model_type, year, resolution, p_train, r_train, p_test, r_test):
    scale = preprocessing.StandardScaler()
    p_train = scale.fit_transform(p_train)
    p_test = scale.transform(p_test)

    if model_type == "Lasso":
        lr = linear_model.LassoCV()
    elif model_type == "Ridge":
        lr = linear_model.RidgeCV()
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

    filename = "Filtered_{0}_year_{1}_resolution_{2}.csv".format(model_type, year, resolution)
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

    training_data_dir = "../data/SweData/train/"
    train_predictors_name = "grid/predictors_leave_out_{0}_resolution_{1}.npy".format(YEAR, RESOLUTION)
    train_response_name = "daily_total/response_daily_total_leave_out_{}.npy".format(YEAR)
    train_predictors_file = os.path.join(training_data_dir, train_predictors_name)
    train_response_file = os.path.join(training_data_dir, train_response_name)
    p_train, r_train = numpy.load(train_predictors_file), numpy.load(train_response_file)
    r_train = symbreg.numpy_protected_log_abs(r_train)

    test_data_dir = "../data/SweData/test/"
    test_predictors_name = "grid/predictors_validate_{0}_resolution_{1}.npy".format(YEAR, RESOLUTION)
    test_response_name = "daily_total/response_daily_total_validate_{}.npy".format(YEAR)
    test_predictors_file = os.path.join(test_data_dir, test_predictors_name)
    test_response_file = os.path.join(test_data_dir, test_response_name)
    p_test, r_test = numpy.load(test_predictors_file), numpy.load(test_response_file)
    r_test = symbreg.numpy_protected_log_abs(r_test)

    save_linear_fit("Lasso", YEAR, RESOLUTION, p_train, r_train, p_test, r_test)
    # save_linear_fit("Ridge", YEAR, RESOLUTION, p_train, r_train, p_test, r_test)
