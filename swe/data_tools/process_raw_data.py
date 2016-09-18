from __future__ import division
import os
import h5py
import calendar
import numpy as np
from scipy import interpolate
import glob
import gdal
import osr


def read_data_from_h5_as_ndarray(predictors_file, response_file, predictors_dataset, response_dataset,
                                 columns=None):
    predictors = h5py.File(predictors_file, 'r')
    response = h5py.File(response_file, 'r')
    predictors_data = predictors[predictors_dataset][:]
    response_data = response[response_dataset][:].flatten()

    if columns is not None:
        predictors_data = predictors_data[:, columns]

    return predictors_data, response_data


def length_of_recon(year, directory='SweData/raw_data/'):
    recon_file = 'Reconstruction_10km/ReconSWE_h23v05_{}_10km.h5'.format(year)
    recon_path = os.path.join(directory,  recon_file)
    recon = h5py.File(recon_path, 'r')
    return len(np.array(recon['Grid']['MODIS_GRID_10km']['swe_mean']))


def read_coordinates(directory='SweData/raw_data/'):
    data_file = os.path.join(directory, 'afghan_h23v05_latlon.h5')
    database = h5py.File(data_file, 'r')
    latitude = np.array(database['Grid']['MODIS_GRID_10km']['latitude'])
    longitude = np.array(database['Grid']['MODIS_GRID_10km']['longitude'])
    return latitude, longitude


def coordinates_to_pixel(tif_path, coordinates):
    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srs_coordinates = srs.CloneGeogCS()
    pixel_transform = osr.CoordinateTransformation(srs_coordinates, srs)
    elevation = []
    for point in coordinates:
        (point[1], point[0], holder) = pixel_transform.TransformPoint(point[1], point[0])
        x = (point[1]-gt[0])/gt[1]
        y = (point[0]-gt[3])/gt[5]
        elevation.append(ds.ReadAsArray(int(x), int(y), 1, 1))
    return [e[0][0] if e is not None else 0 for e in elevation]


def map_elevation(latitude, longitude):
    coordinates = np.array(zip(latitude.flatten(), longitude.flatten()))
    elevation_files = glob.glob('SweData/raw_data/strm/srtm*/*.tif')
    elevation = np.zeros(len(coordinates))
    for image in elevation_files:
        mapped_from_file = np.array(coordinates_to_pixel(image, coordinates))
        elevation += mapped_from_file
    elevation = np.reshape(elevation, [113, 113])
    elevation[np.where(elevation == 0)] = np.nan
    for patch in elevation:
        fill_nans_with_interpolation(patch)
    return elevation


def read_data_from_h5(years, data_file, variable_name):
    data_cube = np.array([]).reshape(0, 113, 113)
    file_name = data_file  # without making this copy, formatting never changed after 1st iteration
    for year in years:
        data_file = file_name.format(year)
        database = h5py.File(data_file, 'r')
        end = length_of_recon(year)  # 273-274
        if calendar.isleap(year):
            start = 59  # March 1st is 60th day of year
        else:
            start = 58
        data_cube = np.append(data_cube, np.array(database['Grid']['MODIS_GRID_10km'][variable_name])[start:end],
                              axis=0)
    return data_cube


def set_negatives_to_zero(data):
    data[np.where(data < 0)] = 0


def fill_nans_with_zero(data):
    invalid_mask = np.isnan(data)
    data[invalid_mask] = 0


def fill_nans_with_interpolation(data):
    x, y = np.arange(113), np.arange(113)
    xx, yy = np.meshgrid(x, y)
    z = data
    f = interpolate.RectBivariateSpline(xx, yy, z)
    znew = f(x, y)

    invalid_mask = np.isnan(data)
    data[invalid_mask] = np.interp(np.flatnonzero(invalid_mask), np.flatnonzero(~invalid_mask), data[~invalid_mask])


def cube_to_table(cube):
    cube = np.array(cube)
    return cube.reshape(cube.shape[0], np.product(cube.shape[1:3]))


def daily_total(variable):
    return np.sum(cube_to_table(variable), axis=1)


def circle_binned_mean(variable, new_resolution, old_resolution=113):
    latitude, longitude = np.load("SweData/metadata/latlon.npy")
    latitude = latitude.reshape((113, 113))
    longitude = longitude.reshape((113, 113))

    radius = new_resolution * (2 ** 0.5) / 2
    cut_point = (old_resolution - 1) / new_resolution - 1
    remainder = (old_resolution - 1) % cut_point
    start = - remainder / 2
    dataframe = np.empty((len(variable), new_resolution ** 2))
    column = 0
    for i in range(start, 112, cut_point):
        for j in range(start, 112, cut_point):
            centroid = [latitude[i][j], longitude[i][j]]
            dataframe[:, column] = get_circle_input_vector(variable, centroid, radius)
            column += 1
    return dataframe


def get_circle_input_vector(variable, centroid, radius):
    latitude, longitude = np.load("SweData/metadata/latlon.npy")
    space = np.array(zip(latitude, longitude))
    distances = distances_to_centroid(centroid, space)
    choice = distances < radius
    if not np.any(choice):  # there are no points in the selected bin
        return np.zeros(shape=(len(variable),))
    return np.mean(np.compress(choice, variable, axis=1), axis=1)


def distances_to_centroid(centroid, space):
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


def run_all_data(years, level, radius, directory='SweData/raw_data/'):
    if level not in ['daily_total', 'elevation_binned', 'aspect_binned',
                     'aspect_elevation_binned', 'geographically_binned', 'pixel_level',
                     'circle_binned']:
        raise NameError(str(level))

    modis_file = '{0}MODIS_snow_10km/h23v05_WY{1}_10km.h5'.format(directory, '{}')
    amsr_file = '{0}AMSR-E_SWE_annualCube/smooth_afghan_AMSRE_h23v05_WY{1}.h5'.format(directory, '{}')
    recon_file = '{0}Reconstruction_10km/ReconSWE_h23v05_{1}_10km.h5'.format(directory, '{}')

    modis_sca_mean = read_data_from_h5(years, modis_file, 'sca_mean')
    modis_sca_stddev = read_data_from_h5(years, modis_file, 'sca_stddev')
    amsr_swe_smoothed = read_data_from_h5(years, amsr_file, 'amsr_swe_smoothed')
    recon_swe_mean = read_data_from_h5(years, recon_file, 'swe_mean')

    fill_nans_with_zero(modis_sca_mean)
    fill_nans_with_zero(modis_sca_stddev)
    fill_nans_with_zero(amsr_swe_smoothed)
    fill_nans_with_zero(recon_swe_mean)

    modis_sca_mean = cube_to_table(modis_sca_mean)
    modis_sca_stddev = cube_to_table(modis_sca_stddev)
    amsr_swe_smoothed = cube_to_table(amsr_swe_smoothed)
    recon_swe_mean = cube_to_table(recon_swe_mean)

    if level == 'daily_total':
        modis_sca_mean = np.mean(modis_sca_mean, axis=1)
        modis_sca_stddev = np.mean(modis_sca_stddev, axis=1)
        amsr_swe_smoothed = np.mean(amsr_swe_smoothed, axis=1)
        predictors = zip(modis_sca_mean, modis_sca_stddev, amsr_swe_smoothed)
        response = daily_total(recon_swe_mean)

    elif level == 'circle_binned':
        modis_sca_mean = circle_binned_mean(modis_sca_mean, radius)
        modis_sca_stddev = circle_binned_mean(modis_sca_stddev, radius)
        amsr_swe_smoothed = circle_binned_mean(amsr_swe_smoothed, radius)
        recon_swe_mean = circle_binned_mean(recon_swe_mean, radius)

    if level != 'daily_total':
        predictors = []
        for day in zip(modis_sca_mean, modis_sca_stddev, amsr_swe_smoothed):
            predictors.append(zip(day[0], day[1], day[2]))
        predictors = cube_to_table(predictors)
        response = np.array(recon_swe_mean)

    return predictors, response


def leave_one_year_out(level, radius=None, directory='SweData/'):
    years = np.arange(2003, 2012)
    index = range(len(years))
    k_folds = [years[index[:k] + index[(k+1):]].tolist() for k in index]
    for n, fold in enumerate(k_folds):
        p, r = run_all_data(fold, level=level, radius=radius)
        print "saving fold {0}, leave out {1}".format(n, years[n])
        np.save("{0}train/{1}/predictors_{1}_leave_out_{2}".format(directory, level, years[n]), p)
        np.save("{0}train/{1}/response_{1}_leave_out_{2}".format(directory, level, years[n]), r)


def generate_test_set(level, radius=None, directory='SweData/'):
    years = np.arange(2003, 2012)
    for year in years:
        p, r = run_all_data([year], level=level, radius=radius)
        print "saving {}".format(year)
        np.save("{0}test/{1}/predictors_{1}_validate_{2}".format(directory, level, year), p)
        np.save("{0}test/{1}/response_{1}_validate_{2}".format(directory, level, year), r)
