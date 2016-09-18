import csv
from deap import tools, gp, base
import numpy
import operator


def configure_basic_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("max", numpy.max)
    return mstats


def configure_inf_protected_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", lambda values: numpy.mean(filter(numpy.isfinite, values)))
    mstats.register("std", lambda values: numpy.std(filter(numpy.isfinite, values)))
    mstats.register("min", lambda values: numpy.min(filter(numpy.isfinite, values)))
    mstats.register("max", lambda values: numpy.max(filter(numpy.isfinite, values)))
    mstats.register("median", lambda values: numpy.median(filter(numpy.isfinite, values)))
    stats_best_ind = tools.Statistics(lambda ind: (ind.fitness.values[0], len(ind)))
    stats_best_ind.register("size_min", lambda values: min(values)[1])
    stats_best_ind.register("size_max", lambda values: max(values)[1])
    mstats["best_tree"] = stats_best_ind
    return mstats


def save_log_to_csv(pop, log, file_name):
    columns = [log.select("cpu_time")]
    columns_names = ["cpu_time"]
    for chapter_name, chapter in log.chapters.items():
        for column in chapter[0].keys():
            columns_names.append(str(column) + "_" + str(chapter_name))
            columns.append(chapter.select(column))

    rows = zip(*columns)
    with open(file_name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(columns_names)
        for row in rows:
            writer.writerow(row)


def save_hof(hof, test_toolbox=None):
    def decorator(func):
        def wrapper(pop, log, file_name):
            func(pop, log, file_name)
            hof_file_name = "trees_" + file_name
            with open(hof_file_name, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(["gen", "fitness", "tree"])
                for gen, ind in enumerate(hof.historical_trees):
                    if test_toolbox is not None:
                        test_error = test_toolbox.test_evaluate(ind)[0]
                        writer.writerow([gen, ind.fitness, str(ind), test_error])
                    else:
                        writer.writerow([gen, ind.fitness, str(ind)])

        return wrapper

    return decorator


def save_archive(archive):
    def decorator(func):
        def wrapper(pop, log, file_name):
            func(pop, log, file_name)
            archive.save(file_name)

        return wrapper

    return decorator
