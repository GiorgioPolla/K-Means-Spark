

import time
import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.interpolate import make_interp_spline


FILE_PATH = '../data/iris'
OUT_PATH = '../out_py/'
IMG_FORMAT = 'png'
N_WORKERS = 8
N_PARTITIONS = 8


def read_file(file_path, n_part):
    rdd = sc.textFile(file_path, n_part)
    
    rdd = rdd.map(
        lambda x:
            x.split()
    )
    
    rdd_x = rdd.map(
        lambda x: 
            np.array([float(el) for el in x[:-1]])
    )
    
    rdd_y = rdd.map(
        lambda x:           
            x[-1]
    )
    return rdd_x, rdd_y


def standardise(rdd):    
    n_rows = rdd.count()
    
    col_sum = rdd.reduce(
        lambda x, y:
            [sum(el) for el in zip(x, y)]
    )
    mean = np.divide(col_sum, n_rows)
    
    variance = rdd.map(
        lambda x: np.square(x - mean)
    ).reduce(
        lambda x, y:
            [sum(el) for el in zip(x, y)]
    )
    std_dev = np.sqrt(np.divide(variance, n_rows))
    
    rdd = rdd.map(
        lambda x:
            np.divide((x - mean), std_dev)
    )
    
    return rdd


def init_rdd(rdd):
    rdd = rdd.map(
        lambda x: (
            -1,
            x
        )
    )
    return rdd


def init_centroids(rdd, k, random=True):
    if random:
        return rdd.takeSample(False, k)
    return rdd.take(k)
    
    
def calc_distance(c, x, dist_type='euclidean'):
    if dist_type == 'euclidean':
        return np.linalg.norm(c - x)
    else:
        raise Exception('Distance Type not recognized.')
    
    
def assign_cluster(centroids, x, dist_type):
    distances = [calc_distance(c, x, dist_type=dist_type) for c in centroids]
    min_index = np.argmin(distances)
    return min_index


def redefine_clusters(rdd, centroids, dist_type):
    rdd = rdd.map(
        lambda x: (
            assign_cluster(centroids, x[1], dist_type),
            x[1]
        )
    )
    return rdd

    
def calculate_centroids(rdd):
    centroids = rdd.map(
        lambda x: (
            x[0], (
                x[1],
                1
            )
        )
    ).reduceByKey(
        lambda x, y: (
            x[0] + y[0],
            x[1] + y[1]
        )
    ).map(
        lambda x: [
            x[0],
            x[1][0] / x[1][1]
        ]
    ).collect()
    
    centroids = [el[1] for el in sorted(centroids, key=itemgetter(0))]
    return np.array(centroids)


def cluster_variance(rdd, centroids, dist_type):
    variance = rdd.map(
        lambda x: 
            calc_distance(centroids[x[0]], x[1], dist_type)
    ).reduce(
        lambda x, y:
            x + y
    )
    
    return variance


def k_means(rdd_x, k, iterations=20, tolerance=0.001, 
            n_init=5, random_init=True,
            distance_metric='euclidean', verbose=False):
    if k > rdd_x.count():
        print('Invalid K: too big!')
        return
    print('Starting: %d inits\t|\t%d partitions\n' % 
          (n_init, rdd_x.getNumPartitions()))
    init_time = time.time()
    y_arr = []
    cent_arr = []
    variance_arr = []
    
    if not random_init:
        n_init = 1
    
    for trial in range(n_init):
        start_time = time.time()

        centroids = init_centroids(rdd_x, k, random=random_init)
        rdd = init_rdd(rdd_x)
        for it in range(iterations):
            rdd = redefine_clusters(rdd, centroids, distance_metric)

            new_centroids = calculate_centroids(rdd)
            
            delta = np.linalg.norm(
                new_centroids - centroids[:len(new_centroids)]
            )
            
            if delta < tolerance:
                print('%d. Converged.\t|\t' % (trial + 1), end='')
                break
            centroids = new_centroids

            if verbose and it % 5 == 0:
                print('It: %2d\t|\tDelta: %0.4f' % (it, delta))

        y = rdd.map(
            lambda x:
                x[0]
        ).collect()
        
        total_time = time.time() - start_time
        
        y_arr.append(y)
        cent_arr.append(centroids)
        variance_arr.append(cluster_variance(rdd, centroids, distance_metric))
        print('Variance: %0.2f\t|\tTime: %0.2fs' % 
              (variance_arr[-1], total_time)
        )
    
    index = np.argmin(variance_arr)
    y = y_arr[index]
    finish_time = time.time() - init_time
    
    print('\nFinished in %0.2f s.\t|\tChosen init %d.' %
          (finish_time, (index + 1)))
    if verbose:
        [print('Cluster %d: %d' % (n, y.count(n))) for n in sorted(np.unique(y))]
        
    return y, cent_arr[index], finish_time


def speed_up_curve(time_array, n_init, save=False, interpolate=True):
    t = [time_array[0]/x for x in time_array]
    x_axis = [x for x in range(1, N_WORKERS + 1)]

    if interpolate:
        spl = make_interp_spline(x_axis, t, k=3)
        x_axis = np.linspace(1, N_WORKERS, 300)
        t = spl(x_axis)

    plt.plot(x_axis, t, linewidth=2)
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.style.use('seaborn-darkgrid')
    plt.title('Speed Up Curve')
    plt.xlabel('Workers')
    plt.ylabel('Time ratio')
    if save:
        if interpolate:
            name = 'speed_up_curve_int-'
        else:
            name = 'speed_up_curve-'
        plt.savefig(OUT_PATH + str(n_init)  + 'it-' + name + IMG_FORMAT)
    plt.show()


def performance_curve(time_array, n_init, save=False, interpolate=True):
    t = np.array(time_array.copy())
    x_axis = [x for x in range(1, N_WORKERS + 1)]

    if interpolate:
        spl = make_interp_spline(x_axis, t, k=3)
        x_axis = np.linspace(1, N_WORKERS, 300)
        t = spl(x_axis)

    plt.plot(x_axis, t, linewidth=2)
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.style.use('seaborn-darkgrid')
    plt.title('Performance Curve')
    plt.xlabel('Workers')
    plt.ylabel('Time (s)')
    if save:
        if interpolate:
            name = 'performance_curve_int.'
        else:
            name = 'performance_curve.'
        plt.savefig(OUT_PATH + str(n_init) + 'it-' + name + IMG_FORMAT)
    plt.show()


if __name__ == '__main__':

    perf = True

    if not perf:
        sc = pyspark.SparkContext('local[' + str(N_WORKERS) + ']')
        rdd_x, rdd_y = read_file(FILE_PATH)
        rdd_x = standardise(rdd_x)

        y, ctrd, tot_time = k_means(rdd_x, 3, random_init=True)

    else:

        worker_list = list(range(1, N_WORKERS + 1))
        time_list = []
        n_init = 10

        for workers in worker_list:
            print('Starting with %d workers' % (worker_list.index(workers) + 1))
            if 'sc' in locals():
                sc.stop()
            sc = pyspark.SparkContext('local[' + str(workers) + ']')

            rdd, _ = read_file(FILE_PATH, 1)
            rdd = standardise(rdd)

            _, _, tmp_time = k_means(rdd, 3, n_init=n_init, verbose=False)
            time_list.append(tmp_time)
            print('-----')
        print('Ended.')

        performance_curve(time_list, n_init, save=True, interpolate=False)

        speed_up_curve(time_list, n_init, save=True, interpolate=False)
