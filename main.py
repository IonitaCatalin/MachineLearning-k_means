import csv
import os
import random
import math
import shutil
import sys
import time
import numpy
import matplotlib.pyplot as plt
from copy import deepcopy


def parse_csv_data(path):
    try:
        csv_file = open(path, 'r')
        csv_reader = csv.reader(csv_file, delimiter=',')
        points = list()
        for row in csv_reader:
            points.append([float(i) for i in row])
        return points
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        exit(1)


def init_k_centroids(k_value, dataset):
    maximums = list()
    minimums = list()
    for index in range(0, len(dataset[0])):
        maximums.append(max([element[index] for element in dataset]))
        minimums.append(min([element[index] for element in dataset]))

    centroids = [[] for i in range(0, k)]

    for i in range(0, k_value):
        for j in range(0, len(dataset[0])):
            centroids[i].append(round(random.uniform(minimums[j], maximums[j]), 2))

    return centroids


def init_k_existing_centroids(k_value, dataset):
    return [dataset[random.randrange(0, len(dataset))] for i in range(0, k_value)]


def init_k_plus_centroids(k_value, dataset):
    centroids = list()
    centroids.append(dataset[random.randrange(0, len(dataset))])
    for i in range(k - 1):
        distances = []
        for j in range(len(dataset)):
            point = dataset[j]
            distance = sys.maxsize
            for j in range(len(centroids)):
                temporary = euclidean_distance(point, centroids[j])
                distance = min(distance, temporary)
            distances.append(distance)
        centroids.append(dataset[distances.index(max(distances))])

    return centroids


def init_k1_centroids(k_value, dataset):
    return [[5.33, 2.0], [8.0, 3.0], [16.0, 6.0]]


def init_k2_centroids(k_value, dataset):
    return [[4.0, 6.0], [4.0, 0.0], [16.0, 3.0]]


def init_k3_centroids(k_value, dataset):
    return [[0.0, 0.0], [8.0, 3.0], [16.0, 3.0]]


def init_k4_centroids(k_value, dataset):
    return [[4.0, 0.0], [0.0, 6.0], [14.0, 4.0]]


def init_k5_centroids(k_value, dataset):
    return [[2.0, 2.0], [12.0, 6.0], [16.0, 0.0]]


def update_centroids(centroids, clustered_items):
    for count, item in enumerate(clustered_items):
        if item:
            centroids[count] = [round(sum(i) / len(item), 2) for i in zip(*item)]


def euclidean_distance(p, q):
    sum_of_sq = 0
    for i in range(len(p)):
        sum_of_sq += (p[i] - q[i]) ** 2
    return math.sqrt(sum_of_sq)


def compute_J(clusters, centroids):
    value = 0
    for count, cluster in enumerate(clusters):
        difference = 0
        for item in cluster:
            difference = difference + sum([item[i] - centroids[count][i] for i in range(0, len(item))])
        value = value + difference * difference
    return value


def export_iteration(iteration, clusters, previous_centroids, current_centroids, colors):
    plt.title('Iteration:' + str(iteration) + " J:" + str(compute_J(clusters, current_centroids)))
    for count, item in enumerate(clusters):
        plt.scatter([i[0] for i in item], [i[1] for i in item], color=colors[count], label='Cluster ' + str(count))
        # plt.legend(loc="lower right")
    centroid_x = [i[0] for i in current_centroids]
    centroid_y = [i[1] for i in current_centroids]

    for i in range(0, len(current_centroids)):
        plt.scatter(centroid_x[i], centroid_y[i], color=colors[i], marker='x')
    # if iteration > 0:
    #     previous_centroid_x = [i[0] for i in previous_centroids]
    #     previous_centroid_y = [i[1] for i in previous_centroids]
    #     for i in range(0, len(current_centroids)):
    #         plt.scatter(previous_centroid_x[i], previous_centroid_y[i], color=colors[i], marker='x', alpha=1.0)
    #     for i in range(0, len(current_centroids)):
    #         plt.arrow(previous_centroid_x[i], previous_centroid_y[i],
    #                   centroid_x[i] - previous_centroid_x[i],
    #                   centroid_y[i] - previous_centroid_y[i], head_width=0.03,
    #                   head_length=0.03, linestyle='--', color=colors[i])

    plt.savefig('execution\\' + str(iteration) + '.png')
    plt.clf()


def export_j_values(values):
    print(values)
    plt.title('Variation of J-value for #' + str(len(values)) + ' iterations')
    plt.xlabel('J')
    plt.ylabel('Iteration')
    plt.plot(values, [str(i) for i in range(1, len(values) + 1)])
    plt.savefig('execution\\j_values.png')


def k_means_algorithm(k_value, dataset, max_iterations, distance_function, init_centroids_function):
    previous_centroids = None
    current_iteration = 0
    j_values = list()
    centroids = init_centroids_function(k_value, dataset)
    colors = [numpy.random.random(3) for i in range(k_value)]

    while current_iteration in range(0, max_iterations) and previous_centroids != centroids:
        clusters = classify(centroids, dataset, distance_function)
        print('Iterations:', current_iteration)
        print('Clusters:', *clusters, sep='\n')
        export_iteration(current_iteration, clusters, previous_centroids, centroids, colors)
        j_values.append(compute_J(clusters, centroids))
        previous_centroids = deepcopy(centroids)
        update_centroids(centroids, clusters)
        print('Current centroid:', centroids)
        current_iteration = current_iteration + 1
    export_j_values(j_values)


def delete_prev_execution():
    folder = 'execution'
    for filename in os.listdir('execution'):
        path = os.path.join(folder, filename)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (path, e))
            exit(1)


def classify(centroids, items, distance_function):
    clusters = [[] for i in range(len(centroids))]
    for item in items:
        minimum = sys.maxsize
        cluster = -1
        for count, centroid in enumerate(centroids):
            distance = distance_function(item, centroid)
            if distance < minimum:
                minimum = distance
                cluster = count
        clusters[cluster].append(item)
    return clusters


if __name__ == '__main__':
    initialization = str(sys.argv[1])
    k = int(sys.argv[2])
    file_path = str(sys.argv[3])
    iterations = int(sys.argv[4])

    data = parse_csv_data(file_path)
    delete_prev_execution()
    start_time = time.time()

    if initialization == 'simple':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k_existing_centroids)
    elif initialization == 'random':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k_centroids)
    elif initialization == 'plus':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k_plus_centroids)
    elif initialization == 'k1':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k1_centroids)
    elif initialization == 'k2':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k2_centroids)
    elif initialization == 'k3':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k3_centroids)
    elif initialization == 'k4':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k4_centroids)
    elif initialization == 'k5':
        k_means_algorithm(k, data, iterations, euclidean_distance, init_k5_centroids)
    else:
        print('Unknown format for initialization of the centroids')
        exit(1)
    print("--- Program ended in:%s seconds ---" % round(time.time() - start_time, 2))
