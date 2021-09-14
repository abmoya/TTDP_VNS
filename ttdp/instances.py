import csv
from random import randint
import os
import numpy as np
import ttdp

INSTANCE_TYPE = ['Solomon', 'Verbeck']


def load_instances(problem_type: str, instance_type: str, file_name: str):
    # Load POI data from file for TOPTW problem

    if instance_type == 'Solomon':
        return load_instances_type_s(problem_type, file_name)
    elif instance_type == 'Verbeck':
        return load_instances_type_v(problem_type, file_name)


def load_instances_type_s(problem_type: str, file_name: str):
    available_pois = []
    opt_k = 0
    max_score = 0
    with open(file_name, mode='r', newline='') as input_file:
        line_count = 0
        for input_line in input_file:
            input_split = input_line.split()
            if len(input_split) <= 0:
                continue
            if line_count == 0:
                # First line contains general info
                opt_k = int(input_split[1])  # With this number or routes all POIs can be visited
            if line_count >= 2:
                init_col = 0
                if not input_split[0].isnumeric():
                    init_col = 1
                # POI data
                index = int(input_split[0 + init_col])
                x = float(input_split[1 + init_col])
                y = float(input_split[2 + init_col])
                visit = float(input_split[3 + init_col])
                score = float(input_split[4 + init_col])
                max_score = max_score + score
                if line_count == 2:  # Init/End POI
                    opening = 0
                else:
                    opening = float(input_split[len(input_split) - 2])
                closing = float(input_split[-1]) + visit
                available_pois.append(ttdp.POI(index, None, x, y, opening, closing, visit, score))
            line_count += 1

    input_file.close()
    t_max = available_pois[0].closing_time
    no_pois = len(available_pois) - 1

    distances = calculate_distances(available_pois)
    time_distances = calculate_time_distance(distances, 1)
    time_matrix = None
    if problem_type == 'TDTOPTW':
        # Simulate variable time
        # First element contains max time; init time is always 0
        time_matrix = simulate_time_matrix(time_distances, 20)

    for i in range(len(available_pois)):
        available_pois[i].distances = distances[i]
        available_pois[i].time_distances = time_distances[i]
        if problem_type == 'TDTOPTW':
            available_pois[i].time_matrix = time_matrix[i]
        else:
            available_pois[i].time_matrix = time_distances[i]

    available_pois = ttdp.filter_time_windows_error(available_pois)

    return available_pois, no_pois, t_max, opt_k


def load_instances_type_v(problem_type: str, file_name: str):
    available_pois = []

    opt_k = None
    no_pois = 0
    t_max = 0

    f_name = os.path.basename(file_name)
    dir_name = os.path.dirname(file_name)

    if problem_type == 'TOPTW':
        # Load time independent data
        f_time = dir_name + '/titt' + f_name.split('.')[0] + '.txt'
        time_distances = load_time_data(f_time)
    else:
        # Load time-dependent data
        f_time = dir_name + '/tt' + f_name.split('.')[0] + '.txt'
        time_matrix = load_time_dependent_data(f_time)
    index = 0

    with open(file_name, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';', quoting=csv.QUOTE_NONE)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                no_pois = int(row[0]) - 1  # Number of instances
            elif line_count == 1:
                # First line contains general info
                t_max = float(row[0])  # Maximum total travel time
            elif line_count >= 2:
                # POI data
                poi_no = int(row[0])
                score = float(row[1])
                visit = float(row[3])
                opening = float(row[4])
                closing = float(row[5])
                available_pois.append(ttdp.POI(index, poi_no=poi_no, x_coord=None, y_coord=None,
                                               open_time=opening, close_time=closing, visit_time=visit, score=score))
                index = index + 1
            line_count += 1

    t_max = t_max + available_pois[0].opening_time

    for i in range(len(available_pois)):
        if problem_type == 'TOPTW':
            available_pois[i].time_matrix = time_distances[i]
        else:
            available_pois[i].time_matrix = time_matrix[i]

    available_pois = ttdp.filter_time_windows_error(available_pois)

    return available_pois, no_pois, t_max, opt_k


def calculate_distances(list_of_pois: list):
    n = len(list_of_pois)
    distance = [[list_of_pois[i].distance(list_of_pois[j]) for i in range(n)] for j in range(n)]
    return distance


# Calculate travel time between pois from distance matrix
def calculate_time_distance(distances: list, speed: float = 1):
    n = len(distances)
    times = [[round(distances[i][j] / speed, 2) for j in range(n)] for i in range(n)]
    return times


def load_time_data(file_name: str):
    # Load POI data from file for TOPTW problem

    with open(file_name, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        time_matrix = list(csv_reader)
    csv_file.close()
    return time_matrix


# Create a new (m x n) matrix where each element is an array of time cost values created from input matrix
#       time_matrix: input matrix (m x n) - each element is a time cost value
#       time_range: size of time cost distribution array
#       delta (optional): adjustment value for range [0,1] - default value = 0.5
def simulate_time_matrix(time_matrix, time_range: int, delta: float = 0.5):
    if delta < 0:
        delta = 0
        print("## WARNING: delta parameter out of Range. Set to 0. ")

    if delta > 1:
        delta = 1
        print("## WARNING: delta parameter out of Range. Set to 1. ")

    if time_range <= 0:
        time_range = 1
        print("## WARNING: range parameter out of Range. Set to 1. ")

    new_matrix = []

    for lst in time_matrix:
        new_list = []
        for element in lst:
            # Apply adjustment to value
            delta_element = abs(element * delta)
            # Generate time distribution
            new_element = np.random.default_rng().uniform(element - delta_element,
                                                          element + delta_element, time_range).tolist()
            new_list.append(new_element)
        new_matrix.append(new_list)

    return new_matrix


def test_instances(num_pois: int):
    pois = []

    delta_x = 0
    delta_y = 0

    theta = np.linspace(0, 2 * np.pi, num_pois)
    radius = 20

    a = radius * np.sin(theta)
    b = radius * (1 - np.cos(theta))

    coordinate_lst = []
    for i, val in enumerate(a):
        coordinate = (round(a[i]) + randint(-delta_x, delta_x), round(b[i]) + randint(-delta_y, delta_y))
        if coordinate not in coordinate_lst:  # Avoid duplicates
            coordinate_lst.append(coordinate)

    i = 0
    for coordinate in coordinate_lst:
        if i < round(num_pois / 2):  # Early time windows
            opening_time = float(0.0)
            closing_time = float(100.0)
        else:  # Late time windows
            opening_time = float(100.0)
            closing_time = float(200.0)

        pois.append(ttdp.POI(i, i, float(coordinate[0]), float(coordinate[1]), opening_time, closing_time,
                             float(randint(1, 10)),
                             float(randint(1, 20))))
        i = i + 1

    return pois


def load_time_dependent_data(file_name: str, file_prefix: str = "tt", max_data_array: int = 56):

    f_name = os.path.basename(file_name)
    max_columns_array = int(f_name.replace(file_prefix, "").split(".")[0])

    data_raw = np.genfromtxt(file_name, delimiter=';', dtype=int)

    matrix = []
    data_array = []
    row_array = []

    new_data_array = False
    new_row_array = False

    for element in data_raw:
        if new_data_array:
            data_array = []
            new_data_array = False

        if new_row_array:
            row_array = []
            new_row_array = False

        data_array.append(element)

        if len(data_array) == max_data_array:
            new_data_array = True
            row_array.append(data_array)

        if len(row_array) == max_columns_array:
            new_row_array = True
            matrix.append(row_array)

    return matrix
