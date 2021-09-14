import json
import os
import sys
import getopt
import time

import instances
import results
import ttdp
import vns
import json_utils as js
import results as res
from os import listdir
from os.path import isfile, join
import copy as cp


class Configuration(object):
    def __init__(self, instances_path: str, output_path: str, file_prefix: str, file_suffix: str,
                 num_iterations: int, instance_file: str, result_file: str, json_file: str,
                 problem_type: str, vns_variant: str, no_routes: int, ns: list, max_ns_size: int,
                 ns_insert_size: int, ns_select_criteria: str, ls: str, ls_size: int, ls_improve_criteria: str,
                 ls_select_criteria: str, init_build_mod: str, init_select_criteria: str,
                 ls_max_iter: int, sk_max_iter: int, init_max_iter: int, vns_max_iter: int):

        self.instances_path = instances_path
        self.output_path = output_path
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.num_iterations = num_iterations
        self.instance_file = instance_file
        self.result_file = result_file
        self.json_file = json_file
        self.problem_type = problem_type
        self.vns_variant = vns_variant
        self.no_routes = no_routes
        self.ns = ns
        self.max_ns_size = max_ns_size
        self.ns_insert_size = ns_insert_size
        self.ns_select_criteria = ns_select_criteria
        self.ls = ls
        self.ls_size = ls_size
        self.ls_improve_criteria = ls_improve_criteria
        self.ls_select_criteria = ls_select_criteria
        self.init_build_mod = init_build_mod
        self.init_select_criteria = init_select_criteria
        self.ls_max_iter = ls_max_iter
        self.sk_max_iter = sk_max_iter
        self.init_max_iter = init_max_iter
        self.vns_max_iter = vns_max_iter

    @classmethod
    def from_config_file(cls, configuration_file):
        with open(configuration_file) as json_file:
            json_data = json.load(json_file)

            _instances_path = ''
            _output_path = ''
            _file_prefix = ''
            _file_suffix = ''
            _num_iterations = 0
            _instance_file = ''
            _result_file = ''
            _json_file = ''
            _problem_type = ''
            _vns_variant = ''
            _no_routes = 0
            _ns = []
            _max_ns_size = 0
            _ns_insert_size = 0
            _ns_select_criteria = ''
            _ls = ''
            _ls_size = 0
            _ls_improve_criteria = ''
            _ls_select_criteria = ''
            _init_build_mod = ''
            _init_select_criteria = ''
            _ls_max_iter = 0
            _sk_max_iter = 0
            _init_max_iter = 0
            _vns_max_iter = 0

            if 'instances_path' in json_data:
                _instances_path = json_data['instances_path']
            if 'output_path' in json_data:
                _output_path = json_data['output_path']
            if 'file_prefix' in json_data:
                _file_prefix = json_data['file_prefix']
            if 'file_suffix' in json_data:
                _file_suffix = json_data['file_suffix']
            if 'num_iterations' in json_data:
                _num_iterations = int(json_data['num_iterations'])
            if 'instance_file' in json_data:
                _instance_file = json_data['instance_file']
            if 'result_file' in json_data:
                _result_file = json_data['result_file']
            if 'json_file' in json_data:
                _json_file = json_data['json_file']
            if 'problem_type' in json_data:
                _problem_type = json_data['problem_type']
            if 'vns_variant' in json_data:
                _vns_variant = json_data['vns_variant']
            if 'no_routes' in json_data:
                _no_routes = int(json_data['no_routes'])
            if 'ns' in json_data:
                _ns = json_data['ns']
            if 'max_ns_size' in json_data:
                _max_ns_size = int(json_data['max_ns_size'])
            if 'ns_insert_size' in json_data:
                _ns_insert_size = int(json_data['ns_insert_size'])
            if 'ns_select_criteria' in json_data:
                _ns_select_criteria = json_data['ns_select_criteria']
            if 'ls' in json_data:
                _ls = json_data['ls']
            if 'ls_size' in json_data:
                _ls_size = int(json_data['ls_size'])
            if 'ls_improve_criteria' in json_data:
                _ls_improve_criteria = json_data['ls_improve_criteria']
            if 'ls_select_criteria' in json_data:
                _ls_select_criteria = json_data['ls_select_criteria']
            if 'init_build_mod' in json_data:
                _init_build_mod = json_data['init_build_mod']
            if 'init_select_criteria' in json_data:
                _init_select_criteria = json_data['init_select_criteria']
            if 'ls_max_iter' in json_data:
                _ls_max_iter = int(json_data['ls_max_iter'])
            if 'sk_max_iter' in json_data:
                _sk_max_iter = int(json_data['sk_max_iter'])
            if 'init_max_iter' in json_data:
                _init_max_iter = int(json_data['init_max_iter'])
            if 'vns_max_iter' in json_data:
                _vns_max_iter = int(json_data['vns_max_iter'])

        return cls(_instances_path, _output_path, _file_prefix, _file_suffix,
                   _num_iterations, _instance_file, _result_file, _json_file,
                   _problem_type, _vns_variant, _no_routes, _ns, _max_ns_size,
                   _ns_insert_size, _ns_select_criteria, _ls, _ls_size, _ls_improve_criteria,
                   _ls_select_criteria, _init_build_mod, _init_select_criteria,
                   _ls_max_iter, _sk_max_iter, _init_max_iter, _vns_max_iter)

    def to_print(self):
        print('   + instances_path = ' + self.instances_path)
        print('   + output_path = ' + self.output_path)
        print('   + file_prefix = ' + self.file_prefix)
        print('   + file_suffix = ' + self.file_suffix)
        print('   + num_iterations = ' + str(self.num_iterations))
        print('   + instance_file = ' + self.instance_file)
        print('   + result_file = ' + self.result_file)
        print('   + json_file = ' + self.json_file)
        print('   + problem_type = ' + self.problem_type)
        print('   + vns_variant = ' + self.vns_variant)
        print('   + no_routes = ' + str(self.no_routes))
        print('   + ns = ' + ', '.join([str(elem) for elem in self.ns]))
        print('   + max_ns_size = ' + str(self.max_ns_size))
        print('   + ns_insert_size = ' + str(self.ns_insert_size))
        print('   + ns_select_criteria = ' + self.ns_select_criteria)
        print('   + ls = ' + self.ls)
        print('   + ls_size = ' + str(self.ls_size))
        print('   + ls_improve_criteria = ' + self.ls_improve_criteria)
        print('   + ls_select_criteria = ' + self.ls_select_criteria)
        print('   + init_build_mod = ' + self.init_build_mod)
        print('   + init_select_criteria = ' + self.init_select_criteria)
        print('   + ls_max_iter = ' + str(self.ls_max_iter))
        print('   + sk_max_iter = ' + str(self.sk_max_iter))
        print('   + init_max_iter = ' + str(self.init_max_iter))
        print('   + vns_max_iter = ' + str(self.vns_max_iter))


def main(test_mode: str, config: Configuration):
    print('Running Test: ' + test_mode)
    config.to_print()

    if test_mode == TEST_MODE[0]:
        print("You must indicate the type of test.")
    elif test_mode == TEST_MODE[1]:
        individual_test(config)
    elif test_mode == TEST_MODE[2]:
        massive_test(config)


def test(problem_type: str, vns_variant: str, instances_path: str, output_path: str,
         instance_file: str, result_file: str, json_file: str,
         no_routes: int, ns: list, max_ns_size: int, ns_insert_size: int, ns_select_criteria: str,
         ls: str, ls_size: int, ls_improve_criteria: str, ls_select_criteria: str,
         init_build_mod: str, init_select_criteria: str,
         ls_max_iter: int, sk_max_iter: int, init_max_iter: int, vns_max_iter: int,
         is_test: bool = False):
    init_score = 0
    # Load data from file
    init_loading_time = time.time()
    # Identify instance class for file
    file_name = os.path.basename(instance_file)
    if file_name[0].isnumeric() and 'r' not in file_name and 'c' not in file_name:
        instance_type = instances.INSTANCE_TYPE[1]
        # --> 20.n.n.txt, 50.n.n.txt, 100.n.n.txt and 250.n.n.txt --> No spatial info, time info
        list_of_pois, instances_no, end_time, opt_k = instances.load_instances(problem_type, instance_type,
                                                                               instances_path + "/" + instance_file)
        # initial_node = list_of_pois[0]
        available_pois = cp.deepcopy(list_of_pois)
        initial_node = available_pois.pop(0)
        max_score = sum(p.score for p in available_pois)
        start_time = float(initial_node.opening_time)

    else:
        # --> cnnn.txt, prnn.txt, rnnn.txt and crnnn.txt --> Coord. x and y, no time info
        instance_type = instances.INSTANCE_TYPE[0]
        list_of_pois, instances_no, end_time, opt_k = instances.load_instances(problem_type, instance_type,
                                                                               instances_path + "/" + instance_file)
        initial_node = list_of_pois[0]
        available_pois = cp.deepcopy(list_of_pois)
        available_pois.pop(0)
        max_score = sum(p.score for p in available_pois)
        start_time = float(0)
        end_time = float(initial_node.closing_time)

    if instance_file[:2] == 'cr':
        instance_type = instance_type + '-cr'
    else:
        instance_type = instance_type + '-' + instance_file[0]
    end_loading_time = time.time()
    loading_time = end_loading_time - init_loading_time

    if is_test:
        # Save x,y coords in json format for plotting
        json_pois = [p.json_self_encoder() for p in list_of_pois]
        js.save_to_json_file(output_path + '/json/' + json_file + '_pois', json_pois)

    # Prepare result info
    result = res.ResultInfo(problem_type, instance_type, instance_file, no_routes,
                            ''.join(ns), max_ns_size,
                            init_build_mod,
                            ls)

    init_run = time.time()
    if vns_variant == 'BVNS':
        tour, init_score, intermediate_tours = vns.basic_vns(available_pois, initial_node, start_time, end_time,
                                                             no_routes,
                                                             ns, max_ns_size, ns_insert_size, ns_select_criteria,
                                                             ls, ls_size, ls_improve_criteria, ls_select_criteria,
                                                             init_build_mod,
                                                             ls_max_iter, sk_max_iter, init_max_iter, vns_max_iter,
                                                             is_test)
    else:
        # RVNS
        tour, init_score, intermediate_tours = vns.reduced_vns(available_pois, initial_node, start_time, end_time,
                                                               no_routes,
                                                               ns, max_ns_size, ns_insert_size, ns_select_criteria,
                                                               init_build_mod,
                                                               sk_max_iter, init_max_iter, vns_max_iter, is_test)
    end_run = time.time()
    run_time = end_run - init_run  # Run time
    if tour is not None:
        result.update_result(tour.total_size, init_score, tour.total_score, tour.total_travel_time,
                             tour.total_wait_time,
                             loading_time + run_time)
        # Write results to file
        result.append_to_csv(output_path + '/csv/' + result_file)

        print(f'Test: {problem_type} - {vns_variant} - {instance_file} - {no_routes}'
              f'       Ns: {ns} - {max_ns_size} - {ns_insert_size} - {ns_select_criteria}'
              f'       Ls: {ls} - {ls_size} - {ls_improve_criteria} - {ls} - {ls_select_criteria}'
              f'       Init: {init_build_mod} - {init_select_criteria}'
              f'       Result: {result.score} - {result.run_time}')

        if is_test:
            # Write results to json
            if intermediate_tours is not None:
                js.save_to_json_file(output_path + '/json/' + json_file, intermediate_tours)

            results.print_complete_tour(tour, 'Best')
            print(tour.check_feasible_tour())

            # js.plot_result_from_json(output_path + '/json/' + json_file + '.json',
            #                         output_path + '/json/' + json_file + '_pois.json', 1, False)


def massive_test(config: Configuration):
    # Possible Neighbourhood Structures to test
    ns_options = [vns.VNS_NS[0],  # NS for 1-route or n-route tours (Replace)
                  vns.VNS_NS[1],  # NS for n-route tours, n > 1 (Relocate)
                  vns.VNS_NS[2],  # NS for n-route tours, n > 1 (Cross-exchange)
                  vns.VNS_NS[3],  # NS for n-route tours, n > 1 (Replace + Insert)
                  vns.VNS_NS[4],  # NS for n-route tours, n > 1 (Relocate + Insert)
                  vns.VNS_NS[5]]  # ,  # NS for n-route tours, n > 1 (Cross + Insert)

    # Possible Local Search Neighbourhood to test
    ls_options = vns.LOCAL_SEARCH_NS[2:]
    ls_size = 1

    # Select criteria for insert at different steps
    ns_select_criteria = [vns.SELECT_CRITERIA[1]]
    ls_improve_criteria = [vns.LOCAL_SEARCH_MODE[0]]
    ls_select_criteria = [vns.SELECT_CRITERIA[1]]
    initial_solution_build = [vns.INIT_SEARCH_MODE[0]]

    # Fixed values
    init_s = vns.SELECT_CRITERIA[0]
    ls_i = ls_improve_criteria[0]
    ns_s = vns.SELECT_CRITERIA[1]
    ls_s = vns.SELECT_CRITERIA[1]

    ls_iter = 10
    init_iter = 10
    sk_iter = 10
    vns_iter = 100
    max_ns_size = 3

    pb = ''
    if config.problem_type != 'TOPTW':
        pb = '_td'
    pb = pb + '_' + config.file_suffix

    for i in range(1, config.num_iterations + 1):
        print(f'Iteration: {i}')
        # Load files from dir
        instance_files = [f for f in listdir(config.instances_path) if isfile(join(config.instances_path, f))]
        for f in instance_files:
            # For every instance file in path
            if f[0] == 't':  # Exclude time matrix
                continue

            if config.file_prefix != '' and f[0] != config.file_prefix[0]:  # Test only files starting with file_ref
                continue

            if config.file_suffix != '' and f[-5] != config.file_suffix:  # Test only files ending at file_ref
                continue

            # Extract reference information
            print('Extracting reference data')
            for r in range(1, 5):
                # Random init solution, insert-1 in local search, replace-1to3 in Ns
                ns_opt = [vns.VNS_NS[0]]
                ls_opt = vns.LOCAL_SEARCH_NS[2]
                init_b = vns.INIT_SEARCH_MODE[0]
                test(config.problem_type, config.vns_variant, config.instances_path, config.output_path, f,
                     config.file_prefix + '_result_ref' + pb, 'json',
                     r, ns_opt, max_ns_size, 1, ns_s,
                     ls_opt, ls_size, ls_i, ls_s,
                     init_b, init_s,
                     ls_iter, sk_iter, init_iter, vns_iter, False)

            # Extract Initial Solution data
            print('Extracting init solution data')
            for r in range(1, 5):
                # Try different initial solution algorithms
                ns_opt = [vns.VNS_NS[0]]
                ls_opt = vns.LOCAL_SEARCH_NS[2]
                for init_b in vns.INIT_SEARCH_MODE[1:]:
                    test(config.problem_type, config.vns_variant, config.instances_path, config.output_path, f,
                         config.file_prefix + '_result_init' + pb, 'json',
                         r, ns_opt, max_ns_size, 1, ns_s,
                         ls_opt, ls_size, ls_i, ls_s,
                         init_b, init_s,
                         ls_iter, sk_iter, init_iter, vns_iter, False)

            # Extract Initial Solution data
            print('Extracting ls data')
            for r in range(1, 5):
                # Try diverse local search
                ns_opt = [vns.VNS_NS[0]]
                init_b = vns.INIT_SEARCH_MODE[0]
                for ls_opt in vns.LOCAL_SEARCH_NS[3:]:
                    test(config.problem_type, config.vns_variant, config.instances_path, config.output_path, f,
                         config.file_prefix + '_result_ls' + pb, 'json',
                         r, ns_opt, max_ns_size, 1, ns_s,
                         ls_opt, ls_size, ls_i, ls_s,
                         init_b, init_s,
                         ls_iter, sk_iter, init_iter, vns_iter, False)

            # Extract Initial Solution data
            print('Extracting ns data')
            for r in range(1, 5):
                # Try different neighbourhood structures
                init_b = vns.INIT_SEARCH_MODE[0]
                ls_opt = vns.LOCAL_SEARCH_NS[2]
                for ns_opt in vns.VNS_NS[3:]:
                    if r == 1 and ns_opt != vns.VNS_NS[3]:
                        continue
                    test(config.problem_type, config.vns_variant, config.instances_path, config.output_path, f,
                         config.file_prefix + '_result_ns' + pb, 'json',
                         r, [ns_opt], max_ns_size, 1, ns_s,
                         ls_opt, ls_size, ls_i, ls_s,
                         init_b, init_s,
                         ls_iter, sk_iter, init_iter, vns_iter, False)


def individual_test(config: Configuration):
    # Individual testing

    test(config.problem_type, config.vns_variant, config.instances_path, config.output_path, config.instance_file,
         config.result_file, config.json_file, config.no_routes, config.ns, config.max_ns_size, config.ns_insert_size,
         config.ns_select_criteria, config.ls, config.ls_size, config.ls_improve_criteria, config.ls_select_criteria,
         config.init_build_mod, config.init_select_criteria, config.ls_max_iter, config.sk_max_iter,
         config.init_max_iter,
         config.vns_max_iter, True)


def reference_test():
    pois = instances.test_instances(101)
    pois[0].opening_time = float(0)
    pois[0].visit_time = float(0)
    pois[0].closing_time = float(200)
    pois[0].score = float(0)

    distances = instances.calculate_distances(pois)
    time_distances = instances.calculate_time_distance(distances, 1)

    for i in range(len(pois)):
        pois[i].distances = distances[i]
        pois[i].time_distances = time_distances[i]
        pois[i].time_matrix = time_distances[i]

    initial_poi = pois.pop(0)
    solution, init_score, remaining_points = vns.basic_vns(pois, initial_poi, initial_poi.opening_time,
                                                           initial_poi.closing_time,
                                                           2,
                                                           [vns.VNS_NS[5]], 3, 1, 'ClosestBest',
                                                           vns.LOCAL_SEARCH_NS[4], 1, 'FirstImprovement', 'ClosestBest',
                                                           'Random',
                                                           10, 10, 10, 100, True)
    for i in range(0, len(solution.tours)):
        results.plot_test_result([initial_poi] + solution.tours[i].route + [initial_poi])


def mini_reference_test():
    pois = []
    pois = pois + [ttdp.POI(0, 0, float(0.0), float(0.0), float(0.0), float(30.0), float(0.0), float(0.0))]
    pois = pois + [ttdp.POI(1, 1, float(0.0), float(1.0), float(0.0), float(10.0), float(2.0), float(8.0))]
    pois = pois + [ttdp.POI(2, 2, float(1.0), float(1.0), float(0.0), float(10.0), float(2.0), float(10.0))]
    pois = pois + [ttdp.POI(3, 3, float(1.0), float(2.0), float(0.0), float(10.0), float(2.0), float(9.0))]
    pois = pois + [ttdp.POI(4, 4, float(0.0), float(2.0), float(6.0), float(30.0), float(2.0), float(10.0))]
    pois = pois + [ttdp.POI(5, 5, float(-1.0), float(2.0), float(6.0), float(30.0), float(2.0), float(15.0))]
    pois = pois + [ttdp.POI(6, 6, float(-1.0), float(1.0), float(6.0), float(30.0), float(2.0), float(1.0))]
    pois = pois + [ttdp.POI(7, 7, float(-1.0), float(0.0), float(17.0), float(30.0), float(2.0), float(7.0))]

    distances = instances.calculate_distances(pois)
    time_distances = instances.calculate_time_distance(distances, 1)

    for i in range(len(pois)):
        pois[i].distances = distances[i]
        pois[i].time_distances = time_distances[i]
        pois[i].time_matrix = time_distances[i]

    initial_poi = pois.pop(0)

    solution, init_score, available_pois = vns.basic_vns(pois, initial_poi, float(0), float(100), 2, [vns.VNS_NS[5]], 1,
                                                         1, 'BestFirst',
                                                         vns.LOCAL_SEARCH_NS[2], 1, 'FirstImprovement', 'BestFirst',
                                                         'StochasticHillClimbing',
                                                         10, 10, 10, 20, True)
    results.print_complete_tour(solution, 'Shaked')


if __name__ == "__main__":

    TEST_MODE = ['NONE', 'INDIVIDUAL', 'MASSIVE']

    test_mode = TEST_MODE[0]
    config_file = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], "him", ["config="])
    except getopt.GetoptError:
        print('test.py -h -i -m --config=<configuration_json_file>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('test.py -h -i -m --config=<configuration_json_file>')
            sys.exit()
        elif opt == '-i':
            test_mode = TEST_MODE[1]
        elif opt == '-m':
            test_mode = TEST_MODE[2]
        elif opt == "--config":
            config_file = arg

    if test_mode == TEST_MODE[0]:
        print("You must indicate the type of test: -i for individual or -m for massive")
        print('test.py -h -i -m --config=<configuration_json_file>')
        sys.exit(2)

    if os.path.exists(config_file):
        main(test_mode, Configuration.from_config_file(config_file))
    else:
        print("Config file does not exist:" + config_file)
        sys.exit(2)
