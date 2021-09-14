import time

import results
import ttdp
import search_utils as sutil
import random
import copy as cp

SELECT_CRITERIA = ['Random', 'BestFirst', 'ClosestFirst', 'FirstBest', 'EarliestBest']
INIT_SEARCH_MODE = ['Random', 'HillClimbing', 'StochasticHillClimbing']
LOCAL_SEARCH_NS = ['Swap', '2-Opt', 'Insert', 'SwapInsert', '2-OptInsert']
LOCAL_SEARCH_MODE = ['FirstImprovement', 'BestImprovement']
VNS_NS = ['Replace', 'Relocate', 'Cross', 'ReplaceInsert', 'RelocateInsert', 'CrossInsert']


def build_initial_solution(instances: list, initial_node: ttdp.POI, routes_no: int = 1, start_time: float = 0,
                           t_max: float = 0,
                           build_mod: str = 'Random',
                           max_iterations: int = 10,
                           max_run_time: float = 300.0):
    # Build Initial Solution.
    # Search mode can be 'HillClimbing', ''Random' (INIT_SEARCH_MODE)
    if build_mod == 'HillClimbing':
        # Hill Climbing
        return hill_climbing(instances, initial_node, routes_no, start_time, t_max, max_run_time)
    elif build_mod == 'StochasticHillClimbing':
        # Stochastic Hill Climbing
        return stochastic_hill_climbing(instances, initial_node, routes_no, start_time, t_max,
                                        max_iterations, max_run_time)
    else:
        # Random, best first, first best...
        return random_solution(instances, initial_node, routes_no, start_time, t_max, 'Random',
                               max_iterations, max_run_time)


def hill_climbing(instances: list, initial_node: ttdp.POI, routes_no: int = 1, start_time: float = 0,
                  t_max: float = 0, max_run_time: float = 300.0):

    init_time = time.time()
    run_time = init_time
    current_max_run_time = max_run_time

    # First solution (random)
    solution, available_pois = random_solution(instances, initial_node,
                                               routes_no, start_time, t_max, 'BestFirst', 10, current_max_run_time)

    # Improvement
    for i in range(routes_no):

        run_time = time.time()
        current_max_run_time = max_run_time - (run_time - init_time)
        if run_time - init_time >= max_run_time:
            break

        new_tour, remaining_pois = search_insert_thorough(solution.tours[i], available_pois, 1,
                                                          0, len(solution.tours[i].route),
                                                          'FirstImprovement', current_max_run_time)
        if new_tour is not None:
            # Check eval function
            if ttdp.is_best_daily_tour1(new_tour, solution.tours[i]):
                solution.tours[i] = cp.deepcopy(new_tour)
                available_pois = cp.deepcopy(remaining_pois)

    return solution, available_pois


def stochastic_hill_climbing(instances: list, initial_node: ttdp.POI, routes_no: int = 1,
                             start_time: float = 0, t_max: float = 0,
                             max_iterations: int = 10, max_run_time: float = 300.0):

    init_time = time.time()

    # First solution (random)
    solution, available_pois = random_solution(instances, initial_node, routes_no, start_time, t_max,
                                               'BestFirst', 10, max_run_time)

    run_time = time.time()

    # Improvement
    iteration_count = 1
    while (iteration_count <= max_iterations) and (len(available_pois) > 0) and (run_time - init_time <= max_run_time):

        for i in range(routes_no):

            current_max_run_time = max_run_time - (run_time - init_time)

            new_tour, remaining_pois = search_insert_random(solution.tours[i], available_pois, 1, 0,
                                                            len(solution.tours[i].route), 'Random',
                                                            'FirstImprovement', max(5, int(max_iterations / 100)),
                                                            current_max_run_time)
            if new_tour is not None:
                # Check eval function
                if ttdp.is_best_daily_tour1(new_tour, solution.tours[i]):
                    solution.tours[i] = cp.deepcopy(new_tour)
                    available_pois = cp.deepcopy(remaining_pois)

            run_time = time.time()
            if run_time - init_time > max_run_time:
                break

        iteration_count = iteration_count + 1

    return solution, available_pois


def random_solution(instances: list, initial_node: ttdp.POI, routes_no: int = 1, start_time: float = 0,
                    t_max: float = 0, insert_mode: str = 'Random', max_iterations: int = 10,
                    max_run_time: float = 300.0):

    # Random, best first, first best...

    init_time = time.time()
    run_time = init_time

    solution = ttdp.CompleteTour(routes_no, initial_node, start_time, t_max)
    available_pois = cp.deepcopy(instances)

    iteration_count = 1

    while (iteration_count <= max_iterations) and (len(available_pois) > 0) and (run_time - init_time <= max_run_time):

        for i in range(routes_no):
            insert_pos = random.randint(0, max(0, len(solution.tours[i].route)))
            new_route, selected_pois, remaining_pois = sutil.insert_pois(solution.tours[i].route,
                                                                         solution.tours[i].initial_poi,
                                                                         solution.tours[i].initial_time,
                                                                         solution.tours[i].t_max,
                                                                         available_pois, insert_pos, 1, insert_mode)
            if len(selected_pois) > 0:
                new_route = ttdp.update_arrival_times(new_route, solution.tours[i].initial_poi, insert_pos - 1,
                                                      solution.tours[i].initial_time,
                                                      solution.tours[i].t_max)
                is_feasible, first_error_pos = ttdp.check_feasible_route(new_route, solution.tours[i].initial_poi,
                                                                         solution.tours[i].initial_time,
                                                                         solution.tours[i].t_max)
                if is_feasible:
                    solution.tours[i].route = cp.deepcopy(new_route)
                    # solution.tours[i].update_arrival_times(insert_pos - 1)
                    available_pois = cp.deepcopy(remaining_pois)

            run_time = time.time()
            if run_time - init_time > max_run_time:
                break

        iteration_count = iteration_count + 1

    return solution, available_pois


def shake(current_tour: ttdp.CompleteTour, instances: list, max_iterations: int, current_ns: str,
          size1: int, size2: int, insert_size: int, insert_type: str, max_run_time: float = 300.0):
    # Find new solution in current neighbourhood structure ns

    candidate_tour = None
    if 'Insert' in current_ns:
        insert_step_size = insert_size
    else:
        insert_step_size = 0
    if 'Replace' in current_ns:
        # Remove size1 POIs from route and replace for size2 unvisited POIs
        candidate_tour, remaining_pois = shake_replace_insert(current_tour, instances, size1, size2,
                                                              insert_step_size, insert_type,
                                                              max_iterations, max_run_time)

    elif 'Relocate' in current_ns:
        # Relocate size1 POIs from one route into another
        candidate_tour, remaining_pois = shake_relocate_insert(current_tour, instances, size1,
                                                               insert_step_size, insert_type,
                                                               max_iterations, max_run_time)

    elif 'Cross' in current_ns:
        # Exchange 2 blocks of max size size1 and size2 between routes
        candidate_tour, remaining_pois = shake_cross_insert(current_tour, instances,
                                                            size1,
                                                            size2,
                                                            insert_step_size,
                                                            insert_type,
                                                            max_iterations, max_run_time)
    if candidate_tour is not None:
        return candidate_tour, remaining_pois
    else:
        return None, None


def shake_replace_insert(current_tour: ttdp.CompleteTour, instances: list, max_remove_no: int, max_insert_no: int,
                         insert_size: int, insert_type: str,
                         max_iterations: int, max_run_time: float = 300.0):
    # Shake. Neighbourhood structure = Replace (and Insert)

    init_time = time.time()

    shaked_tour = cp.deepcopy(current_tour)
    available_pois = cp.deepcopy(instances)

    is_replaced = False
    is_inserted = False

    # Select random route order for replacement
    if len(shaked_tour.tours) > 0:
        route_order = random.sample(range(0, len(shaked_tour.tours)), len(shaked_tour.tours))
        insert_route_order = random.sample(range(0, len(shaked_tour.tours)), len(shaked_tour.tours))
    else:
        route_order = [0]
        insert_route_order = [0]

    remove_pos = []
    for i in range(0, len(shaked_tour.tours)):
        if len(shaked_tour.tours[i].route) > 0:
            if len(shaked_tour.tours[i].route) < max_remove_no:
                route_remove_pos = [0]
            else:
                route_remove_pos = random.sample(range(0, max(0, len(shaked_tour.tours[i].route) - max_remove_no + 1)),
                                                 max(1, len(shaked_tour.tours[i].route) - max_remove_no + 1))
        else:
            route_remove_pos = [0]
        remove_pos.append(route_remove_pos)

    route_no = route_order[0]
    # Select random route for insertion
    insert_route_no = insert_route_order[0]

    # Replace Step
    if not is_replaced:

        remove_no = max_remove_no
        insert_no = max_insert_no
        shaked_route = cp.deepcopy(shaked_tour.tours[route_no].route)
        if isinstance(remove_pos[route_no], list):
            pos = remove_pos[route_no][0]
            remove_pos[route_no] = remove_pos[route_no][1:] + [remove_pos[route_no][0]]
        else:
            pos = 0

        new_route, remaining_pois = sutil.replace(shaked_route, available_pois,
                                                  shaked_tour.tours[route_no].initial_poi,
                                                  shaked_tour.tours[route_no].initial_time,
                                                  shaked_tour.tours[route_no].t_max,
                                                  pos, remove_no, insert_no, insert_type)
        if new_route is not None:
            # Update route information
            new_route = ttdp.update_arrival_times(new_route, shaked_tour.tours[route_no].initial_poi,
                                                  pos - 1,
                                                  shaked_tour.tours[route_no].initial_time,
                                                  shaked_tour.tours[route_no].t_max)
            # Check constraints
            is_feasible_route, first_invalid_pos = ttdp.check_feasible_route(new_route,
                                                                             shaked_tour.tours[route_no].initial_poi,
                                                                             shaked_tour.tours[route_no].initial_time,
                                                                             shaked_tour.tours[route_no].t_max)

            if is_feasible_route:
                is_replaced = True
                shaked_tour.tours[route_no].route = cp.deepcopy(new_route)
                available_pois = cp.deepcopy(remaining_pois)
                if insert_size <= 0:
                    # No insert step, return solution
                    return shaked_tour, available_pois

    # Insertion Step
    run_time = time.time()
    if (insert_size > 0) and (run_time - init_time <= max_run_time):

        max_insert_pos = len(shaked_tour.tours[insert_route_no].route)
        insert_daily_tour, remaining_pois = shake_daily_insert(shaked_tour.tours[insert_route_no],
                                                               available_pois,
                                                               insert_size, 0, max_insert_pos, insert_type)
        if insert_daily_tour is not None:
            # POI(s) inserted, check constraints
            is_feasible_insert, first_error_pos = insert_daily_tour.check_feasible_daily_tour()
            if is_feasible_insert:
                is_inserted = True
                shaked_tour.tours[insert_route_no].route = cp.deepcopy(insert_daily_tour.route)
                available_pois = cp.deepcopy(remaining_pois)

    if is_inserted or is_replaced:
        return shaked_tour, available_pois
    else:
        return None, None


def shake_relocate_insert(current_tour: ttdp.CompleteTour, instances: list,
                          relocate_no: int, insert_size: int, insert_type: str, max_iterations: int,
                          max_run_time: float = 300.0):
    # Shake. Neighbourhood structure = Move relocate_no POIs from one route to another (and Insert)

    shaked_tour = cp.deepcopy(current_tour)
    available_pois = cp.deepcopy(instances)

    init_time = time.time()

    tour_index = [i for i in range(0, len(current_tour.tours))]

    is_relocated = False
    is_inserted = False

    # Select 2 random routes
    tour_index_list = random.sample(tour_index, k=2)
    insert_tour_no = random.choice(tour_index)
    # Select random position
    pos1 = random.randint(0, max(0, len(shaked_tour.tours[tour_index_list[0]].route) - relocate_no))
    pos2 = random.randint(0, max(0, len(shaked_tour.tours[tour_index_list[1]].route) - relocate_no))

    if pos2 is not None:
        # Relocate from route1, pos1, to route2, pos2
        new_route1, new_route2 = sutil.relocate(shaked_tour.tours[tour_index_list[0]].route,
                                                shaked_tour.tours[tour_index_list[1]].route,
                                                relocate_no,
                                                pos1, pos2)
        # Update information
        new_route1 = ttdp.update_arrival_times(new_route1, shaked_tour.initial_poi, pos1 - 1,
                                               shaked_tour.initial_time, shaked_tour.t_max)
        new_route2 = ttdp.update_arrival_times(new_route2, shaked_tour.initial_poi, pos2 - 1,
                                               shaked_tour.initial_time, shaked_tour.t_max)
        # Check Constraints
        is_feasible_route1, first_error_pos1 = ttdp.check_feasible_route(new_route1,
                                                                         shaked_tour.initial_poi,
                                                                         shaked_tour.initial_time,
                                                                         shaked_tour.t_max)
        is_feasible_route2, first_error_pos2 = ttdp.check_feasible_route(new_route2,
                                                                         shaked_tour.initial_poi,
                                                                         shaked_tour.initial_time,
                                                                         shaked_tour.t_max)
        if is_feasible_route1 and is_feasible_route2:
            is_relocated = True
            # Update routes
            shaked_tour.tours[tour_index_list[0]].route = cp.deepcopy(new_route1)
            shaked_tour.tours[tour_index_list[1]].route = cp.deepcopy(new_route2)
            if insert_size <= 0:
                return shaked_tour, available_pois

    # Insertion Step
    run_time = time.time()
    if (insert_size > 0) and (run_time - init_time <= max_run_time):

        max_insert_pos = len(shaked_tour.tours[insert_tour_no].route)
        insert_daily_tour, remaining_pois = shake_daily_insert(shaked_tour.tours[insert_tour_no],
                                                               available_pois,
                                                               insert_size, 0, max_insert_pos, insert_type)
        if insert_daily_tour is not None:
            is_feasible_insert, first_insert_error = insert_daily_tour.check_feasible_daily_tour()
            if is_feasible_insert:
                is_inserted = True
                shaked_tour.tours[insert_tour_no].route = cp.deepcopy(insert_daily_tour.route)
                available_pois = cp.deepcopy(remaining_pois)

    if is_relocated or is_inserted:
        return shaked_tour, available_pois
    else:
        return None, None


def shake_cross_insert(current_tour: ttdp.CompleteTour, instances: list,
                       max_block_size1: int, max_block_size2: int,
                       insert_size: int, insert_type: str,
                       max_iterations: int, max_run_time: float = 300.0):
    # Shake. Neighbourhood structure = Exchange subroutes (and Insert)

    init_time = time.time()

    shaked_tour = cp.deepcopy(current_tour)
    available_pois = cp.deepcopy(instances)

    tour_index = [i for i in range(0, len(current_tour.tours))]

    is_crossed = False  # No exchanged POIs yet
    is_inserted = False  # No inserted POIs yet

    # Select 2 random routes for exchange
    tour_no_list = random.sample(tour_index, k=2)
    # Select 1 random route for insert
    insert_tour_no = random.choice(tour_index)

    # Exchange/Cross Step
    if not is_crossed:
        # Select random position in route1
        block_size1 = random.randint(0, min(max_block_size1, len(shaked_tour.tours[tour_no_list[0]].route)))
        pos1 = random.randint(0, max(0, len(shaked_tour.tours[tour_no_list[0]].route) - block_size1))
        # Select random position in route2
        block_size2 = random.randint(0, min(max_block_size2, len(shaked_tour.tours[tour_no_list[1]].route)))
        pos2 = random.randint(0, max(0, len(shaked_tour.tours[tour_no_list[1]].route) - block_size2))
        # Exchange subroutes
        new_route1, new_route2 = sutil.cross(shaked_tour.tours[tour_no_list[0]].route, pos1, block_size1,
                                             shaked_tour.tours[tour_no_list[1]].route, pos2, block_size2)
        # Update route information
        new_route1 = ttdp.update_arrival_times(new_route1, shaked_tour.initial_poi, pos1 - 1,
                                               shaked_tour.initial_time, shaked_tour.t_max)
        new_route2 = ttdp.update_arrival_times(new_route2, shaked_tour.initial_poi, pos2 - 1,
                                               shaked_tour.initial_time, shaked_tour.t_max)
        # Check feasibility
        is_feasible1, first_unfeasible_pos1 = ttdp.check_feasible_route(new_route1, shaked_tour.initial_poi,
                                                                        shaked_tour.initial_time,
                                                                        shaked_tour.t_max)
        is_feasible2, first_unfeasible_pos2 = ttdp.check_feasible_route(new_route2, shaked_tour.initial_poi,
                                                                        shaked_tour.initial_time,
                                                                        shaked_tour.t_max)
        if is_feasible1 and is_feasible2:
            # Cross completed, store results
            is_crossed = True
            shaked_tour.tours[tour_no_list[0]].route = cp.deepcopy(new_route1)
            shaked_tour.tours[tour_no_list[1]].route = cp.deepcopy(new_route2)
            if insert_size <= 0:
                # No insertion, shake step finished
                return shaked_tour, available_pois

    # Insertion Step
    run_time = time.time()

    if (insert_size > 0) and (run_time - init_time <= max_run_time):
        max_insert_pos = len(shaked_tour.tours[insert_tour_no].route)
        insert_daily_tour, remaining_pois = shake_daily_insert(shaked_tour.tours[insert_tour_no],
                                                               available_pois,
                                                               insert_size, 0, max_insert_pos, insert_type)
        if insert_daily_tour is not None:
            # POI(s) inserted, check feasibility
            is_feasible_insert, first_insert_error = insert_daily_tour.check_feasible_daily_tour()
            if is_feasible_insert:
                is_inserted = True
                shaked_tour.tours[insert_tour_no].route = cp.deepcopy(insert_daily_tour.route)
                available_pois = cp.deepcopy(remaining_pois)

    if is_crossed or is_inserted:
        return shaked_tour, available_pois
    else:
        return None, None


def shake_daily_insert(current_tour: ttdp.DailyTour, instances: list,
                       insert_size: int, min_insert_pos: int, max_insert_pos: int,
                       insert_type: str):

    # Shake. Insertion Step
    insert_tour = cp.deepcopy(current_tour)
    available_pois = cp.deepcopy(instances)

    # Select random pos for insertion
    if len(insert_tour.route) <= 0:
        insert_pos = 0
    else:
        insert_pos = random.randint(min_insert_pos,
                                    min(max_insert_pos, len(insert_tour.route)))

    # Try insert
    insert_route, inserted_pois, remaining_pois = sutil.insert_pois(insert_tour.route,
                                                                    current_tour.initial_poi,
                                                                    current_tour.initial_time,
                                                                    current_tour.t_max,
                                                                    available_pois,
                                                                    insert_pos, insert_size, insert_type)
    if len(inserted_pois) > 0:
        insert_route = ttdp.update_arrival_times(insert_route, current_tour.initial_poi, insert_pos - 1,
                                                 current_tour.initial_time, current_tour.t_max)
        # Check Constraints
        feasible_route, first_error_pos = ttdp.check_feasible_route(insert_route,
                                                                    current_tour.initial_poi,
                                                                    current_tour.initial_time,
                                                                    current_tour.t_max)
        if feasible_route:
            # Update tour and return
            insert_tour.route = cp.deepcopy(insert_route)
            available_pois = cp.deepcopy(remaining_pois)
            return insert_tour, available_pois

    return None, available_pois


def local_search(current_tour: ttdp.CompleteTour, instances: list, local_ns: str,
                 size: int = 1, search_mode: str = 'FirstImprovement', insert_size: int = 1,
                 insert_type: str = 'BestFirst', max_iterations: int = 5, max_run_time: float = 300.0):
    # Local Search. Search in local neighbourhood (local_ns)

    best_tour = cp.deepcopy(current_tour)
    best_available_pois = cp.deepcopy(instances)

    insert_step_size = insert_size
    if 'Insert' not in local_ns:
        insert_step_size = 0

    # Check all daily tours
    check_tour_order = random.sample([i for i in range(0, len(current_tour.tours))], current_tour.max_routes)
    for i in check_tour_order:
        route_no = i
        candidate_tour = cp.deepcopy(current_tour)
        available_pois = cp.deepcopy(instances)
        remaining_pois = cp.deepcopy(instances)

        if 'Swap' in local_ns:
            candidate_daily_tour, remaining_pois = search_swap_insert(current_tour.tours[route_no], available_pois,
                                                                      size, insert_step_size,
                                                                      insert_type, search_mode, max_iterations,
                                                                      max_run_time)
        elif '2-Opt' in local_ns:
            candidate_daily_tour, remaining_pois = search_two_opt_insert(current_tour.tours[route_no], available_pois,
                                                                         insert_step_size, insert_type,
                                                                         search_mode, max_iterations,
                                                                         max_run_time)
        elif local_ns == 'Insert':
            candidate_daily_tour, remaining_pois = search_insert_thorough(current_tour.tours[route_no],
                                                                          available_pois,
                                                                          size,
                                                                          min_insert_pos=0,
                                                                          max_insert_pos=len(
                                                                              current_tour.tours[route_no].route) - 1,
                                                                          search_mode=search_mode,
                                                                          max_run_time=max_run_time)

        if candidate_daily_tour is not None:
            candidate_tour.tours[route_no] = cp.deepcopy(candidate_daily_tour)
            # Check eval function
            if ttdp.is_best_tour1(candidate_tour, best_tour):
                best_tour = cp.deepcopy(candidate_tour)
                best_available_pois = cp.deepcopy(remaining_pois)
                if search_mode == 'FirstImprovement':
                    return best_tour, best_available_pois

    return best_tour, best_available_pois


def search_swap_insert(current_daily_tour: ttdp.DailyTour, instances: list, block_size: int,
                       insert_size: int, insert_type: str, search_mode: str, max_iterations: int,
                       max_run_time: float = 300.0):
    # Local Search. Neighbourhood structure = Swap (block_size) and Insert (insert_size)

    init_time = time.time()
    run_time = init_time

    best_daily_tour = cp.deepcopy(current_daily_tour)
    available_pois = cp.deepcopy(instances)

    improvement = False
    # Find new solution in Local Ns
    # Swapping Step
    for j in range(0, len(current_daily_tour.route) - 2 * block_size + 1):
        for k in range(j + block_size, len(current_daily_tour.route) - block_size + 1):
            # Try swapping position j and position k, no swap_gap
            candidate_tour = cp.deepcopy(current_daily_tour)
            candidate_route = sutil.block_swap(candidate_tour.route, block_size, 0, j, k)
            # first_unfeasible_pos = None
            if candidate_route is not None:
                # print('Update after swapping')
                candidate_route = ttdp.update_arrival_times(candidate_route, candidate_tour.initial_poi, j - 1,
                                                            candidate_tour.initial_time, candidate_tour.t_max)
                candidate_tour.route = cp.deepcopy(candidate_route)
                is_feasible_tour, first_unfeasible_pos = candidate_tour.check_feasible_daily_tour()
                if is_feasible_tour:
                    if ttdp.is_best_daily_tour1(candidate_tour, best_daily_tour):
                        # Improvement
                        best_daily_tour.route = cp.deepcopy(candidate_tour.route)
                        improvement = True
                        if search_mode == 'FirstImprovement':
                            break

            run_time = time.time()
            if (run_time - init_time) > max_run_time:
                break

        run_time = time.time()
        if (run_time - init_time) > max_run_time:
            break

        if search_mode == 'FirstImprovement' and improvement:
            break

    # Inserting Step
    if insert_size > 0:
        # Try insert n
        run_time = time.time()
        current_max_run_time = max_run_time - (run_time - init_time)

        min_insert_pos = 0
        max_insert_pos = len(best_daily_tour.route)
        current_available_pois = cp.deepcopy(instances)

        if len(current_available_pois) <= len(best_daily_tour.route):
            insert_tour, remaining_pois = search_insert_thorough2(best_daily_tour, current_available_pois,
                                                                  insert_size,
                                                                  search_mode, current_max_run_time)
        else:
            insert_tour, remaining_pois = search_insert_thorough(best_daily_tour, current_available_pois,
                                                                 insert_size, min_insert_pos, max_insert_pos,
                                                                 search_mode, current_max_run_time)

        is_feasible_tour, insert_unfeasible_pos = insert_tour.check_feasible_daily_tour()
        if is_feasible_tour:
            if ttdp.is_best_daily_tour1(insert_tour, best_daily_tour):
                best_daily_tour.route = cp.deepcopy(insert_tour.route)
                available_pois = cp.deepcopy(remaining_pois)

    return best_daily_tour, available_pois


def search_two_opt_insert(current_daily_tour: ttdp.DailyTour, instances: list, insert_size: int, insert_type: str,
                          search_mode: str, max_iterations: int, max_run_time: float = 300.0):
    # Local Search. Neighbourhood structure = 2-Opt - Insert (insert-size)

    init_time = time.time()
    run_time = init_time

    best_daily_tour = cp.deepcopy(current_daily_tour)  # Best known solution so far
    available_pois = cp.deepcopy(instances)

    improvement = False
    # Find new solution in Local Ns
    for j in range(-1, len(current_daily_tour.route) - 2):  # Initial POI not in route
        for k in range(j + 2, len(current_daily_tour.route) - 2):
            # Try 2-opt at this position
            candidate_tour = cp.deepcopy(current_daily_tour)
            candidate_route = sutil.two_opt(candidate_tour.route, j, k)
            if candidate_route is not None:
                candidate_route = ttdp.update_arrival_times(candidate_route, candidate_tour.initial_poi, j - 1,
                                                            candidate_tour.initial_time, candidate_tour.t_max)
                candidate_tour.route = cp.deepcopy(candidate_route)
                is_feasible_tour, first_unfeasible_pos = candidate_tour.check_feasible_daily_tour()
                if is_feasible_tour:
                    if ttdp.is_best_daily_tour1(candidate_tour, best_daily_tour):
                        # Improvement
                        best_daily_tour.route = cp.deepcopy(candidate_tour.route)
                        improvement = True
                        if search_mode == 'FirstImprovement':
                            break
            run_time = time.time()
            if (run_time - init_time) > max_run_time:
                break

        run_time = time.time()
        if (run_time - init_time) > max_run_time:
            break

        if search_mode == 'FirstImprovement' and improvement:
            break

    # Insertion Step
    if insert_size > 0:
        # Try insert n

        run_time = time.time()
        current_max_run_time = max_run_time - (run_time - init_time)

        current_available_pois = cp.deepcopy(instances)
        min_insert_pos = 0
        max_insert_pos = len(best_daily_tour.route)
        if len(current_available_pois) >= len(best_daily_tour.route):
            insert_tour, remaining_pois = search_insert_thorough(best_daily_tour, current_available_pois,
                                                                 insert_size, min_insert_pos, max_insert_pos,
                                                                 search_mode, current_max_run_time)
        else:
            insert_tour, remaining_pois = search_insert_thorough2(best_daily_tour, current_available_pois,
                                                                  insert_size,
                                                                  search_mode, current_max_run_time)
        is_feasible_tour, insert_unfeasible_pos = insert_tour.check_feasible_daily_tour()
        if is_feasible_tour:
            if ttdp.is_best_daily_tour1(insert_tour, best_daily_tour):
                best_daily_tour.route = cp.deepcopy(insert_tour.route)
                available_pois = cp.deepcopy(remaining_pois)

    return best_daily_tour, available_pois


def search_insert_random(current_daily_tour: ttdp.DailyTour, instances: list,
                         insert_size: int, min_insert_pos: int, max_insert_pos: int, insert_type: str, search_mode: str,
                         max_iterations: int, max_run_time: float = 300.0):
    # Local Search. Neighbourhood structure = Insert

    init_time = time.time()
    run_time = init_time

    best_daily_tour = cp.deepcopy(current_daily_tour)  # Best known solution so far
    available_pois = cp.deepcopy(instances)

    # Find new solution in Local Ns
    insert_i = 1
    iteration = 1
    while (iteration <= max_iterations) and (insert_i <= insert_size) and (run_time - init_time <= max_run_time):

        iteration = iteration + 1
        for j in range(min_insert_pos, max_insert_pos + 1):
            # Try insert at position j
            candidate_tour = cp.deepcopy(current_daily_tour)
            candidate_route, inserted_pois, remaining_pois = sutil.insert_pois(current_daily_tour.route,
                                                                               current_daily_tour.initial_poi,
                                                                               current_daily_tour.initial_time,
                                                                               current_daily_tour.t_max,
                                                                               instances, j, insert_size,
                                                                               insert_type)
            if len(inserted_pois) > 0:
                candidate_route = ttdp.update_arrival_times(candidate_route, current_daily_tour.initial_poi, j - 1,
                                                            current_daily_tour.initial_time, current_daily_tour.t_max)
                candidate_tour.route = cp.deepcopy(candidate_route)
                feasible_tour, first_unfeasible_pos = ttdp.check_feasible_route(candidate_route,
                                                                                candidate_tour.initial_poi,
                                                                                candidate_tour.initial_time,
                                                                                candidate_tour.t_max)
                insert_i = insert_i + 1
                if feasible_tour:
                    if ttdp.is_best_daily_tour1(candidate_tour, best_daily_tour):
                        # Improvement
                        best_daily_tour = cp.deepcopy(candidate_tour)
                        available_pois = cp.deepcopy(remaining_pois)
                        if search_mode == 'FirstImprovement':
                            iteration = max_iterations + 1
                            break
                else:
                    if first_unfeasible_pos is not None and first_unfeasible_pos < j - 1:
                        break

            run_time = time.time()
            if run_time - init_time > max_run_time:
                break

    return best_daily_tour, available_pois


def search_insert_thorough(current_daily_tour: ttdp.DailyTour, instances: list,
                           insert_size: int,
                           min_insert_pos: int, max_insert_pos: int,
                           search_mode: str, max_run_time: float = 300.0):
    # Local Search. Neighbourhood structure = Insert; Thorough Search

    init_time = time.time()
    run_time = init_time

    best_daily_tour = cp.deepcopy(current_daily_tour)  # Best known solution so far
    best_available_pois = cp.deepcopy(instances)

    # Find new solution in Local Ns
    insert_i = 1

    while (insert_i <= insert_size) and (run_time - init_time <= max_run_time):  # and iteration <= max_iterations:

        candidate_tour = cp.deepcopy(best_daily_tour)  # Insert sobre el mejor previo
        available_pois = cp.deepcopy(best_available_pois)

        min_insert = max(0, min_insert_pos)
        max_insert = min(max_insert_pos, len(candidate_tour.route) + 1)

        improvement = False

        for i in range(min_insert, max_insert + 1):

            if i == 0:
                # Insert at the beginning
                previous_poi = current_daily_tour.initial_poi
            else:
                previous_poi = candidate_tour.route[i - 1]

            if i == len(candidate_tour.route):
                next_poi = current_daily_tour.initial_poi
            else:
                next_poi = candidate_tour.route[i]

            # Filter potential pois for this position
            potential_pois = ttdp.filter_valid_from_to(available_pois, previous_poi, next_poi,
                                                       candidate_tour.initial_time, candidate_tour.t_max)
            if len(potential_pois) > 0:
                for j in range(0, len(potential_pois)):
                    insert_route = cp.deepcopy(candidate_tour.route)
                    insert_route.insert(i, potential_pois[j])
                    insert_route = ttdp.update_arrival_times(insert_route, candidate_tour.initial_poi, i - 1,
                                                             candidate_tour.initial_time, candidate_tour.t_max)
                    is_feasible_insert, first_error_insert = ttdp.check_feasible_route(insert_route,
                                                                                       candidate_tour.initial_poi,
                                                                                       candidate_tour.initial_time,
                                                                                       candidate_tour.t_max)
                    if is_feasible_insert:
                        insert_tour = cp.deepcopy(candidate_tour)
                        insert_tour.route = cp.deepcopy(insert_route)
                        if ttdp.is_best_daily_tour1(insert_tour, best_daily_tour):
                            best_daily_tour = cp.deepcopy(insert_tour)
                            best_available_pois = [p for p in available_pois if
                                                   (p.poi_index not in [s.poi_index for s in insert_tour.route])]
                            if search_mode == 'FirstImprovement':
                                improvement = True
                                break

                    run_time = time.time()
                    if run_time - init_time > max_run_time:
                        break

            if improvement:
                insert_i = insert_i + 1
                if search_mode == 'FirstImprovement':
                    break

            run_time = time.time()
            if run_time - init_time > max_run_time:
                break

        if not improvement:
            # Every position was tried with every available POI and no improvement made: exit
            break

    return best_daily_tour, best_available_pois


def search_insert_thorough2(current_daily_tour: ttdp.DailyTour, instances: list,
                            insert_size: int,
                            search_mode: str,
                            max_run_time: float = 300.0):
    # Local Search. Neighbourhood structure = Insert; Thorough Search

    init_time = time.time()
    run_time = init_time

    best_daily_tour = cp.deepcopy(current_daily_tour)  # Best known solution so far
    best_available_pois = cp.deepcopy(instances)

    # Find new solution in Local Ns
    insert_i = 1

    while (insert_i <= insert_size) and (run_time - init_time <= max_run_time):

        candidate_tour = cp.deepcopy(best_daily_tour)  # Best accepted solution
        available_pois = cp.deepcopy(best_available_pois)
        available_pois.sort(key=lambda p: p.max_arrival_time)

        improvement = False

        for i in range(0, len(available_pois)):

            route_pois = cp.deepcopy(candidate_tour.route)
            route_pois.insert(0, candidate_tour.initial_poi)  # Include initial POI in position testing
            min_insert_pos, max_insert_pos = ttdp.get_potential_insert_range(route_pois, available_pois[i])

            if max_insert_pos < min_insert_pos:
                # No insertion possible for this candidate poi
                break

            for j in range(min_insert_pos, max_insert_pos + 1):
                insert_route = cp.deepcopy(candidate_tour.route)
                insert_route.insert(j, available_pois[i])
                insert_route = ttdp.update_arrival_times(insert_route, candidate_tour.initial_poi, j - 1,
                                                         candidate_tour.initial_time, candidate_tour.t_max)
                is_feasible_insert, first_error_insert = ttdp.check_feasible_route(insert_route,
                                                                                   candidate_tour.initial_poi,
                                                                                   candidate_tour.initial_time,
                                                                                   candidate_tour.t_max)
                if is_feasible_insert:
                    insert_tour = cp.deepcopy(candidate_tour)
                    insert_tour.route = cp.deepcopy(insert_route)
                    if ttdp.is_best_daily_tour1(insert_tour, best_daily_tour):
                        best_daily_tour = cp.deepcopy(insert_tour)
                        best_available_pois = [p for p in available_pois if
                                               (p.poi_index not in [s.poi_index for s in insert_tour.route])]
                        improvement = True
                        if search_mode == 'FirstImprovement':
                            break
                if improvement:
                    insert_i = insert_i + 1
                    if search_mode == 'FirstImprovement':
                        break

                run_time = time.time()
                if run_time - init_time > max_run_time:
                    break

            run_time = time.time()
            if run_time - init_time > max_run_time:
                break

        if not improvement:
            # No possible insertion found
            break

    return best_daily_tour, best_available_pois


def neighbourhood_change(current_best_tour: ttdp.CompleteTour, current_available_pois: list,
                         new_tour: ttdp.CompleteTour, new_available_pois: list,
                         ind_ns: int, max_size: int, current_size: int):
    if ttdp.is_best_tour1(new_tour, current_best_tour):
        # Improvement
        new_best_tour = cp.deepcopy(new_tour)
        new_best_available_pois = cp.deepcopy(new_available_pois)
        # Initial neighbourhood
        new_ind_ns = 0
        new_size = 1
    else:
        # No improvement, change neighbourhood
        new_best_tour = cp.deepcopy(current_best_tour)
        new_best_available_pois = cp.deepcopy(current_available_pois)
        if current_size < max_size:
            new_ind_ns = ind_ns  # Keep method, change size
            new_size = current_size + 1
        else:
            new_ind_ns = ind_ns + 1  # New method (if available)
            new_size = 1  # Start

    return new_best_tour, new_best_available_pois, new_ind_ns, new_size


def basic_vns(instances: list, init_poi: ttdp.POI, start_time: float, max_time: float,
              num_routes: int,
              ns: list, max_ns_size: int, ns_insert_size: int, ns_select_criteria: str,
              ls_neighbourhood: str, ls_size: int, ls_improve_criteria: str, ls_select_criteria: str,
              init_build_mode: str,
              ls_max_iterations: int, sk_max_iterations: int, init_max_iterations: int, vns_max_iterations: int,
              test_mode: bool = False):

    list_of_tours = []  # List of solutions (if plotting required)

    max_run_time = 300  # Tmax = 300 sec = 5 min
    init_time = time.time()

    # Find initial solution
    best_tour, available_pois = build_initial_solution(instances, init_poi, num_routes, start_time, max_time,
                                                       init_build_mode, init_max_iterations, max_run_time)

    init_score = best_tour.total_score

    if test_mode:
        results.print_complete_tour(best_tour, 'Initial', False)
        # Add initial solution to follow-up list
        list_of_tours.append(ttdp.IntermediateSolution("initial", best_tour))

    no_improvement_count = 0
    max_no_improvement_count = max(30, int(vns_max_iterations / 3))

    iter_no = 1

    run_time = init_time

    # Improve solution
    while (iter_no <= vns_max_iterations) \
            and (run_time - init_time <= max_run_time) \
            and (no_improvement_count < max_no_improvement_count):

        # Initial neighbourhood
        # Neighbourhood change might be a combination of type (replace, relocate, cross) and size
        ind_ns = 0
        ns_change_size = 1

        while ind_ns <= (len(ns) - 1) \
                and (run_time - init_time <= max_run_time):

            run_time = time.time()
            current_max_run_time = max_run_time - (run_time - init_time)

            # Shake
            shaked_tour, remaining_pois = shake(best_tour, available_pois, sk_max_iterations, ns[ind_ns],
                                                ns_change_size, ns_change_size, ns_insert_size, ns_select_criteria,
                                                current_max_run_time)
            if test_mode and shaked_tour is not None:
                results.print_complete_tour(shaked_tour, 'Shaked', False)
                # Add perturbation solution to follow-up list
                list_of_tours.append(ttdp.IntermediateSolution("shake", shaked_tour))

            # Local Search
            if shaked_tour is not None:

                run_time = time.time()
                current_max_run_time = max_run_time - (run_time - init_time)

                new_tour, remaining_pois = local_search(shaked_tour, remaining_pois,
                                                        ls_neighbourhood, ls_size, ls_improve_criteria,
                                                        1, ls_select_criteria,
                                                        ls_max_iterations,
                                                        current_max_run_time)
            else:
                new_tour = cp.deepcopy(best_tour)
                remaining_pois = cp.deepcopy(available_pois)
            if test_mode and shaked_tour is not None:
                results.print_complete_tour(new_tour, 'Local', False)
                list_of_tours.append(ttdp.IntermediateSolution("local_search", new_tour))

            # Mover or not
            best_tour, available_pois, ind_ns, ns_change_size = neighbourhood_change(best_tour, available_pois,
                                                                                     new_tour, remaining_pois,
                                                                                     ind_ns, max_ns_size,
                                                                                     ns_change_size)

            if ind_ns == 0 and ns_change_size == 1:
                # Neighbourhood restart --> improvement
                no_improvement_count = 0
            else:
                no_improvement_count = no_improvement_count + 1

        iter_no = iter_no + 1
        run_time = time.time()

    if test_mode:
        return best_tour, init_score, list_of_tours
    else:
        return best_tour, init_score, None


def reduced_vns(instances: list, init_poi: ttdp.POI, start_time: float, max_time: float,
                num_routes: int,
                ns: list, max_ns_size: int, ns_insert_size: int, ns_select_criteria: str,
                init_build_mode: str,
                sk_max_iterations: int, init_max_iterations, vns_max_iterations: int,
                test_mode: bool = False):
    list_of_tours = []  # List of solutions (if plotting required)

    max_run_time = 300  # Max 5 min
    init_time = time.time()
    run_time = init_time

    # Find initial solution
    best_tour, available_pois = build_initial_solution(instances, init_poi, num_routes, start_time, max_time,
                                                       init_build_mode, init_max_iterations, max_run_time)

    init_score = best_tour.total_score

    if test_mode:
        results.print_complete_tour(best_tour, 'Initial', False)
        # Add initial solution to follow-up list
        list_of_tours.append(ttdp.IntermediateSolution("initial", best_tour))

    no_improvement_count = 0
    max_no_improvement_count = max(30, int(vns_max_iterations / 3))

    iter_no = 1

    # Improve solution
    while (iter_no <= vns_max_iterations) \
            and (run_time - init_time <= max_run_time) \
            and (no_improvement_count < max_no_improvement_count):

        # Initial neighbourhood
        # Neighbourhood change might be a combination of type (replace, relocate, cross) and size
        ind_ns = 0
        ns_change_size = 1

        while ind_ns <= (len(ns) - 1) \
                and (run_time - init_time <= max_run_time):

            run_time = time.time()
            current_max_run_time = max_run_time - (run_time - init_time)

            # Shake
            shaked_tour, remaining_pois = shake(best_tour, available_pois, sk_max_iterations, ns[ind_ns],
                                                ns_change_size, ns_change_size, ns_insert_size, ns_select_criteria,
                                                current_max_run_time)

            if test_mode:
                results.print_complete_tour(shaked_tour, 'Shaked', False)
                # Add perturbation solution to follow-up list
                list_of_tours.append(ttdp.IntermediateSolution("shake", shaked_tour))

            # Neighbourhood change
            best_tour, available_pois, ind_ns, ns_change_size = neighbourhood_change(best_tour, available_pois,
                                                                                     shaked_tour, remaining_pois,
                                                                                     ind_ns, max_ns_size,
                                                                                     ns_change_size)

            if ind_ns == 0 and ns_change_size == 1:
                # Neighbourhood restart --> improvement
                no_improvement_count = 0
            else:
                no_improvement_count = no_improvement_count + 1

            run_time = time.time()

        iter_no = iter_no + 1

    if test_mode:
        return best_tour, init_score, list_of_tours
    else:
        return best_tour, init_score, None
