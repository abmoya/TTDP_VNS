import random
import ttdp
import copy as cp


def insert_pois(route: list, initial_poi: ttdp.POI, t_min: float, t_max: float, available_pois: list,
                pos: int = -1, insert_size: int = 1, selection_mode: str = 'Random'):
    # Attempts to insert a POI into a tour in position after_index + 1
    # Returns new tour if valid, None otherwise

    new_route = cp.deepcopy(route)
    remaining_pois = cp.deepcopy(available_pois)

    if pos > (len(new_route) - 1) or (pos == -1):
        new_index = len(new_route)  # Insert at the end
    else:
        new_index = pos  # Insert

    if new_index == 0:
        # Insert at the beginning
        previous_poi = initial_poi
    else:
        previous_poi = route[new_index - 1]

    if new_index == len(route):
        next_poi = initial_poi
    else:
        next_poi = route[new_index]

    potential_pois = ttdp.filter_valid_from_to(remaining_pois, previous_poi, next_poi, t_min, t_max)

    selected_pois = []

    if len(potential_pois) > 0:
        selected_pois = select_pois(potential_pois, selection_mode, insert_size)
        if len(selected_pois) > 0:
            new_route[new_index:new_index] = selected_pois
            remaining_pois = [p for p in available_pois if (p.poi_index not in [s.poi_index for s in selected_pois])]

    return new_route, selected_pois, remaining_pois


def select_pois(available_pois: list, select_mode: str = 'Random',
                select_size: int = 1):
    # Selects n (select_size) available POIs for insertion
    # Selection will depend on chosen mode (random by default)
    # Returns POI if selected, None otherwise

    if select_mode == 'Random':
        if len(available_pois) >= select_size:
            return random.sample(available_pois, k=select_size)
        else:
            return available_pois
    else:
        # Sorting required
        sorted_pois = cp.deepcopy(available_pois)
        if select_mode == 'BestFirst':
            sorted_pois.sort(key=lambda p: p.score, reverse=True)
        elif select_mode == 'ClosestBest':
            sorted_pois.sort(key=lambda p: p.arrival_time)
        elif select_mode == 'EarliestBest':
            sorted_pois.sort(key=lambda p: p.opening_time)
        return sorted_pois[:select_size]


def block_swap(route: list, block_size=1, swap_gap=0, index1=-1, index2=-1):
    # Swap 2 blocks of size block_size in index1 and index2
    # A gap of at least swap-gap between swapped blocks is to be respected
    # If index = -1, random index is selected

    max_index1 = len(route) - ((2 * block_size) + swap_gap)
    max_index2 = len(route) - block_size

    if max_index2 - max_index1 < block_size + swap_gap:
        return None
    from_index1 = index1
    from_index2 = index2
    if from_index1 == -1:
        if max_index1 < 0:
            # Swapping not possible
            from_index1 = 0
        else:
            # Random swapping
            from_index1 = random.randint(0, max_index1)
    if index2 == -1:
        if max_index2 < (max_index1 + block_size + swap_gap):
            # Swapping not possible
            from_index2 = max_index2 + 1
        else:
            # Random swapping
            from_index2 = random.randint(index1 + block_size + swap_gap, max_index2)

    if from_index1 <= max_index1 and from_index2 <= max_index2:
        new_route = cp.deepcopy(route)
        new_route[from_index1:from_index1 + block_size], new_route[from_index2:from_index2 + block_size] = \
            new_route[from_index2:from_index2 + block_size], new_route[from_index1:from_index1 + block_size]
        return new_route


def block_reverse(route: list, block_size=2, index=0):
    new_route = cp.deepcopy(route)
    if block_size <= len(new_route):
        new_route = new_route[: index] + new_route[index: index + block_size][::-1] + \
                    new_route[index + block_size:]
    else:
        new_route.reverse()
    return new_route


def route_reverse(route: list):
    new_route = cp.deepcopy(route)
    new_route.reverse()
    return new_route


def two_opt(route: list, index1: int, index2: int):
    # 2-opt movement with arcs index1, index1 +1 and index2, index2 + 1
    # Index1 and index2 range from 0 to len(route) - 1
    # If index2 = index
    # Note: Initial/End POI are not included in route

    if index1 + 1 < index2 < len(route):
        new_route = route[:index1 + 1] + \
                    route[index1 + 1:index2 + 1][::-1] + \
                    route[index2 + 1:]
        return new_route


def replace(route: list, available_pois: list, initial_poi: ttdp.POI, t_min: float, t_max: float,
            pos: int, remove_no: int, insert_no: int, selection_mode: str = 'Random'):
    # Remove POI at position pos and insert another POI from index at that position

    if 0 <= pos <= len(route) - 1:
        # Remove POI from route
        removed_route = cp.deepcopy(route)
        removed_pois = removed_route[pos:pos + remove_no]
        removed_route = removed_route[:pos] + removed_route[pos + remove_no:]
        # Insert POI from available
        insert_route, inserted_pois, remaining_pois = insert_pois(removed_route, initial_poi,
                                                                  t_min, t_max, available_pois,
                                                                  pos, insert_no, selection_mode)
        remaining_pois.extend(removed_pois)
        # If POI inserted, update remaining POIs and route
        if inserted_pois is not None:
            new_route = cp.deepcopy(insert_route)
            return new_route, remaining_pois
        else:
            return removed_route, remaining_pois
    else:
        return None, available_pois


def relocate(route1: list, route2: list, relocate_no: int, remove_index: int, insert_index: int):
    # Move one POI from route 1, at position remove_index, to route 2, at position insert_index

    new_route1 = cp.deepcopy(route1)
    new_route2 = cp.deepcopy(route2)
    # Relocate POIs
    new_route2 = new_route2[:insert_index] + new_route1[remove_index:remove_index + relocate_no] + \
                 new_route2[insert_index:]
    new_route1 = new_route1[:remove_index] + new_route1[remove_index + relocate_no:]

    return new_route1, new_route2


def cross(route1: list, index1: int, block_size1: int, route2: list, index2: int, block_size2: int):
    # Swap a sequence of POIs of size block_size1 in route 1 with a sequence of POIs of block_size2 in route2

    # Check block_size1, index1
    bs1 = block_size1
    if bs1 > len(route1):
        bs1 = len(route1)
    max_index1 = len(route1) - bs1
    ind1 = index1
    if index1 > max_index1:
        ind1 = 0
    # Check block_size2, index2
    bs2 = block_size2
    if bs2 > len(route2):
        bs2 = len(route2)
    max_index2 = len(route2) - bs2
    ind2 = index2
    if index2 > max_index2:
        ind2 = 0
    # Swap blocks
    new_route1 = route1[:ind1] + route2[ind2: ind2 + bs2] + route1[ind1 + bs1:]
    new_route2 = route2[:ind2] + route1[ind1: ind1 + bs1] + route2[ind2 + bs2:]

    return new_route1, new_route2
