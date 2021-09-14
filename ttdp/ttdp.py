import math

from multipledispatch import dispatch
import copy as cp


class POI:
    # Point Of Interest

    def __init__(self, index: int, poi_no: int = None, x_coord: float = None, y_coord: float = None,
                 open_time: float = 0, close_time: float = 0, visit_time: float = 0,
                 score: float = 0, distances: list = None, time_distances: list = None, time_matrix=None,
                 arrival_time: float = 0, route_no=-1):

        self.poi_index = index  # POI index
        self.poi_no = poi_no  # POI number
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.opening_time = open_time  # Time window start in ms
        self.closing_time = close_time  # Time window end in ms
        self.visit_time = visit_time  # Visit time in ms
        self.score = score  # Profit
        self.distances = distances  # Distance matrix
        self.time_distances = time_distances  # Time matrix (non time-dependent)
        self.time_matrix = time_matrix  # Time matrix (time-dependent)
        self.arrival_time = arrival_time  # Arrival time at visited POI
        self.route_no = route_no

    @dispatch(int, list)
    def distance(self, other_poi_index: int, distance_matrix: list):
        # Returns pre-calculated distance in distance_matrix
        if distance_matrix is not None:
            return distance_matrix[self.poi_index][other_poi_index]
        else:
            return -1

    @dispatch(object)
    def distance(self, other_poi):
        # Euclidean distance between POIs
        if self.distances is not None:
            # Distances already calculated
            return self.distances[other_poi.poi_index]
        if self.x_coord is not None and self.y_coord is not None and \
                other_poi.x_coord is not None and other_poi.y_coord is not None:
            # If spatial info, calculate distance
            return round(math.sqrt(pow(self.x_coord - other_poi.x_coord, 2) + pow(self.y_coord - other_poi.y_coord, 2)),
                         1)
        else:
            return -1

    @dispatch(object)
    def time_gap_to(self, other_poi):
        # Returns pre-calculated time from this POI to otherPOI in self internal time matrix
        if self.time_distances is not None:
            return self.time_distances[other_poi.poi_index]
        else:
            return -1

    @dispatch(int)
    def time_gap_to(self, other_poi_index: int):
        # Returns pre-calculated time from this POI to otherPOI at index otherPOI_index in self internal time matrix
        if self.time_distances is not None:
            return self.time_distances[other_poi_index]
        else:
            return -1

    @dispatch(object, float, float)
    def time_gap_to(self, other_poi, min_time: float, max_time: float):
        # Returns pre-calculated time from this POI to otherPOI in self internal time matrix at exit_time
        if self.time_matrix is not None:
            if isinstance(self.time_matrix[0], list):
                intervals = len(self.time_matrix[0])
                time_index = get_time_index(min_time, max_time, intervals, self.exit_time)
                return self.time_matrix[other_poi.poi_index][time_index]
            else:
                return self.time_matrix[other_poi.poi_index]
        else:
            return -1

    @dispatch(int, float, float)
    def time_gap_to(self, other_poi_index: int, min_time: float, max_time: float):
        # Returns pre-calculated time from this POI to otherPOI at index otherPOI_index
        # in self internal time matrix at move_time
        if self.time_matrix is not None:
            if isinstance(self.time_matrix[0], list):
                intervals = len(self.time_matrix[0])
                time_index = get_time_index(min_time, max_time, intervals, self.exit_time)
                return self.time_matrix[other_poi_index][time_index]
            else:
                return self.time_matrix[other_poi_index]
        else:
            return -1

    @dispatch(object)
    def time_gap_from(self, other_poi):
        # Returns pre-calculated time from other POI to this POI in internal time matrix
        return other_poi.time_gap_to[self]

    @dispatch(object, float, float)
    def time_gap_from(self, other_poi, min_time: float, max_time: float):
        # Returns pre-calculated time from otherPOI to this POI in self internal time matrix at at_time
        return other_poi.time_gap_to(self, min_time, max_time, other_poi.exit_time)

    @property
    def max_arrival_time(self):
        # Limit for arrival time in TW, if POI is to be visited (in ms)
        return self.closing_time - self.visit_time

    def max_exit_time(self, final_poi, min_time: float, max_time: float):
        # Limit time for leaving POI (in ms), if returning to starting point
        return max_time - self.time_gap_to(final_poi, min_time, max_time)

    @property
    def wait_time(self):
        # Time to wait at entrance (in ms) - it will be non-zero if arrival-time < opening-time
        return max(float(0), self.opening_time - self.arrival_time)

    @property
    def exit_time(self):
        # Exit time
        return max(self.arrival_time, self.opening_time) + self.visit_time

    def json_self_encoder(self):  # -> dict:
        # Convert POI to json string
        return {
            "poi_index": self.poi_index,
            "poi_no": self.poi_no,
            "x_coord": self.x_coord,
            "y_coord": self.y_coord,
            "opening_time": self.opening_time,
            "closing_time": self.closing_time,
            "visit_time": self.visit_time,
            "score": self.score
        }


class DailyTour:
    # Route of POIs

    def __init__(self, route_no: int, initial_poi: POI, start_time: float, t_max: float):
        self.initial_poi = initial_poi
        self.route_no = route_no
        self.initial_time = start_time
        self.t_max = t_max
        self.route = []

    @property
    def effective_size(self):
        # Number of POIs in route
        return len(self.route)  # Init and end node not included

    @property
    def total_score(self):
        # Total profit of route
        return sum(everyPOI.score for everyPOI in self.route)

    @property
    def total_wait_time(self):
        # Total waiting time in route
        if self.route is None or len(self.route) == 0:
            return 0
        else:
            return sum(everyPOI.wait_time for everyPOI in self.route)

    @property
    def total_idle_time(self):
        # Total available time in route
        if self.route is None or len(self.route) == 0:
            return max(float(0), self.t_max - self.initial_time)
        else:
            visit_time = sum(everyPOI.visit_time for everyPOI in self.route)
            return self.total_wait_time + self.t_max - self.total_travel_time - visit_time

    @property
    def total_travel_time(self):
        # Total time spent moving from one POI to another in route
        if self.route is None or len(self.route) == 0:
            return 0
        else:
            # Time spent in visit
            visit_time = sum(everyPOI.visit_time for everyPOI in self.route)
            # Time spent while moving
            travel_time = self.route[-1].exit_time + \
                          self.route[-1].time_gap_to(self.initial_poi, self.initial_time, self.t_max) - \
                          visit_time - self.total_wait_time
            return travel_time

    @property
    def total_visit_time(self):
        # Total time spent moving from one POI to another in route
        if self.route is None or len(self.route) == 0:
            return 0
        else:
            # Time spent in visit
            return sum(everyPOI.visit_time for everyPOI in self.route)

    def update_arrival_times(self, from_poi_index: int = 0):
        if from_poi_index <= 0:
            index = 0
            before_poi = self.initial_poi
        else:
            index = from_poi_index
            before_poi = self.route[from_poi_index - 1]
        for i in range(index, len(self.route)):
            self.route[i].arrival_time = before_poi.exit_time + before_poi.time_gap_to(self.route[i].poi_index,
                                                                                       self.initial_time, self.t_max)
            before_poi = self.route[i]

    def check_feasible_daily_tour(self):
        return check_feasible_route(self.route, self.initial_poi, self.initial_time, self.t_max)

    def json_self_encoder(self):  # -> dict:
        # Convert DailyTour to json string
        return {
            "route_no": self.route_no,
            "initial_time": self.initial_time,
            "t_max": self.t_max,
            "initial_poi": self.initial_poi,
            "route": self.route,
            "effective_size": self.effective_size,
            "total_score": self.total_score,
        }


class CompleteTour:
    # k routes of POIs

    def __init__(self, max_routes: int, initial_poi: POI, start_time: float, t_max: float):

        self.tours = []
        self.max_routes = max_routes
        self.initial_poi = initial_poi
        self.initial_time = start_time
        self.t_max = t_max

        for i in range(max_routes):
            self.tours.append(DailyTour(i, self.initial_poi, self.initial_time, self.t_max))

    @property
    def total_size(self):
        # Total size of route
        return sum([t.effective_size for t in self.tours])

    @property
    def total_score(self):
        # Total profit of route
        return sum([t.total_score for t in self.tours])

    @property
    def total_wait_time(self):
        # Total waiting time in route
        return sum([t.total_wait_time for t in self.tours])

    @property
    def total_idle_time(self):
        # Total available time in route
        return sum([t.total_idle_time for t in self.tours])

    @property
    def total_travel_time(self):
        # Total time spent moving from one POI to another in route
        return sum([t.total_travel_time for t in self.tours])

    @property
    def total_visito_time(self):
        # Total time spent moving from one POI to another in route
        return sum([t.total_visit_time for t in self.tours])

    def update_arrival_times(self, from_tour_no: int, to_tour_no: int, from_poi_index: int == 0):
        init_tour_no = 0
        if from_tour_no is not None:
            if from_tour_no >= 0:
                init_tour_no = from_tour_no
        end_tour_no = self.max_routes
        if to_tour_no is not None:
            if 0 <= to_tour_no <= self.max_routes:
                end_tour_no = to_tour_no
        for i in range(init_tour_no, end_tour_no + 1):
            self.tours[i].update_arrival_times(from_poi_index)

    def check_feasible_tour(self):
        # Checks constraints route
        for tour in self.tours:
            feasible_tour, first_unfeasible_pos = tour.check_feasible_daily_tour()
            if not feasible_tour:
                return False, self.tours.index(tour), first_unfeasible_pos
        return True, None, None

    def json_self_encoder(self):  # -> dict:
        # Convert tour to json string
        return {
            "max_routes": self.max_routes,
            "initial_time": self.initial_time,
            "t_max": self.t_max,
            "initial_poi": self.initial_poi,
            "tours": self.tours,
            "total_score": self.total_score,
        }


class IntermediateSolution:
    # Auxiliary class to improve json serialization of intermediate results

    def __init__(self, movement_type: str, tour: CompleteTour):
        self.movement_type = movement_type
        self.tour = tour


def get_time_index(min_time_value: float, max_time_value: float, time_intervals: float, time_value: float):
    # Get index for time_value in time matrix

    if time_value <= min_time_value:
        return 0

    if time_value >= max_time_value:
        return time_intervals - 1

    delta = (max_time_value - min_time_value) / time_intervals
    index = math.trunc((time_value - min_time_value) / delta)

    return index


def estimate_arrival_exit_times(list_of_pois: list, from_poi: POI, min_time: float, max_time: float):
    # Calculate arrival (and exit) times of a list of POIS if previous POI is from_poi
    calculated = cp.deepcopy(list_of_pois)
    for i in range(0, len(list_of_pois)):
        calculated[i].arrival_time = from_poi.exit_time + \
                                     from_poi.time_gap_to(list_of_pois[i], min_time, max_time)
    return calculated


def update_arrival_times(route: list, initial_poi: POI, from_poi_index: int = 0, min_time: float = 0,
                         max_time: float = 0):
    # Update arrival times of a list of POIS, starting at from_poi_index
    updated_route = cp.deepcopy(route)
    if from_poi_index <= 0:
        # Previous POI is initial_poi
        index = 0
        before_poi = initial_poi
    else:
        # Previous POI in list
        index = from_poi_index
        before_poi = route[from_poi_index - 1]
    for i in range(index, len(route)):
        updated_route[i].arrival_time = before_poi.exit_time + \
                                        before_poi.time_gap_to(route[i].poi_index, min_time, max_time)
        before_poi = cp.deepcopy(updated_route[i])
    return updated_route


def filter_max_arrival_time(list_of_pois: list, from_poi: POI):
    # Select compliant POIs in list, in terms of max_arrival_time
    updated_list = [p for p in list_of_pois if from_poi.exit_time <= p.max_arrival_time]
    return updated_list


def filter_valid_from(list_of_pois: list, previous_poi: POI,
                      t_min: float, t_max: float):
    potential_pois = filter_max_arrival_time(list_of_pois, previous_poi)
    potential_pois = estimate_arrival_exit_times(potential_pois, previous_poi, t_min, t_max)
    potential_pois = [p for p in potential_pois if
                      (p.opening_time <= p.arrival_time <= p.max_arrival_time)]
    return potential_pois


def filter_valid_to(list_of_pois: list, next_poi: POI):
    next_max_arrival_time = next_poi.max_arrival_time
    updated_list = [p for p in list_of_pois if
                    (p.exit_time <= next_max_arrival_time)]
    return updated_list


def filter_valid_to_index(list_of_pois: list, next_poi: POI):
    next_max_arrival_time = next_poi.max_arrival_time
    index_list = [i for i, p in enumerate(list_of_pois) if (p.exit_time <= next_max_arrival_time)]
    return index_list


def filter_valid_from_to(list_of_pois: list, previous_poi: POI, next_poi: POI,
                         t_min: float, t_max: float):
    potential_pois = filter_max_arrival_time(list_of_pois, previous_poi)
    potential_pois = estimate_arrival_exit_times(potential_pois, previous_poi, t_min, t_max)
    next_max_arrival_time = next_poi.max_arrival_time
    potential_pois = filter_valid_times(potential_pois, next_max_arrival_time)
    return potential_pois


def filter_valid_times(list_of_pois: list, next_max_arrival_time: float):  # TODO
    # Select compliant POIs in list, in terms of time windows and max_exit_time
    updated_list = [p for p in list_of_pois if
                    (p.opening_time <= p.arrival_time <= p.max_arrival_time and p.exit_time <= next_max_arrival_time)]
    return updated_list


def filter_time_windows_error(list_of_pois: list):
    # Select compliant POIs in list, in terms of time_windows
    updated_list = [p for p in list_of_pois if
                    (p.closing_time >= p.opening_time + p.visit_time)]
    return updated_list


def get_potential_insert_range(list_of_pois: list, candidate_poi: POI):
    # Select potential compliant POIs in list for candidate_poi:
    #  - Next poi closing time needs to be later than candidate_poi.opening_time, if candidate to be inserted before
    #  - Previous poi visit needs to be ended at least at candidate_poi.next_max_arrival_time, if to be inserted after
    next_max_arrival_time = candidate_poi.max_arrival_time
    opening_time = candidate_poi.opening_time
    # Get max position for pois whose closing time is less than candidate opening time;
    # candidate poi cannot be inserted before those pois
    no_possible_before = [i for i, p in enumerate(list_of_pois) if (p.closing_time < opening_time)]
    min_insert_pos = 0
    if no_possible_before is not None and len(no_possible_before) > 0:
        min_insert_pos = max(no_possible_before)
    # Get min position for pois whose exit time is higher than candidate max arrival time;
    # candidate poi cannot be inserted after those pois
    no_possible_after = [i for i, p in enumerate(list_of_pois) if (p.exit_time > next_max_arrival_time)]
    max_insert_pos = len(list_of_pois)
    if no_possible_after is not None and (len(no_possible_after) > 0):
        max_insert_pos = min(no_possible_after)
    return min_insert_pos, max_insert_pos


def check_feasible_route(route: list, initial_poi: POI, min_time: float, t_max: float):  # TODO
    # Checks constraints for route
    for i in range(0, len(route)):
        # Check time_windows
        if route[i].arrival_time < route[i].opening_time:
            return False, i
        if max(route[i].arrival_time, route[i].opening_time) + route[i].visit_time > route[i].closing_time:
            return False, i
    if len(route) >= 1:
        # Check t_max
        last_poi = route[-1]
        if last_poi.exit_time + last_poi.time_gap_to(initial_poi, min_time, t_max) > t_max:
            return False, len(route)
    return True, None


def is_best_daily_tour1(tour1: DailyTour, tour2: DailyTour):
    # Check solution quality - Single tour
    if tour1.total_score > tour2.total_score:
        # Tour 1 has better score
        return True
    elif tour1.total_score == tour2.total_score:
        # Same score
        if tour1.total_travel_time < tour2.total_travel_time:
            # Tour1 has less travel time
            return True
    return False


def is_best_tour1(tour1: CompleteTour, tour2: CompleteTour):
    # Check solution quality - Complete tour
    if tour1.total_score > tour2.total_score:
        # Tour 1 has better score
        return True
    elif tour1.total_score == tour2.total_score:
        # Same score
        if tour1.total_travel_time < tour2.total_travel_time:
            # Tour1 has less travel time
            return True
    return False
