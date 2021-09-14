import csv
import os

from matplotlib import pyplot as plt

from ttdp import DailyTour, CompleteTour


class ResultInfo:

    def __init__(self, problem: str, instance_file_id: str, instance_file: str,
                 no_routes: int,
                 ns: str, max_ns_size: int,
                 init_build_mode: str,
                 ls: str):

        # Parameters
        self.problem = problem  # Type of problem (TOPTW/TDTOPTW)
        self.instance_file_id = instance_file_id  # Type of instance
        self.instance_file = instance_file  # Instance file
        self.no_routes = no_routes  # Number of routes to be planned
        self.ns = ns  # Neighbourhood structure (Replace, Relocate,...) - list of values from VNS_NS
        self.ns_size = max_ns_size  # Max size of Ns, block, (replace block, relocate block, cross block)
        self.init_mode = init_build_mode  # Initial solution construction method - value from SELECT_CRITERIA
        self.ls = ls  # Local search neighbourhood - SELECT_CRITERIA
        # Results
        self.no_visited_pois = 0  # Number of visited points
        self.init_score = 0  # Total score
        self.score = 0  # Total score
        self.travel_time = 0  # Total travel time
        self.wait_time = 0  # Total wait time
        self.run_time = 0  # Total run time
        self.info = ""  # Comments

    def update_result(self, no_visited_pois: int, init_score: float, score: float, travel_time: float, wait_time: float,
                      run_time: float, info: str = ""):
        self.no_visited_pois = no_visited_pois  # Number of visited points
        self.init_score = init_score  # Score of initial solution
        self.score = score  # Total score
        self.travel_time = travel_time  # Total travel time
        self.wait_time = wait_time  # Total wait time
        self.run_time = run_time  # Total run time
        self.info = info  # Additional info

    def append_to_csv(self, file_name: str):
        # Write results to csv file

        file_exists = os.path.isfile(file_name)

        with open(file_name + '.csv', 'a') as csvfile:

            headers = [key for key, value in self.__dict__.items()]
            writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=headers, quoting=csv.QUOTE_NONE, dialect='excel')
            if not file_exists:
                if csvfile.tell() == 0:
                    writer.writeheader()
            writer.writerow(self.__dict__)

        csvfile.close()


def print_route(route: list):
    formatted_route = str(route[0].poi_index)
    for j in range(1, len(route)):
        formatted_route = formatted_route + '-' + str(route[j].poi_index)
    print(f' Route Nodes: {formatted_route} (Total: {len(route)})')


def print_tour(tour: DailyTour, explicit: bool = True):
    if len(tour.route) > 0:
        formatted_tour = str(tour.route[0].poi_index)
    else:
        formatted_tour = 'Empty'
    for j in range(1, len(tour.route)):
        formatted_tour = formatted_tour + '-' + str(tour.route[j].poi_index)
    print(
        f' (Score={tour.total_score}), Total Wait Time: {tour.total_wait_time}, '
        f'Total Idle Time: {tour.total_idle_time}, Total Travel Time: {tour.total_travel_time}, '
        f'Total Visit Time: {tour.total_visit_time}')
    print(f' Nodes: {formatted_tour} - Count node: {tour.effective_size}')
    if explicit:
        for j in range(len(tour.route)):
            print(
                f'   Arrival at node {tour.route[j].poi_index} at {tour.route[j].arrival_time}, '
                f'opening time {tour.route[j].opening_time}, wait time {tour.route[j].wait_time}, '
                f'visit time {tour.route[j].visit_time}, exit at {tour.route[j].exit_time}')


def print_complete_tour(complete_tour: CompleteTour, step_type: str, explicit: bool = True):
    print(f'{step_type} Solution ----------------------------------')
    if len(complete_tour.tours) > 1:
        print(
            f'Total {len(complete_tour.tours)} tours (Score={complete_tour.total_score})'
            f' (Size={complete_tour.total_size}) ')
    for i in range(len(complete_tour.tours)):
        print_tour(complete_tour.tours[i], explicit)
    print('--------------------------------------------------')


def plot_test_result(list_of_pois: list):
    fig, axs = plt.subplots()
    fig.suptitle('POI')

    x = []
    y = []
    for position in list_of_pois:
        x.append(position.x_coord)
        y.append(position.y_coord)

    axs.scatter(x, y, s=100, c='r', alpha=0.25)

    for i, val in enumerate(x):
        if i < len(x) - 1:
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]

            plt.arrow(x[i], y[i], dx, dy, head_width=0.5, length_includes_head=True)

    plt.grid()
    plt.show()

