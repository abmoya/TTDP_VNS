import json as js
from matplotlib import pyplot as plt


def save_to_json_file(filename: str, json_encoded_obj):
    # Save tour info to json file
    with open(filename + '.json', 'w') as f:
        js.dump(json_encoded_obj, f, default=json_encoder, indent=2)


def json_encoder(obj):
    # Encode data
    if hasattr(obj, 'json_self_encoder'):  # Use class encoder
        return obj.json_self_encoder()
    else:  # Default Behaviour
        return obj.__dict__


class Position:
    def __init__(self, poi_index, poi_no, x_coord, y_coord, opening_time, closing_time, visit_time, score):
        self.poi_index = poi_index
        self.poi_no = poi_no
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.visit_time = visit_time
        self.score = score

    @classmethod
    def from_json(cls, data):
        # Convert json info to Position object
        return cls(data["poi_index"], data["poi_no"], data["x_coord"], data["y_coord"], data["opening_time"],
                   data["closing_time"], data["visit_time"], data["score"])


class Route:
    def __init__(self, route_no: int, initial_time: int, t_max: float, initial_poi: Position, route: list,
                 effective_size: int, total_score: float):
        self.route_no = route_no
        self.initial_time = initial_time
        self.t_max = t_max
        self.initial_poi = initial_poi
        self.route = route
        self.effective_size = effective_size
        self.total_score = total_score

    @classmethod
    def from_json(cls, data):
        # Convert json info to Route object
        route = list(map(Position.from_json, data["route"]))
        position = Position.from_json(data["initial_poi"])
        return cls(data["route_no"], data["initial_time"], data["t_max"], position, route, data["effective_size"],
                   data["total_score"])


class Tour(object):
    def __init__(self, max_routes: int, initial_time: int, t_max: float, initial_poi: Position, tours: list,
                 total_score: float):
        self.max_routes = max_routes
        self.initial_time = initial_time
        self.t_max = t_max
        self.initial_poi = initial_poi
        self.tours = tours
        self.total_score = total_score

    @classmethod
    def from_json(cls, data):
        # Convert json info to Tour object
        tours = list(map(Route.from_json, data["tours"]))
        position = Position.from_json(data["initial_poi"])
        return cls(data["max_routes"], data["initial_time"], data["t_max"], position, tours, data["total_score"])


class Result:
    def __init__(self, movement_type: str, tour: Tour):
        self.movement_type = movement_type
        self.tour = tour

    @classmethod
    def from_json(cls, data):
        return cls(data["movement_type"], Tour.from_json(data["tour"]))


def decode_result_from_json(filename):
    with open(filename) as json_file:
        json_data = js.load(json_file)

    results = []
    for data in json_data:
        results.append(Result.from_json(data))

    return results


def summary_result_from_json(filename):
    results = decode_result_from_json(filename)
    print('Results[' + str(len(results)) + ']:')

    for result in results:
        print('\t[+] movement_type=' + str(result.movement_type))
        print('\t | max_routes=' + str(result.tour.max_routes))
        print('\t | initial_time=' + str(result.tour.initial_time))
        print('\t | t_max=' + str(result.tour.t_max))
        print('\t | total_score=' + str(result.tour.total_score))

        for route in result.tour.tours:
            print('\t[+] route_no=' + str(route.route_no))
            print('\t\t\t| initial_time=' + str(route.initial_time))
            print('\t\t\t| t_max=' + str(route.t_max))
            print('\t\t\t| effective_size=' + str(route.effective_size))
            print('\t\t\t| total_score=' + str(route.total_score))


def plot_poi_from_json(filename: str, axs):
    position_array = position_array_from_json(filename)

    x = []
    y = []
    for position in position_array:
        x.append(position[0])
        y.append(position[1])

    axs.scatter(x, y, s=150, c='r', alpha=0.25)


def plot_result_from_json(result_filename: str, poi_filename: str, result_index: int, show_order: bool = False):
    colors = ["blue", "green", "black", "cyan", "red", "yellow"]
    colors_len = len(colors)
    colors_index = 0
    plot_index = 0

    results = decode_result_from_json(result_filename)
    assert result_index < len(results), 'Index result_index out of range'
    result = results[result_index]

    fig, axs = plt.subplots()
    fig.suptitle('Result: ' + result.movement_type)

    for route in result.tour.tours:
        x = []
        y = []
        n = []

        x.append(route.initial_poi.x_coord)
        y.append(route.initial_poi.y_coord)
        n.append("Start/End")

        order = 1
        for position in route.route:
            x.append(position.x_coord)
            y.append(position.y_coord)
            n.append(str(order))
            order = order + 1

        x.append(route.initial_poi.x_coord)
        y.append(route.initial_poi.y_coord)

        plot_poi_from_json(poi_filename, axs)
        axs.plot(x, y, marker="o", color=colors[colors_index], linestyle='dashed', label='Ruta: ' + str(route.route_no))
        axs.scatter(route.initial_poi.x_coord, route.initial_poi.y_coord, s=300, c='black', alpha=0.25)
        axs.legend()

        if show_order:
            for i, txt in enumerate(n):
                axs.annotate(txt, (x[i], y[i]), weight='bold', color=colors[colors_index], horizontalalignment='left',
                             verticalalignment='top')

        plot_index = plot_index + 1
        colors_index = colors_index + 1
        if colors_index >= colors_len:
            colors_index = 0

    plt.show()


def plot_poi_from_json(position_array: list, filename: str):
    data = [{"x": position[0], "y": position[1]} for position in position_array]
    with open(filename, 'r') as json_file:
        js.dump(data, json_file, ensure_ascii=False, indent=4)


def position_array_from_json(filename: str):
    with open(filename) as json_file:
        json_data = js.load(json_file)

    results = []
    for data in json_data:
        results.append((data['x'], data['y']))

    return results
