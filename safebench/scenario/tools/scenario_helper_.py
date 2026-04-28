from safebench.carla_agents.navigation.local_planner import RoadOption
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


def get_location_in_distance_from_wp(waypoint, distance, stop_at_junction=True):
    """
        Obtain a location in a given distance from the current actor's location.
        Note: Search is stopped on first intersection.
    """
    traveled_distance = 0
    while not (waypoint.is_intersection and stop_at_junction) and traveled_distance < distance:
        wp_next = waypoint.next(1.0)
        if wp_next:
            waypoint_new = wp_next[-1]
            traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
            waypoint = waypoint_new
        else:
            break

    return waypoint.transform.location, traveled_distance


def get_waypoint_in_distance(waypoint, distance):
    """
        Obtain a waypoint in a given distance from the current actor's location.
        Note: Search is stopped on first intersection.
    """
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint, traveled_distance

def generate_target_waypoint_in_route(waypoint, route):
    """
        This method follow waypoints to a junction and returns a waypoint list according to turn input
    """
    wmap = CarlaDataProvider.get_map()
    reached_junction = False

    # Get the route location
    shortest_distance = float('inf')
    for index, route_pos in enumerate(route):
        wp = route_pos[0]
        trigger_location = waypoint.transform.location

        dist_to_route = trigger_location.distance(wp)
        if dist_to_route <= shortest_distance:
            closest_index = index
            shortest_distance = dist_to_route

    route_location = route[closest_index][0]
    index = closest_index

    while True:
        # Get the next route location
        index = min(index + 1, len(route))
        route_location = route[index][0]
        road_option = route[index][1]

        # Enter the junction
        if not reached_junction and (road_option in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT)):
            reached_junction = True

        # End condition for the behavior, at the end of the junction
        if reached_junction and (road_option not in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT)):
            break

    return wmap.get_waypoint(route_location)
