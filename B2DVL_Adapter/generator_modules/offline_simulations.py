import carla
import math
import numpy as np

def _manage_route_obstacle_scenarios(self, target_speed, ego_speed, route_waypoints, list_vehicles, route_points):
        """
        From pdm_lite
        This method handles various obstacle and scenario situations that may arise during navigation.
        It adjusts the target speed, modifies the route, and determines if the ego vehicle should keep driving or wait.
        The method supports different scenario types such as InvadingTurn, Accident, ConstructionObstacle, 
        ParkedObstacle, AccidentTwoWays, ConstructionObstacleTwoWays, ParkedObstacleTwoWays, VehicleOpensDoorTwoWays, 
        HazardAtSideLaneTwoWays, HazardAtSideLane, and YieldToEmergencyVehicle.

        Args:
            target_speed (float): The current target speed of the ego vehicle.
            ego_speed (float): The current speed of the ego vehicle.
            route_waypoints (list): A list of waypoints representing the current route.
            list_vehicles (list): A list of all vehicles in the simulation.
            route_points (numpy.ndarray): A numpy array containing the current route points.

        Returns:
            tuple: A tuple containing the updated target speed, a boolean indicating whether to keep driving,
                and a list containing information about a potential decreased target speed due to an object.
        """

        def compute_min_time_for_distance(distance, target_speed, ego_speed):
            """
            Computes the minimum time the ego vehicle needs to travel a given distance.

            Args:
                distance (float): The distance to be traveled.
                target_speed (float): The target speed of the ego vehicle.
                ego_speed (float): The current speed of the ego vehicle.

            Returns:
                float: The minimum time needed to travel the given distance.
            """
            min_time_needed = 0.
            remaining_distance = distance
            current_speed = ego_speed

            # Iterate over time steps until the distance is covered
            while True:
                # Takes less than a tick to cover remaining_distance with current_speed
                if remaining_distance - current_speed * self.config.fps_inv < 0:
                    break

                remaining_distance -= current_speed * self.config.fps_inv
                min_time_needed += self.config.fps_inv

                # Values from kinematic bicycle model
                normalized_speed = current_speed / 120.
                speed_change_params = self.config.compute_min_time_to_cover_distance_params
                speed_change = np.clip(
                    speed_change_params[0] + normalized_speed * speed_change_params[1] +
                    speed_change_params[2] * normalized_speed**2 + speed_change_params[3] * normalized_speed**3, 0.,
                    np.inf)
                current_speed = np.clip(120 * (normalized_speed + speed_change), 0, target_speed)

            # Add remaining time at the current speed
            min_time_needed += remaining_distance / current_speed

            return min_time_needed

        def get_previous_road_lane_ids(starting_waypoint):
            """
            Retrieves the previous road and lane IDs for a given starting waypoint.

            Args:
                starting_waypoint (carla.Waypoint): The starting waypoint.

            Returns:
                list: A list of tuples containing road IDs and lane IDs.
            """
            current_waypoint = starting_waypoint
            previous_lane_ids = [(current_waypoint.road_id, current_waypoint.lane_id)]

            # Traverse backwards up to 100 waypoints to find previous lane IDs
            for _ in range(self.config.previous_road_lane_retrieve_distance):
                previous_waypoints = current_waypoint.previous(1)

                # Check if the road ends and no previous route waypoints exist
                if len(previous_waypoints) == 0:
                    break
                current_waypoint = previous_waypoints[0]

                if (current_waypoint.road_id, current_waypoint.lane_id) not in previous_lane_ids:
                    previous_lane_ids.append((current_waypoint.road_id, current_waypoint.lane_id))

            return previous_lane_ids

        def is_overtaking_path_clear(from_index,
                                     to_index,
                                     list_vehicles,
                                     ego_location,
                                     target_speed,
                                     ego_speed,
                                     previous_lane_ids,
                                     min_speed=50. / 3.6):
            """
            Checks if the path between two route indices is clear for the ego vehicle to overtake.

            Args:
                from_index (int): The starting route index.
                to_index (int): The ending route index.
                list_vehicles (list): A list of all vehicles in the simulation.
                ego_location (carla.Location): The location of the ego vehicle.
                target_speed (float): The target speed of the ego vehicle.
                ego_speed (float): The current speed of the ego vehicle.
                previous_lane_ids (list): A list of tuples containing previous road IDs and lane IDs.
                min_speed (float, optional): The minimum speed to consider for overtaking. Defaults to 50/3.6 km/h.

            Returns:
                bool: True if the path is clear for overtaking, False otherwise.
            """
            # 10 m safety distance, overtake with max. 50 km/h
            to_location = self._waypoint_planner.route_points[to_index]
            to_location = carla.Location(to_location[0], to_location[1], to_location[2])

            from_location = self._waypoint_planner.route_points[from_index]
            from_location = carla.Location(from_location[0], from_location[1], from_location[2])

            # Compute the distance and time needed for the ego vehicle to overtake
            ego_distance = to_location.distance(
                ego_location) + self._vehicle.bounding_box.extent.x * 2 + self.config.check_path_free_safety_distance
            ego_time = compute_min_time_for_distance(ego_distance, min(min_speed, target_speed), ego_speed)

            path_clear = True
            for vehicle in list_vehicles:
                # Sort out ego vehicle
                if vehicle.id == self._vehicle.id:
                    continue

                vehicle_location = vehicle.get_location()
                vehicle_waypoint = self.world_map.get_waypoint(vehicle_location)

                # Check if the vehicle is on the previous lane IDs
                if (vehicle_waypoint.road_id, vehicle_waypoint.lane_id) in previous_lane_ids:
                    diff_vector = vehicle_location - ego_location
                    dot_product = self._vehicle.get_transform().get_forward_vector().dot(diff_vector)
                    # Skip if the vehicle is not relevant, because its not on the overtaking path and behind
                    # the ego vehicle
                    if dot_product < 0:
                        continue

                    diff_vector_2 = to_location - vehicle_location
                    dot_product_2 = vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
                    # The overtaking path is blocked by vehicle
                    if dot_product_2 < 0:
                        path_clear = False
                        break

                    other_vehicle_distance = to_location.distance(vehicle_location) - vehicle.bounding_box.extent.x
                    other_vehicle_time = other_vehicle_distance / max(1., vehicle.get_velocity().length())

                    # Add 200 ms safety margin
                    # Vehicle needs less time to arrive at to_location than the ego vehicle
                    if other_vehicle_time < ego_time + self.config.check_path_free_safety_time:
                        path_clear = False
                        break

            return path_clear

        def get_horizontal_distance(actor1, actor2):
            """
            Calculates the horizontal distance between two actors (ignoring the z-coordinate).

            Args:
                actor1 (carla.Actor): The first actor.
                actor2 (carla.Actor): The second actor.

            Returns:
                float: The horizontal distance between the two actors.
            """
            location1, location2 = actor1.get_location(), actor2.get_location()

            # Compute the distance vector (ignoring the z-coordinate)
            diff_vector = carla.Vector3D(location1.x - location2.x, location1.y - location2.y, 0)

            return diff_vector.length()

        def sort_scenarios_by_distance(ego_location):
            """
            Sorts the active scenarios based on the distance from the ego vehicle.

            Args:
                ego_location (carla.Location): The location of the ego vehicle.
            """
            distances = []

            # Calculate the distance of each scenario's first actor from the ego vehicle
            for (_, scenario_data) in CarlaDataProvider.active_scenarios:
                first_actor = scenario_data[0]
                distances.append(ego_location.distance(first_actor.get_location()))

            # Sort the scenarios based on the calculated distances
            indices = np.argsort(distances)
            CarlaDataProvider.active_scenarios = [CarlaDataProvider.active_scenarios[i] for i in indices]

        keep_driving = False
        speed_reduced_by_obj = [target_speed, None, None, None]  # [target_speed, type, id, distance]

        # Remove scenarios that ended with a scenario timeout
        active_scenarios = CarlaDataProvider.active_scenarios.copy()
        for i, (scenario_type, scenario_data) in enumerate(active_scenarios):
            first_actor, last_actor = scenario_data[:2]
            if not first_actor.is_alive or (last_actor is not None and not last_actor.is_alive):
                CarlaDataProvider.active_scenarios.remove(active_scenarios[i])

        # Only continue if there are some active scenarios available
        if len(CarlaDataProvider.active_scenarios) != 0:
            ego_location = self._vehicle.get_location()

            # Sort the scenarios by distance if there is more than one active scenario
            if len(CarlaDataProvider.active_scenarios) != 1:
                sort_scenarios_by_distance(ego_location)

            scenario_type, scenario_data = CarlaDataProvider.active_scenarios[0]

            if scenario_type == "InvadingTurn":
                first_cone, last_cone, offset = scenario_data

                closest_distance = first_cone.get_location().distance(ego_location)

                if closest_distance < self.config.default_max_distance_to_process_scenario:
                    self._waypoint_planner.shift_route_for_invading_turn(first_cone, last_cone, offset)
                    CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

            elif scenario_type in ["Accident", "ConstructionObstacle", "ParkedObstacle"]:
                first_actor, last_actor, direction = scenario_data[:3]

                horizontal_distance = get_horizontal_distance(self._vehicle, first_actor)

                # Shift the route around the obstacles
                if horizontal_distance < self.config.default_max_distance_to_process_scenario:
                    transition_length = {
                        "Accident": self.config.transition_smoothness_distance,
                        "ConstructionObstacle": self.config.transition_smoothness_factor_construction_obstacle,
                        "ParkedObstacle": self.config.transition_smoothness_distance
                    }[scenario_type]
                    _, _ = self._waypoint_planner.shift_route_around_actors(first_actor, last_actor, direction,
                                                                            transition_length)
                    CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

            elif scenario_type in [
                    "AccidentTwoWays", "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "VehicleOpensDoorTwoWays"
            ]:
                first_actor, last_actor, direction, changed_route, from_index, to_index, path_clear = scenario_data

                # change the route if the ego is close enough to the obstacle
                horizontal_distance = get_horizontal_distance(self._vehicle, first_actor)

                # Shift the route around the obstacles
                if horizontal_distance < self.config.default_max_distance_to_process_scenario and not changed_route:
                    transition_length = {
                        "AccidentTwoWays": self.config.transition_length_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.transition_length_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.transition_length_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.transition_length_vehicle_opens_door_two_ways
                    }[scenario_type]
                    add_before_length = {
                        "AccidentTwoWays": self.config.add_before_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.add_before_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.add_before_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.add_before_vehicle_opens_door_two_ways
                    }[scenario_type]
                    add_after_length = {
                        "AccidentTwoWays": self.config.add_after_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.add_after_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.add_after_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.add_after_vehicle_opens_door_two_ways
                    }[scenario_type]
                    factor = {
                        "AccidentTwoWays": self.config.factor_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.config.factor_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.config.factor_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.config.factor_vehicle_opens_door_two_ways
                    }[scenario_type]

                    from_index, to_index = self._waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, direction, transition_length, factor, add_before_length,
                        add_after_length)

                    changed_route = True
                    scenario_data[3] = changed_route
                    scenario_data[4] = from_index
                    scenario_data[5] = to_index

                # Check if the ego can overtake the obstacle
                if changed_route and from_index - self._waypoint_planner.route_index < \
                                self.config.max_distance_to_overtake_two_way_scnearios and not path_clear:
                    # Get previous roads and lanes of the target lane
                    target_lane = route_waypoints[0].get_left_lane(
                    ) if direction == "right" else route_waypoints[0].get_right_lane()
                    if target_lane is None:
                        return target_speed, keep_driving, speed_reduced_by_obj
                    prev_road_lane_ids = get_previous_road_lane_ids(target_lane)

                    overtake_speed = self.config.overtake_speed_vehicle_opens_door_two_ways \
                                if scenario_type == "VehicleOpensDoorTwoWays" else self.config.default_overtake_speed
                    path_clear = is_overtaking_path_clear(from_index,
                                                          to_index,
                                                          list_vehicles,
                                                          ego_location,
                                                          target_speed,
                                                          ego_speed,
                                                          prev_road_lane_ids,
                                                          min_speed=overtake_speed)

                    scenario_data[6] = path_clear

                # If the overtaking path is clear, keep driving; otherwise, wait behind the obstacle
                if path_clear:
                    if self._waypoint_planner.route_index >= to_index - \
                                                            self.config.distance_to_delete_scenario_in_two_ways:
                        CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]
                    target_speed = {
                        "AccidentTwoWays": self.config.default_overtake_speed,
                        "ConstructionObstacleTwoWays": self.config.default_overtake_speed,
                        "ParkedObstacleTwoWays": self.config.default_overtake_speed,
                        "VehicleOpensDoorTwoWays": self.config.overtake_speed_vehicle_opens_door_two_ways
                    }[scenario_type]
                    keep_driving = True
                else:
                    distance_to_leading_actor = float(from_index + 15 -
                                                      self._waypoint_planner.route_index) / self.config.points_per_meter
                    target_speed = self._compute_target_speed_idm(
                        desired_speed=target_speed,
                        leading_actor_length=self._vehicle.bounding_box.extent.x,
                        ego_speed=ego_speed,
                        leading_actor_speed=0,
                        distance_to_leading_actor=distance_to_leading_actor,
                        s0=self.config.idm_two_way_scenarios_minimum_distance,
                        T=self.config.idm_two_way_scenarios_time_headway
                    )

                    # Update the object causing the most speed reduction
                    if speed_reduced_by_obj is None or speed_reduced_by_obj[0] > target_speed:
                        speed_reduced_by_obj = [
                            target_speed, first_actor.type_id, first_actor.id, distance_to_leading_actor
                        ]

            elif scenario_type == "HazardAtSideLaneTwoWays":
                first_actor, last_actor, changed_route, from_index, to_index, path_clear = scenario_data

                horizontal_distance = get_horizontal_distance(self._vehicle, first_actor)

                if horizontal_distance < self.config.max_distance_to_process_hazard_at_side_lane_two_ways \
                                                                                        and not changed_route:
                    to_index = self._waypoint_planner.get_closest_route_index(self._waypoint_planner.route_index,
                                                                              last_actor.get_location())

                    # Assume the bicycles don't drive more than 7.5 m during the overtaking process
                    to_index += 135
                    from_index = self._waypoint_planner.route_index

                    starting_wp = route_waypoints[0].get_left_lane()
                    prev_road_lane_ids = get_previous_road_lane_ids(starting_wp)
                    path_clear = is_overtaking_path_clear(from_index,
                                                          to_index,
                                                          list_vehicles,
                                                          ego_location,
                                                          target_speed,
                                                          ego_speed,
                                                          prev_road_lane_ids,
                                                          min_speed=self.config.default_overtake_speed)

                    if path_clear:
                        transition_length = self.config.transition_smoothness_distance
                        self._waypoint_planner.shift_route_smoothly(from_index, to_index, True, transition_length)
                        changed_route = True
                        scenario_data[2] = changed_route
                        scenario_data[3] = from_index
                        scenario_data[4] = to_index
                        scenario_data[5] = path_clear

                # the overtaking path is clear
                if path_clear:
                    # Check if the overtaking is done
                    if self._waypoint_planner.route_index >= to_index:
                        CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]
                    # Overtake with max. 50 km/h
                    target_speed, keep_driving = self.config.default_overtake_speed, True

            elif scenario_type == "HazardAtSideLane":
                first_actor, last_actor, changed_first_part_of_route, from_index, to_index, path_clear = scenario_data

                horizontal_distance = get_horizontal_distance(self._vehicle, last_actor)

                if horizontal_distance < self.config.max_distance_to_process_hazard_at_side_lane \
                                                                and not changed_first_part_of_route:
                    transition_length = self.config.transition_smoothness_distance
                    from_index, to_index = self._waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, "right", transition_length)

                    to_index -= transition_length
                    changed_first_part_of_route = True
                    scenario_data[2] = changed_first_part_of_route
                    scenario_data[3] = from_index
                    scenario_data[4] = to_index

                if changed_first_part_of_route:
                    to_idx_ = self._waypoint_planner.extend_lane_shift_transition_for_hazard_at_side_lane(
                        last_actor, to_index)
                    to_index = to_idx_
                    scenario_data[4] = to_index

                if self._waypoint_planner.route_index > to_index:
                    CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

            elif scenario_type == "YieldToEmergencyVehicle":
                emergency_veh, _, changed_route, from_index, to_index, to_left = scenario_data

                horizontal_distance = get_horizontal_distance(self._vehicle, emergency_veh)

                if horizontal_distance < self.config.default_max_distance_to_process_scenario and not changed_route:
                    # Assume the emergency vehicle doesn't drive more than 20 m during the overtaking process
                    from_index = self._waypoint_planner.route_index + 30 * self.config.points_per_meter
                    to_index = from_index + int(2 * self.config.points_per_meter) * self.config.points_per_meter

                    transition_length = self.config.transition_smoothness_distance
                    to_left = self._waypoint_planner.route_waypoints[from_index].lane_change != carla.LaneChange.Right
                    self._waypoint_planner.shift_route_smoothly(from_index, to_index, to_left, transition_length)

                    changed_route = True
                    to_index -= transition_length
                    scenario_data[2] = changed_route
                    scenario_data[3] = from_index
                    scenario_data[4] = to_index
                    scenario_data[5] = to_left

                if changed_route:
                    to_idx_ = self._waypoint_planner.extend_lane_shift_transition_for_yield_to_emergency_vehicle(
                        to_left, to_index)
                    to_index = to_idx_
                    scenario_data[4] = to_index

                    # Check if the emergency vehicle is in front of the ego vehicle
                    diff = emergency_veh.get_location() - ego_location
                    dot_res = self._vehicle.get_transform().get_forward_vector().dot(diff)
                    if dot_res > 0:
                        CarlaDataProvider.active_scenarios = CarlaDataProvider.active_scenarios[1:]

        return target_speed, keep_driving, speed_reduced_by_obj