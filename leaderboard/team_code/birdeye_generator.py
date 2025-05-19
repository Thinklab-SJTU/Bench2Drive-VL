import argparse
import os

import carla
import matplotlib.pyplot as plt
import numpy as np
import pygame

import birdeye_render


class BEVGenerator:
    def __init__(self, hero_actor):
        self.ego = hero_actor
        self.world = self.ego.get_world()
        self.map = self.world.get_map()
        
        pygame.init()
        self.display = pygame.display.set_mode((500, 500), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HIDDEN)
        birdeye_params = {
            'screen_size': [500, 500],
            'pixels_per_meter': 8,
            'pixels_ahead_vehicle': 8 * (32/2 - 12) # (obs_range/2 - self.d_behind) * pixels_per_meter
        }
        self.birdeye_render = birdeye_render.BirdeyeRender(self.world, birdeye_params)
        self.birdeye_render.set_hero(self.ego, self.ego.id)
        
        self.vehicle_polygons = []
        self.walker_polygons = []
        self.full_road_waypoints = self._generate_road_waypoints()
    
    def __del__(self):
        pygame.quit()
    
    def _generate_road_waypoints(self, precision=0.5):
        """Return all, precisely located waypoints from the map.

        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.

        Returns a list of waypoints for each road segment.
        """
        cache_file = os.path.join(os.path.dirname(birdeye_render.__file__), f"birdeye_cache/{self.map.name.split('/')[-1]}_{precision}.npy")
        if os.path.exists(cache_file):
            return np.load(cache_file, allow_pickle=True)
        else:
            road_segments_starts = [
                road_start for road_start, road_end in self.map.get_topology()
            ]
            
            def extract_waypoint(wpt):
                return (wpt.transform.location.x, wpt.transform.location.y, wpt.transform.rotation.yaw)

            each_road_waypoints = []
            for road_start_waypoint in road_segments_starts:
                road_waypoints = [extract_waypoint(road_start_waypoint)]

                # Generate as long as it's the same road
                next_waypoints = road_start_waypoint.next(precision)

                if len(next_waypoints) > 0:
                    # Always take first (may be at intersection)
                    next_waypoint = next_waypoints[0]
                    while next_waypoint.road_id == road_start_waypoint.road_id:
                        road_waypoints.append(extract_waypoint(next_waypoint))
                        next_waypoint = next_waypoint.next(precision)

                        if len(next_waypoint) > 0:
                            next_waypoint = next_waypoint[0]
                        else:
                            # Reached the end of road segment
                            break
                each_road_waypoints.append(road_waypoints)
            
            each_road_waypoints = np.array(each_road_waypoints)
            np.save(cache_file, each_road_waypoints)
            return each_road_waypoints
    
    def render(self):
        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > 3:
            self.vehicle_polygons.pop(0)
        
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > 3:
            self.walker_polygons.pop(0)
        
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self._nearby_waypoints()

        # birdeye view with roadmap, actors and waypoints
        birdeye_render_types = ['roadmap', 'actors', 'waypoints']
        self.birdeye_render.render(self.display, birdeye_render_types)
        return pygame.surfarray.array3d(self.display).swapaxes(0, 1)
    
    def _nearby_waypoints(self):
        """Return only the road that is nearby the ego vehicle."""
        nearby_waypoints = []
        render_x, render_y = self.birdeye_render.params['screen_size']
        pixels_per_meter = self.birdeye_render.params['pixels_per_meter']

        half_width = render_x / (2 * pixels_per_meter)
        half_height = render_y / (2 * pixels_per_meter)

        ego_x = self.ego.get_location().x
        ego_y = self.ego.get_location().y

        # Define bounding box
        left_bound = ego_x - half_width
        right_bound = ego_x + half_width
        top_bound = ego_y - half_height
        bottom_bound = ego_y + half_height

        for road_waypoints in self.full_road_waypoints:
            road = [waypoint for waypoint in road_waypoints if left_bound < waypoint[0] < right_bound and top_bound < waypoint[1] < bottom_bound]
            
            if road:
                nearby_waypoints.append(road)

        return nearby_waypoints

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
        filt: the filter indicating what type of actors we'll look at.

        Returns:
        actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict={}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor\
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l,w], [l,-w], [-l,-w], [-l,w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw),-np.sin(yaw)], [np.sin(yaw),np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R,poly_local).transpose() + np.repeat([[x,y]],4,axis=0)
            actor_poly_dict[actor.id]=poly
        return actor_poly_dict


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Carla birdeye view script")
    argparser.add_argument("--host", default="localhost", help="IP of the host server (default: localhost)")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
    args = argparser.parse_args()
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    world = client.load_world("Town05")
    map = world.get_map()
    
    # Spawn an ego
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'hero')
    spawn_points = map.get_spawn_points()
    hero_actor = world.spawn_actor(ego_bp, spawn_points[0])
    hero_actor.set_autopilot(True)
    
    bev_gen = BEVGenerator(hero_actor)
    data = bev_gen.render()
    
    # Visualize with matplotlib
    plt.imshow(data)
    plt.show()
    
    hero_actor.destroy()