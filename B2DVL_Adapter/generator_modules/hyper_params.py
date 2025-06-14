############ UTILS ############

INF_MAX = 32125118964
# a large impossible number

INVALID_NUM = 32118
# number used for invalid state





###### GENERAL SETTINGS #######

NEGLECT_FRAME_COUNT = 5
# the number of frames which are not trusted when a scenario starts
# this is because the scene is still initializing

CARLA_LANE_WIDTH = 3.5
# lane width in CARLA
# unit: meter

OVER_SPEED_LIMIT_RADIUS = 1.031266635497984
# observed from pdm_lite
# if the ego vehicle's speed / speed limit is above this radius
# it is considered above the speed limit

OPPOSITE_ANGLE_THRESHOLD = 120.0
# if the ego vehicle's deviation from current lane is above this angle
# it is marked 'opposite'
# unit: degree

STEER_THRESHOLD = 0.9
SLIGHT_STEER_THRESHOLD = 0.2
# in vehicle descriptions, `determine_vehicle_trajectory`
# if steer's abs is above STEER_THRESHOLD, it is considered 'turning'
# if it is above SLIGHT_STEER_THRESHOLD, it is considered 'turning slightly'

BACK_CONSIDER_THRESHOLD = -1.5
MIN_OBJECT_NUM_POINT = 15
MIN_BICYCLE_NUM_POINT = 10
MIN_PEDESTRIAN_NUM_POINT = 5
# for normal vehicles (except special roles in scenarios, rear vehicles when changing lane)
# considered vehicles must meet some conditions:
# 1. appears in the front camera
# 2. if the vehicle is not bicycle, LiDAR points >= MIN_OBJECT_NUM_POINT
# 3. if the vehicle is a bicycle, LiDAR points >= MIN_BICYCLE_NUM_POINT
# 4. not in rear, relative position x >= BACK_CONSIDER_THRESHOLD
# 5. not a parked vehicle, unless it's cutting in the ego vehicle's lane
# (please check out `should_consider_vehicle` in `carla_vqa_generator.py`)
# unit: meter, whatever, whatever

FRONT_Y_OFFSET = 2.25
# the vehicle whose relative position y within [-FRONT_Y_OFFSET, FRONT_Y_OFFSET]
# and relative position x larger than BACK_CONSIDER_THRESHOLD
# and meet all consider conditions above
# is considered 'in the front'
# (please check out `is_object_in_front` in `carla_vqa_generator.py`)
# unit: meter

CURVATURE_CHECK_DISTANCE = 20.0
# `get_lane_curvature` uses this parameter
# it checks waypoints from 4m ahead to this position every 2m
# and return the lane curvature direction once the curve is noticed
# in other words, detect the nearest curve within this distance
# unit: meter

CURVATURE_TRIGGER_VALUE = 0.1
# if the abs curvature is above this value, it is noticed.

JUNCTION_EXTEND_DISTANCE = 8.0
# if the vehicle's distance to junction is below this
# it is considered in the junction in many functions
# unit: meter

BRAKE_MIN_SPEED = 1.0
BRAKE_MAX_SPEED = 3.0
# when the final cmd is 'DECCELERATE'
# it will be rewritten to 'ACCERLERATE' if the ego speed is below BRAKE_MIN_SPEED
# it will be rewritten to 'KEEP' if the ego speed is between these two values
# unit: m/s

BRAKE_INTERVAL = 2.5
# in some scenarios, for example obstacles, brake applies within max(ego_speed * BRAKE_INTERVAL, some threshold)
# unit: second

NORMAL_KEEP_SPEED = 5.0
HIGHWAY_KEEP_SPEED = 7.0
SURROUNDING_SPEED_RADIUS = 20.0
DURABLE_SPEED_MARGIN = 1.0
# when the final cmd is 'KEEP'
# it will be rewritten to 'ACCERLERATE' if the ego speed is below these values
# or below (average speed of surrounding vehicles within SURROUNDING_SPEED_RADIUS) - DURABLE_SPEED_MARGIN
# in corresponding situations
# unit: m/s

HIGHWAY_MIN_SPEED = 50.0
# if current speed limit is below this
# the ego vehicle is not on the highway
# unit: km/s

ALIGN_THRESHOLD = 40.0
# in normal situations
# if the angle between ego vehicle's forward vector and its current lane's forward vector
# is above this
# the ego vehicle is adviced to steer to align with current lane
# unit: m/s

OOD_ADJUST_SPEED = 2.0
# if the ego vehicle is out of the road and its speed is below this
# the speed command will be rewritten to 'ACCELERATE'
# unit: m/s

TURN_STEER_THRESHOLD = 0.25
# if steer > this, turning right
# if steer < -this, turning left





########### MERGING ###########

MERGING_IGNORE_INVERVAL = 5
# not used: now white list is controled by vehicle distance
# used when the ego vehicle needs to take chance to merge/cross actor flow
# the vehicle in the list will be ignored during a time interval (MERGING_IGNORE_INVERVAL)
# because the ego vehicle must take a dangerous action
# BEWARE: this time unit is INTERVENE COUNT, in other words, the time vqa is generated
# eg. if all vqas generates every 0.5s and MERGING_IGNORE_INVERVAL = 4
# the actual time is 0.5s * 4 = 2s

EXIT_MIN_X = 2.0
# if the actor is on the ego vehicle's side,
# the ego vehicle will move forward slowly
# until the relative x of targeted junction exit is below this
# unit: meter





########## OBSTACLES ##########

HAZARD_STATIC_THRESHOLD = 10.0
HAZARD_PREDICT_TIME = 2.5
# only used OUT OF CARLA, in `get_hazard_by_future` function.
# static objects within STATIC_THRESHOLD and within the ego vehicle's speed * PREDICT_TIME
# are considered hazardous
# unit: meter, second

NORMAL_CONSIDER_RADIUS = 15.0
SAME_LANE_CONSIDER_RADIUS = 25.0
ACTOR_IN_FRONT_CONSIDER_RADIUS = 45.0
JUNCTION_CONSIDER_RADIUS = 40.0
OPPOSITE_LANE_CONSIDER_RADIUS = 80.0
# when considering overlap vehicles
# consider radius on different situations:
# normal, vehicles in front, 
# vehicle nears a junction,
# vehicles in the same lane as the ego vehicle,
# and when the ego vehicle is driving in opposite direction of the lane
# unit: meter

PEDESTRIAN_STOP_DISTANCE = 10.0
PEDESTRIAN_BRAKE_DISTANCE = 20.0 # not used
PEDESTRIAN_CONSIDER_DISTANCE = 30.0
# the ego vehicle should stop/brake/notice within this distance to pedestrians
# unit: meter

PEDESTRIAN_MIN_SPEED = 1.0
# the pedestrian below this speed and out of STOP_DISTANCE
# will be ignored
# i know it doesn't sound right, but its purpose is to filter peds crossing the road
# unit: m/s

PEDESTRIAN_MIN_Y = -1.75
PEDESTRIAN_MAX_Y = 3.5
# in earlier versions, peds whose relative y within this range
# will be considered crossing the road
# since the lane width in carla is 3.5m
# but now all peds within front camera are considered
# unit: meter

BICYCLE_STOP_DISTANCE = 10.0
BICYCLE_BRAKE_DISTANCE = 20.0 # not used
BICYCLE_CONSIDER_DISTANCE = 30.0
# the ego vehicle should stop/brake/notice within this distance to bicycles
# unit: meter

BICYCLE_MIN_SPEED = 1.0
# the bicycle below this speed and out of STOP_DISTANCE
# will be ignored
# i know it doesn't sound right, but its purpose is to filter peds crossing the road
# unit: m/s

BICYCLE_MIN_Y = -2.75
BICYCLE_MAX_Y = 4.5
# in earlier versions, peds whose relative y within this range
# will be considered crossing the road
# since the lane width in carla is 3.5m
# but now all peds within front camera are considered
# unit: meter

BICYCLE_CROSS_ANGLE = 90.0
# if the bicycle is hazardous in front of the ego vehicke
# and it drives towards the ego vehicle within angle range [-CROSS_ANGLE, CROSS_ANGLE]
# the ego vehicle should wait for it to pass

VEHICLE_TOWARDS_ANGLE = 45.0
# if the vehivle is hazardous in front of the ego vehicke
# and it drives towards the ego vehicle within angle range [-CROSS_ANGLE, CROSS_ANGLE]
# the ego vehicle should wait for it to pass

STRAIGHT_DRIVE_MIN_SPEED = 3.0
STRAIGHT_DRIVE_MIN_STEER = 0.5
# if the ego vehicle's speed <= STRAIGHT_DRIVE_MIN_SPEED
# or the ego vehicle's abs steer <= STRAIGHT_DRIVE_MIN_STEER
# it is considered in normal situations in `determine_braking_requirements`
# unit: meter, steer

NORMAL_BRAKE_CONSIDER_RADIUS = 15.0
JUNCTION_BRAKE_CONSIDER_RADIUS = 12.0
# for normal objects in front of the ego vehicle
# only which within these distances are considered
# unit: meter

SLOW_VEHICLE_SPEED = 4.0
# if the vehicle in front's speed is below this
# the ego vehicle should consider braking
# unit: m/s

SLOW_VEHICLE_RATIO = 2.0 / 3.0
# if the vehicle's speed is below the ego vehicle's speed * SLOW_VEHICLE_RATIO
# it would also be considered slow
# unit: no

STOP_VEHICLE_SPEED = 0.5
# the vehicle below this speed
# is considered stopped
# unit: m/s

STOP_FOR_STOPPED_VEHICLE_DISTANCE = 15.0
BRAKE_FOR_STOPPED_VEHICLE_DISTANCE = 20.0 # not used
# if the stopped vehicle is within this distance to the ego vehicle
# the ego vehicle should stop/brake
# unit: meter

STOP_FOR_SLOW_VEHICLE_DISTANCE = 10.0
BRAKE_FOR_SLOW_VEHICLE_DISTANCE = 15.0
# if the slow vehicle is within this distance to the ego vehicle
# the ego vehicle should stop/brake
# unit: meter

BRAKE_FOR_LEADING_VEHICLE_DISTANCE = 20.0
# if the leading vehicle is out of this distance
# the ego vehicle needn't brake because of it
# unit: meter

VEHICLE_BRAKE_THRESHOLD = 0.6
# if the vehicle in front's brake is above this
# the ego vehicle should consider braking
# unit: brake

TOO_CLOSE_THRESHOLD = 10.0
# if the vehicle in front is closer than this distance
# the ego vehicle will be adviced 'control your speed'
# unit: meter

TOO_DANGEROUS_X_MIN = -1.0
TOO_DANGEROUS_Y_MAX = 1.5
TOO_DANGEROUS_DISTANCE_MAX = 6.5
# the vehicle whose x > X_MIN, |y| < Y_MAX, distance < DISTANCE_MAX
# is too dangerous
# so the ego vehicle MUST STOP
# unit: meter

HIGHWAY_WAIT_FOR_LANE_CHANGE_RATIO = 6.0
HIGHWAY_WAIT_FOR_LANE_CHANGE_DISTANCE = 30.0
# when the ego vehicle wants to change its lane
# if the vehicle is within this distance to highway exit
# and its speed is above its distance to junction / RATIO
# it must slow down and wait for its chance
# otherwise the exit will be missed
# unit: second, meter

HIGHWAY_MEET_MAX_DISTANCE = 20.0
HIGHWAY_MEET_MAX_Y = 7.5
HIGHWAY_MEET_MIN_X = -20.0
HIGHWAY_MEET_STOP_MAX_X = 10.0
HIGHWAY_MEET_STOP_MIN_X = -10.0
HIGHWAY_MEET_STOP_MIN_Y = 0.75
HIGHWAY_MEET_STOP_MAX_Y = 5.5
# used when determining the vehicles meeting the ego vehicle
# near the intersection on the highway
# stop when the vehicle within these area but relative y further than MIN_Y, closer than MAX_Y
# unit: meter

HIGHWAY_MEET_MIN_ANGLE = 0.05
# the minimum relative yaw the vehicle should have
# to meet the ego vehicle near the intersection on the highway
# unit: rad

HIGHWAY_MEET_FRONT_REAR_DIVIDER = -2.0
# divide the front and rear meeting vehicles on the highway
# unit: meter

BLOCKED_INTERSECTION_CONSIDER_DISTANCE = 15.0
# the role in BlockedIntersection is considered within this distance
# unit: meter

BLOCKED_INTERSECTION_STOP_DISTANCE = 10.0
# the ego vehicle must stop of the role in BlockedIntersection is within this distance
# unit: meter

INTERURBAN_ACTOR_FLOW_STOP_DISTANCE = 10.0
# the ego vehicle must stop of the hazardous vehicle in InterurbanActorFlow is within this distance
# unit: meter

INVADING_TURN_FORWARD_DISTANCE = 1.5
# the traffic cone whose relative x > this
# is considered the current obstacle
# unit: meter

PRIORITY_VEHICLE_MIN_INTERSECT_DISTANCE = -2.0
PRIORITY_VEHICLE_MAX_INTERSECT_DISTANCE = 20.0
# when the emergency vehicle is taking priority
# if the ego vehicle meets its track within this range
# it will be conisdered
# unit: meter

PRIORITY_VEHICLE_MIN_SPEED = 4.0
# when the emergency vehicle is taking priority
# it will not be considered if its speed is below this value
# unit: m/s

PRIORITY_VEHICLE_MIN_DISTANCE = 30.0
# when the emergency vehicle is taking priority
# it will not be considered if its distance is above this value
# unit: meter

VEHICLE_OPEN_DOOR_TRIGGER_DISTANCE = 22.0
# the trigger distance of VehicleOpensDoors
# had better not change this
# unit: meter

ACCIDENT_PASS_OFFSET = 16.0
CONSTRUCTION_PASS_OFFSET = 15.0
HAZARDATSIDE_PASS_OFFSET = 10.0
# when calculating "passed_circumvent_obstacle", these extra offsets are applied
# incase the ego vehicle changes lane back too early





######## INTERSECTIONS ########

MAX_SPEED_IN_JUNCTION = 64.0 / 3.6
# the ego vehicle's speed should not pass this in junctions
# unit: m/s

INTERSECTION_CROSS_INTERVAL = 2.1
RIGHT_TURN_INTERSECTION_CROSS_INTERVAL = 1.5
# vehicle is predicted 'cross' if the ego vehicle and the vehicle arrives at the same point between this time interval
# AND under RATIO * (time the ego vehicle gets to the intersection) ^ 2 (below)
# unit: second

INTERSECTION_CROSS_RATIO = 2.4
# INTERSECTION_CROSS_INTERVAL = RATIO * (time the ego vehicle gets to the intersection) ^ 2

RIGHT_TURN_CROSS_RATIO = 2.1
# right turn is easier, so the ego vehicle can act bolder.

TURNING_STRAIGHT_TIGHTEN_RATIO = 1.5
# when turning left or turning right, straight ahead CROSS_INTERVAL = RATIO * time the ego vehicle gets to the intersection / TIGHTEN_RATIO

ADJUST_INTERVAL_COUNT = 4
FRONT_ADJUST_DOWN_MARGIN = 0.2
LEFT_ADJUST_DOWN_MARGIN = 0.1
RIGHT_ADJUST_DOWN_MARGIN = 0.25
FRONT_ADJUST_MIN = 0.8
LEFT_ADJUST_MIN = 1.9
RIGHT_ADJUST_MIN = 0.65
# if recorded first appearance above this number, the ego vehicle will lower the threshold according to history
# to prevent waiting forever
# but the minimum value is ADJUST_MIN

INTERSECTION_CONSIDER_INTERVAL = 5.0
# vehicle is considered if the ego vehicle and the vehicle arrives at the same point between this time interval
# unit: second

INTERSECTION_DISTANCE_THRESHOLD = 12.0
# other vehicle & the ego vehicle's intersection within this distance will be considered
# unit: meter

INTERSECTION_EGO_DEFAULT_SPEED = 4.0
# sometimes the ego vehicle is waiting, so we couldn't use the real speed for calculation
# unit: m/s

TURN_DEVIATE_DEGREE = 45
# left/right intersection ray detect angle
# unit: degree

INTERSECTION_GET_CLOSE_DEGREE = 10
INTERSECTION_GET_CLOSE_Y = 1.8
# if the vehicle's degree is in 180+-DLOSE_DEGREE and relative Y is larger than CLOSE_Y
# the ego vehicle can move slowly
# unit: degree, meter

SAME_DIRECTION_MAX_THETA = 60.0
# when considering crossing vehicles, vehicles whose relative theta below this
# is considered in the same direction as the ego vehicle
# unit: degree

OPPOSITE_DIRECTION_MAX_THETA = 20.0
# when considering crossing vehicles, vehicles whose relative theta within [180-this, 180+this]
# is considered at the opposite direction to the ego vehicle
# unit: degree

DIFFERENT_DIRECTION_MIN_THETA = 30.0
# when considering other vehicles in junction,
# only whose relative theta above this value will be considered
# otherwise are considered as the vehicles moving in the same direction as the ego vehicle
# unit: degree

HIGHWAY_INTERSECTION_SAME_DIRECTION_MAX_Y = 1.5
# when considering crossing vehicles, vehicles whose x offset with in this on the highway
# is considered in the same direction as the ego vehicle
# unit: meter

JUNCTION_POS_OFFSET = 5.0
# if the vehicle's relative y < -JUNCTION_POS_OFFSET, it is described as 'on the left side of the junction'
# y > JUNCTION_POS_OFFSET, 'on the right side of the junction'
# unit: meter

TURNING_STOP_ANGLE = 10.0
# when turning left, if the ego vehicle's yaw degree is within [-TURNING_STOP_ANGLE, TURNING_STOP_ANGLE] + command_far's yaw,
# the ego vehicle should exit the intersection by driving straight.
# if passes TURNING_REVERSE_ANGLE, the turning process should be reversed
# and vice versa.
# unit: degree





######### CHANGE LANE ##########

LANE_CHANGE_STOP_OBSTACLE_DISTANCE = 10.0 
# must stop distance between the ego vehicle and the obstacle in its lane
# unit: meter

CHANGE_LANE_THRESHOLD = 20.0 
# obstacles within this distance are considered
# unit: meter

LANE_CLEAR_THRESHOLD_BASE = 15.0
LANE_FORWARD_THRESHOLD_BASE = 10.0
# if there's no vehicle back within LANE_CLEAR_THRESHOLD_BASE
# or front within LANE_FORWARD_BASE,
# the target lane is considered occupied.
# in real calculations, there're offsets based on different scenarios.
# unit: meter

SIDE_FRONT_CLEAR_ANGLE = 10.0
SIDE_FRONT_CLEAR_X_MIN = 0.5
SIDE_FRONT_CLEAR_X_MAX = 10.0
SIDE_FRONT_CLEAR_Y_MIN = 1.7
SIDE_FRONT_CLEAR_Y_MAX = 20.0
# if ego deviates from the lane to the farther side too much, this will not be accurate
# so side front clear is False in this case.
# these values are used when the vehicle is going to change lane from opposite direction
# in special scenarios, because some obstacles can not be refered 
# from the obstacle vehicle checking function which normally uses
# the area within these coordinates are checked as additional condition.
# unit: meter

LANE_CHANGE_NORMAL_OFFSET = 10.0
# offset added to LANE_CLEAR_THRESHOLD_BASE in normal situations
# unit: meter

LANE_CHANGE_OPPOSITE_FRONT_OFFSET = 30.0
# offset added to LANE_FORWARD_THRESHOLD_BASE in opposite situations
# unit: meter

LANE_CHANGE_DECLINE_RATIO = 3.0
# if the lateral distance to target lane is 0, 
# all danger distances are 1 / LANE_CHANGE_DECLINE_RATIO of normal ones.

LANE_CHANGE_FRONT_DANGER_DISTANCE = 8.0
LANE_CHANGE_BACK_DANGER_DISTANCE = 8.0
LANE_CHANGE_DANGER_INTERVAL = 1.5
EMERGENCY_VEHICLE_DANGER_INTERVAL = 1.2
# the ego vehicle mustn't change lane if
# the vehicle in target lane is closer than max(these distances, ideal_flow_speed * DANGER_INTERVAL) when it's
# in the front / at rear
# because yield to emergency vehicle has time limit
# the interval is shorter
# unit: meter

LANE_CHANGE_HIGHWAY_BACK_DANGER_DISTANCE = 40.0
LANE_CHANGE_PARKING_EXIT_BACK_DANGER_DISTANCE = 6.0
# special back danger distances
# parking exit danger is only used outside carla, normal danger distance is used in carla
# unit: meter
LANE_CHANGE_STATIC_IGNORE_DISTANCE = 2.5
# static vehicle farther than this distance at side rear will be ignored
# unit: meter

LANE_CHANGE_OPPOSITE_DANGER_DISTANCE = 40.0
# danger distance of opposite direction is farther because it is dangerous
# unit: meter

LANE_CHANGE_REAR_ALERT_THRESHOLD = -10.0
LANE_CHANGE_DISTANCE_ALERT_THRESHOLD = 20.0
# used to identify dangerous vehicle when the ego vehicle is in junction
# in some scenarios, common on highways,
# junctions have lanes
# so this is when this value is used
# unit: meter

LANE_CHANGE_BACK_CLEAR_THRESHOLD = 6.0
CARLA_LANE_CHANGE_BACK_CLEAR_THRESHOLD = 6.0
# once an obstacle vehicle's x is less than this
# target lane is not clear
# in carla, safe distance is longer
# unit: meter

LANE_CHANGE_FRONT_BACK_DIVIDER = -3.0
# divides obstacle vehicles in front / at rear by x
# unit: meter

OPPOSITE_SHOULDER_BLOCKER_MIN_X = -2.0
# if the ego vehicle is on opposite side and wants to change back
# vehicles on the shoulder whose x is larger than this are also obstacles
# unit: meter





####### CUT IN SCENARIOS #######

CUT_IN_DEVIATION = 6.0
# if the vehicle's deviation from its current lane below this value
# it is NOT considered cutting in
# unit: degree

CUT_IN_CONSIDER_DISTANCE = 20.0
# if the vehicle's distance is above this value
# it is NOT considered cutting in
# unit: meter

CUT_IN_STOP_DISTANCE = 10.0
# if the cutting in vehicle is within this distance
# the ego vehicle must stop
# unit: meter





######### TRAFFIC SIGN #########

TRAFFIC_SIGN_CONSIDER_RADIUS = 40.0
# traffic signs within this distance are considered
# unit: meter

STOP_SIGN_CONSIDER_RAIUS = 40.0
# stop signs within this distance are considered
# unit: meter

STOP_SIGN_SPEED_THRESHOLD = 2.0
CARLA_STOP_SIGN_SPEED_THRESHOLD = 0.1
STOP_SIGN_DISTANCE_THRESHOLD = 4.0
CARLA_STOP_SIGN_DISTANCE_THRESHOLD = 8.0
# if the ego vehicle is within this distance and below this speed,
# it is considered "stopped" and is able to start up
# and drive away.
# in carla, the criretia is dist < 4.0m, spd < 0.1 m/s.
# outside carla, better not set below 8m
# unit: m/s, meter

STOP_SIGN_STOP_DISTANCE = 4.0
STOP_SIGN_BRAKE_DISTANCE = 15.0
# the ego vehicle should stop/brake within this distance to the stop sign
# unit: meter

STOP_SIGN_AHEAD_THRESHOLD = -2.6
# speed limits whose relative x > this and not affecting the ego vehicle
# can be considered 'cleared'
# unit: meter

RED_LIGHT_STOP_DISTANCE = 6.0
RED_LIGHT_BRAKE_DISTANCE = 16.0
RED_LIGHT_STOP_RADIUS = 12.0
# the ego vehicle should stop/drive forward and stop
# if its distance to junction within this distance
# under red/yellow traffic light
# sometimes the traffic light is ahead of junction (which does not make sense to me), eg. in some TJunction scenarios
# so the ego vehicle should also stop when it is within a distance from the affecting red light.
# unit: meter

SPEED_LIMIT_CONSIDER_RAIUS = 40.0
# speed limits within this distance are considered
# unit: meter

SPEED_LIMIT_VALID_THRESHOLD = -12.0
# once the ego vehicle entered a new road
# speed limit signs behind this x are invalid
# the position threshold is loose because of HighwayExit_Town06_Route291_Weather5
# unit: meter

SPEED_LIMIT_AHEAD_THRESHOLD = 1.0
# speed limits whose relative x > this is considered 'ahead'
# otherwise 'passed'
# unit: meter





####### ENVIRONMENT INFO #######

NOON_SUN_ALTITUDE = 40.0
DUSK_SUN_ALTITUDE = 20.0
# if the sun altitude is above NOON one, the time is noon
# between them, just day
# below DUST one, the time is dawn or dusk

CLOUDY_THRESHOLD = 50.0
# weather which cloudiness above this is considered cloudy

HEAVY_RAIN_THRESHOLD = 80.0
MODERATE_RAIN_THRESHOLD = 50.0
LIGHT_RAIN_THRESHOLD = 20.0
# percipitation threshold for rain identifications

FOGGY_THRESHOLD = 40.0
# fog density above this is considered foggy

WET_THRESHOLD = 80.0
# wetness above this is considered wet

ROAD_FLOOD_THRESHOLD = 80.0
ROAD_PUDDLE_THRESHOLD = 40.0
# persipitation deposit thresholds for identifying road flood / puddles

PARKED_VEHICLE_MAX_X = 40.0
# max x of parked vehicle considered
# unit: meter

BLIND_SPOT_MAX_DISTANCE = 30.0
BLIND_SPOT_MIN_X = 1.5
# condisions restricting blind spots' position
# unit: meter