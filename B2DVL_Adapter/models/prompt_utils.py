
def get_traj_desc():
    return "The objects' historical trajectory in the figure is represented by a polyline, " +\
           "with one segment every 0.5 seconds."

def get_concat_img_desc(frame_number, frame_rate):
    return f"The two concatenated images below are from all cameras taken from the ego vehicle;s perspective on frame {frame_number}, " +\
                                f"notice that there're {frame_rate} frames per second."

def get_front_img_desc(frame_number, frame_rate):
    return f"This image is from front camera taken from the ego vehicle's perspective on frame {frame_number}, " +\
                                f"notice that there're {frame_rate} frames per second."

def get_bev_img_desc(frame_number, frame_rate):
    return f"This figure is a bird's-eye view of a driving scenario, " +\
            "with the ego vehicle located at the center of the image. " +\
            "The upward direction in the image represents the direction in which the ego vehicle is moving forward, " +\
            "while the downward direction represents the rear. " +\
            "(If the traffic signal doesn't control the ego vehicle's lane or at rear, ignore it.) " +\
           f"It is taken on frame {frame_number}, notice that there're {frame_rate} frames per second."

def get_concat_traj_img_desc(frame_number, frame_rate):
    return get_concat_img_desc(frame_number, frame_rate) + " " + get_traj_desc()

def get_front_traj_img_desc(frame_number, frame_rate):
    return get_front_img_desc(frame_number, frame_rate) + " " + get_traj_desc()

def get_bev_traj_img_desc(frame_number, frame_rate):
    return get_bev_img_desc(frame_number, frame_rate) + " " + get_traj_desc()