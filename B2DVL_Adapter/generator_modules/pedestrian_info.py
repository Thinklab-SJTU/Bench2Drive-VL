from .offline_map_calculations import *
from .graph_utils import *
from .hyper_params import *

def process_pedestrians(self, other_pedestrians, important_objects, object_infos):
    """
    How many pedestrians are there? And where they are roughly (front, left, ...)
    """
    qas_pedestrian = []
    close_pedestrians = []

    object_tags = []
    for pedestrian in other_pedestrians:
        if (
            pedestrian['num_points'] < MIN_PEDESTRIAN_NUM_POINT  # Not enough LiDAR points
            or pedestrian['position'][0] < 0  # Too far behind ego vehicle
            or pedestrian['position'][0] > PEDESTRIAN_CONSIDER_DISTANCE  # Too far ahead of ego vehicle
        ):
            # Not enough LiDAR points or far away or occluded
            continue

        close_pedestrians.append(pedestrian)

        # Determine rough position of pedestrian relative to ego vehicle
        important_object_str = get_pedestrian_str(pedestrian)

        important_objects.append(important_object_str)

        # Project pedestrian points and center onto the image plane
        # projected_points, projected_points_meters = project_all_corners(pedestrian, self.CAMERA_MATRIX, self.WORLD2CAM_FRONT)
        project_dict = get_project_camera_and_corners(pedestrian, self.CAM_DICT)

        # Generate key-value pair for the pedestrian object
        key, value = self.generate_object_key_value(
            id=pedestrian['id'],
            category='Pedestrian',
            visual_description=f'Pedestrian',
            detailed_description=important_object_str,
            object_count=len(object_infos),
            obj_dict=pedestrian,
            projected_dict=project_dict
        )
        object_infos[key] = value
        object_tags.append(key)

    question = "How many pedestrians are there?"
    s_or_no_s = 's' if len(close_pedestrians) > 1 else ''
    are_or_is = 'are' if len(close_pedestrians) > 1 else 'is'

    if len(close_pedestrians) == 0:
        answer = "There are no pedestrians."
    else:
        answer = f"There {are_or_is} {len(close_pedestrians)} pedestrian{s_or_no_s}."

    # Add the question and answer to the conversation
    self.add_qas_questions(qa_list=qas_pedestrian,
                            qid=1,
                            chain=5,
                            layer=0,
                            qa_type='perception',
                            connection_up=[(6,0)],
                            connection_down=-1,
                            question=question,
                            answer=answer,
                            object_tags=object_tags)

    return qas_pedestrian, important_objects, object_infos