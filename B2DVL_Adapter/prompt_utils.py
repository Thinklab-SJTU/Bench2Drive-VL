COLOR_PROMPT = """
Notice that the VLM might answer a different but similar color, which is tolerable.
eg. Silver - White, Maroon - Red, Olive - Yellow - Orange. Black car - Police car.
"""

SYSTEM_PROMPT = f"""
You're evaluating answers about autonomous driving given by other visual-language models. 
The question, its ground truth and anwser to be reviewed will be given below.
Rate the answer whith a score in 0-100, and give reasons. use JSON output.
{COLOR_PROMPT}
"""
NORMAL_EXAMPLE = """
EXAMPLE INPUT 1
question: "What is the rough moving status of the black cat to the front?"
gt: "It is steering to its right, driving slowly."
anwser from VLM: "It is turning right, driving slowly."

EXAMPLE JSON OUTPUT 1
{
    "reason": "This answer is perfectly accurate",
    "score": 100
}

EXAMPLE INPUT 2
question: "What is the rough moving status of the black cat to the front?"
gt: "It is steering to its right, driving slowly."
anwser from VLM: "It is going straight, driving at normal speed."

EXAMPLE JSON OUTPUT 2
{
    "reason": "Both direction and speed are not correct.",
    "score": 0
}
"""

LISTED_PROMPT = f"""
You're evaluating answers about autonomous driving given by other visual-language models. 
The question, its ground truth and anwser to be reviewed will be given below.
Notice that the question and answer may involve multiple objects,
but we only focus on one object which will be specified. DO NOT CARE ABOUT OTHER OBJECTS.
{COLOR_PROMPT}
Rate the answer whith a score in 0-100, and give reasons. use JSON output. DO NOT CARE ABOUT OTHER OBJECTS.
"""

LISTED_EXAMPLE = """
EXAMPLE INPUT 1
question: "The list of important vehicles in current scene is 'the navy car that is to the front left of the ego vehicle(c49442, CAM_FRONT, CAM_FRONT_LEFT), the black SUV that is to the rear left of the ego vehicle(c49443, CAM_BACK), the white car that is to the front left of the ego vehicle(c49448, CAM_FRONT_LEFT.' List deep colored vehicles among them."
gt: "the navy car that is to the front left of the ego vehicle(c49442, CAM_FRONT, CAM_FRONT_LEFT), the black SUV that is to the rear left of the ego vehicle(c49443, CAM_BACK)."
anwser from VLM: "the black SUV that is to the rear left of the ego vehicle(c49443, CAM_BACK), the white car that is to the front left of the ego vehicle(c49448, CAM_FRONT_LEFT."
focus: "the navy car that is to the front left of the ego vehicle <c49442, <CAM_FRONT,...,...>, <CAM_FRONT_LEFT,...,...>>"

EXAMPLE JSON OUTPUT 1
{
    "reason": "According to gt, the navy car is deep colored, but VLM didn't include it.",
    "score": 0
}

EXAMPLE INPUT 2
question: "What are the rough moving speed and moving direction of the important vehicles?"
gt: "The navy car that is to the front left of the ego vehicle(<c49442<CAM_FRONT,378.2,581.1><CAM_FRONT_LEFT,1767.2,615.2>>) is driving at a normal speed in the same direction of the ego vehicle, going straight. The olive car that is to the front left of the ego vehicle(<c49441<CAM_FRONT,651.9,482.4>>) is driving at a normal speed in the same direction of the ego vehicle, going straight. The maroon car that is to the front left of the ego vehicle(<c49448<CAM_FRONT,558.0,469.8>>) is driving at a normal speed in the opposite direction of the ego vehicle, going straight."
anwser from VLM: "A(frame 30): \"The orange car is moving forward at a moderate speed, likely maintaining its lane position. The blue car is also moving forward, slightly behind the orange car, indicating it might be following the same lane. The blue car appears to be moving at a similar speed to the orange car.\"\n\nFor the ego vehicle:\n- Moving forward at a moderate speed, likely maintaining its lane position.\n- Following the orange car, which is moving forward at a moderate speed.\n\nThe other vehicles (yellow sign, garages, trees, buildings, streetlights, and sidewalk) do not have specific speeds or directions mentioned in the description."
focus: "The olive car that is to the front left of the ego vehicle(<c49441<CAM_FRONT,651.9,482.4>>)"

EXAMPLE JSON OUTPUT 2
{
    "reason": "According to gt, the olive car, which the vlm recognized as the orange car, is driving at a normal speed in the same direction of the ego vehicle, going straight, which fits the vlm's description, it's moving forward at a moderate speed, likely maintaining its lane position.",
    "score": 100
}

EXAMPLE INPUT 3
question: "Basic infos of the actors: [..] What are the rough moving speed and moving direction of the important vehicles?"
gt: "The black car that is to the front of the ego vehicle(<c14785<CAM_FRONT,799.8,531.8>>) is driving slowly in the same direction of the ego vehicle, going straight."
answer from VLM: "Based on the information provided, here are the rough moving speeds and directions for the important vehicles:\n\n1. **Ego Vehicle (Maroon SUV)**:\n   - Speed: 1.0 m/s\n   - Direction: Same direction as the ego vehicle\n\n2. **Black Car**:\n   - Speed: 2.1 m/s\n   - Direction: Same direction as the ego vehicle\n\n3. **Maroon SUV**:\n   - Speed: 1.0 m/s\n   - Direction: Same direction as the ego vehicle\n\nThe ego vehicle is moving at a speed of 0.8 m/s, while the black car is moving at a speed of 2.1 m/s. Both the maroon SUV and the black car are moving in the same direction as the ego vehicle."
focus: "The black car that is to the front of the ego vehicle"

EXAMPLE JSON OUTPUT 3
{
    "reason": "The VLM correctly identified the direction of the black car as the same as the ego vehicle, which matches the gt. However, the VLM provided a specific speed (2.1 m/s) for the black car, whereas the gt only mentions it is driving slowly without specifying the speed.",
    "score": 90
}

again, DO NOT CARE ABOUT NON-FOCUS OBJECTS.
"""

FUTURE_PROMPT = """
In this task, I will provide answers for some future frames related to this question. If the VLM's answer for the current frame is incorrect but it correctly predicts the answers for future frames, a small portion of the score (e.g., 5-10 points) can be deducted at your discretion. Further in the future, more the score deducted. 
"""

IMPORTANT_OBJECT_PROMPT1 = """
In this sentence, please identify each vehicle and object, convert it into a list and clear their ids (content within parentheses), use JSON output.

EXAMPLE INPUT 1
The black car that is to the front of the ego vehicle(<c1741<CAM_FRONT,1099.8,642.8><CAM_FRONT_RIGHT,-493.6,729.8>>), the junction, the gray car that is to the front right of the ego vehicle(<c1740<CAM_FRONT_RIGHT,397.4,496.8><CAM_FRONT,1677.0,505.8>>), the black car that is to the front right of the ego vehicle(<c1748<CAM_FRONT,1367.2,491.6><CAM_FRONT_RIGHT,134.6,494.1>>), the black car that is to the front right of the ego vehicle(<c1749<CAM_FRONT,1500.2,486.1><CAM_FRONT_RIGHT,274.8,484.2>>).

EXAMPLE JSON OUTPUT 1
{
    "list": ["The black car that is to the front of the ego vehicle", "the junction", "the gray car that is to the front right of the ego vehicle", "the black car that is to the front right of the ego vehicle 1", "the black car that is to the front right of the ego vehicle 2"]
}
"""

IMPORTANT_OBJECT_PROMPT2 = f"""
Next, you will be provided with a list and a sentence. Your task is to identify the objects from the list.
If it is likely an item included in the list, use description from the list, otherwise use original description. 
{COLOR_PROMPT}
Return the result as a list in JSON format.

EXAMPLE INPUT 1
{{
    "list": ["The black car that is to the front of the ego vehicle", "the junction", "the gray car that is to the front right of the ego vehicle", "the black car that is to the front right of the ego vehicle 1", "the black car that is to the front right of the ego vehicle 2"]
    "sentence": "In this context, the important objects in the scene are:\n\n1. The road itself, which is the primary path for the ego vehicle to follow, and the junction ahead.\n2. The car in front of the ego vehicle, which is the immediate object of interest for the vehicle's autonomous driving system.\n3. The streetlights on the side of the road, which provide illumination for the vehicle's sensors and help guide the vehicle."
}}

EXAMPLE JSON OUTPUT 1
{{
    "list": ["the road", "the junction", "The black car that is to the front of the ego vehicle", "the streetlights"]
}}
"""

KEY_43_GT_PROMPT = """
Your task is to extract a speed key and a direction key from below.
direction_keys = {'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'DEVIATE_LEFT', 'DEVIATE_RIGHT'}
'GO_STRAIGHT' is used in both following lane scenarios and going straight in junction scenarios.
'DEVIATE_LEFT' and 'DEVIATE_RIGHT' are used when the ego vehicle wants to occupy a position slightly to the left/right within the current lane.
speed_keys = {'KEEP', 'ACCELERATE', 'DECELERATE', 'STOP'}

EXAMPLE INPUT 1
"The ego vehicle should align to the target lane beacuse the ego vehicle is still changing lane to circumvent the obstacle. The ego vehicle doesn't need to brake because it is changing to the left lane now."

EXAMPLE JSON OUTPUT 1
['CHANGE_LANE_LEFT', 'KEEP']

EXAMPLE INPUT 2
"A: Okay, The ego vehicle needs to initiate a lane change to the *right*. Here's a prioritized breakdown of the steps:\n\n1.  **Accelerate:** The ego vehicle *must* begin to accelerate. It's currently at 0.0 m/s and needs a positive velocity to execute the lane change and follow the road's curve. A moderate acceleration is appropriate, not excessively fast, but enough to gain momentum. 2.  **Signal Intent:** Simultaneously with initiating acceleration, the ego vehicle should activate its right turn signal to indicate its intention to change lanes. This provides a warning to other vehicles."

EXAMPLE JSON OUTPUT 2
['CHANGE_LANE_RIGHT', 'ACCELERATE']

EXAMPLE INPUT 3
"The ego vehicle should continue turning right at the current intersection."

EXAMPLE OUTPUT 3
['TURN_RIGHT', 'KEEP']

EXAMPLE INPUT 4
"The ego vehicle should deviate slightly to the right from its current lane to avoid the invading vehicles."

EXAMPLE OUTPUT 4
['DEVIATE_RIGHT', 'KEEP']
"""

KEY_43_A_PROMPT = """
Your task is to extract a speed key and a direction key from below.
direction_keys = {'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT', 'CHANGE_LANE', 'GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'DEVIATE_LEFT', 'DEVIATE_RIGHT'}
'GO_STRAIGHT' is used in both following lane scenarios and going straight in junction scenarios.
'DEVIATE_LEFT' and 'DEVIATE_RIGHT' are used when the ego vehicle wants to occupy a position slightly to the left/right within the current lane.
'CHANGE_LANE' is used when the answer mentioned lane changing but didn't specify the answer.
speed_keys = {'KEEP', 'ACCELERATE', 'DECELERATE', 'STOP'}

EXAMPLE INPUT 1
"The ego vehicle should align to the target lane beacuse the ego vehicle is still changing lane to circumvent the obstacle. The ego vehicle doesn't need to brake because it is changing to the left lane now."

EXAMPLE JSON OUTPUT 1
['CHANGE_LANE_LEFT', 'KEEP']

EXAMPLE INPUT 2
"A: Okay, The ego vehicle needs to initiate a lane change to the *right*. Here's a prioritized breakdown of the steps:\n\n1.  **Accelerate:** The ego vehicle *must* begin to accelerate. It's currently at 0.0 m/s and needs a positive velocity to execute the lane change and follow the road's curve. A moderate acceleration is appropriate, not excessively fast, but enough to gain momentum. 2.  **Signal Intent:** Simultaneously with initiating acceleration, the ego vehicle should activate its right turn signal to indicate its intention to change lanes. This provides a warning to other vehicles."

EXAMPLE JSON OUTPUT 2
['CHANGE_LANE_RIGHT', 'ACCELERATE']

EXAMPLE INPUT 3
"The ego vehicle should continue turning right at the current intersection."

EXAMPLE OUTPUT 3
['TURN_RIGHT', 'KEEP']

EXAMPLE INPUT 4
"The ego vehicle should change lanes or deviate from the lane to maintain safety and avoid potential collisions with the faster-moving police cars and trucks."

EXAMPLE OUTPUT 4
['CHANGE_LANE', 'KEEP']
"""

MISINTERPRET_COLORS = {
    'navy': ['blue'],
    'olive': ['orange', 'yellow'],
    'maroon': ['red'],
    'silver': ['white']
}

def check_misinterpret_color(raw_str):
    lower_str = raw_str.lower()
    hint_str = ""
    for original_color in MISINTERPRET_COLORS.keys():
        if original_color in lower_str:
            hint_str += f"Color '{original_color}' might be mistaken for '{MISINTERPRET_COLORS[original_color]}'. "
    if hint_str != "":
        hint_str = f"Tolerable mistakes: {hint_str}"
    return hint_str