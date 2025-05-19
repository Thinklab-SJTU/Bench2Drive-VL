
class VLMInterface:
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, input_window: int, 
                    frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        """
        Initialize the model with specific flags and resources.
        :param use_all_cameras: Flag indicating whether to use all cameras.
        :param gpu_id: ID of the GPU to use for initialization.
        """
        print(f"Default initialization: use_all_cameras={use_all_cameras}, gpu_id={gpu_id}, returning gt")

    def interact(self, bubble, context):
        """
        Interact with the VLM using the given Bubble data.
        :param bubble_data: Input Bubble data structure.
        :return: Default response.
        """
        return f"Ground Truth: {bubble.gt}"
