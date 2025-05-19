from .VLMInterface import VLMInterface
from PIL import Image
import torch
from openai import OpenAI
from .my_tokens import *
from .interact_utils import *
import time

vlm_str = "Qwen/Qwen2.5-VL-72B-Instruct"
# vlm_str = "Pro/Qwen/Qwen2-VL-7B-Instruct"
vlm_url = "https://api.siliconflow.cn/v1"
MAX_RETRIES = 32
WAIT_TIME = 2

def get_chat_response(client, model, messages, max_tokens=4096):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            print(f"VLM API error: {e}. Retrying {retries + 1}/{MAX_RETRIES}...")
            time.sleep(WAIT_TIME)
            retries += 1
        # except Exception as e:
        #     print(f"Unexpected error: {e}")
        #     break
    print("VLM API max retries reached. Aborting this question.")
    return None

def image_template(image_path, use_base64):
    return {
        "type": "image_url",
        "image_url": {
            "url": image_path if use_base64 else local_image_to_base64(image_path)
        }
    }

class VLMAPIInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing VLM API Worker {gpu_id}...")
        # Load model, weights, and allocate resources here
        self.in_carla = in_carla
        self.use_bev = use_bev
        self.use_all_cameras = use_all_cameras
        self.input_window = input_window
        self.no_history = no_history
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.use_base64 = use_base64

        self.multi_image_flag = (self.use_all_cameras or (self.no_history == False and self.input_window > 1))

        self.client = OpenAI(
            api_key=VLM_TOKEN,
            base_url=vlm_url
        )

        print(f"Worker {gpu_id} is going to use {vlm_str} from {vlm_url}.")
    
    def get_image_descriptions(self, images_dict, image_frame_list, start_frame, end_frame):
        """
        Returns content list about descriptions of all images in range (start_frame, end_frame]
        """
        if self.in_carla:
            return get_carla_image_descriptions(images_dict=images_dict,
                                                image_frame_list=image_frame_list,
                                                start_frame=start_frame,
                                                end_frame=end_frame,
                                                frame_rate=self.frame_rate,
                                                template_func=image_template,
                                                use_all_cameras=self.use_all_cameras,
                                                use_bev=self.use_bev,
                                                use_base64=self.use_base64)
        else:
            return get_image_descriptions(images_dict=images_dict,
                                          image_frame_list=image_frame_list,
                                          start_frame=start_frame,
                                          end_frame=end_frame,
                                          frame_rate=self.frame_rate,
                                          template_func=image_template,
                                          use_all_cameras=self.use_all_cameras,
                                          use_base64=self.use_base64)

    def interact(self, bubble, conversation):
        # torch.cuda.set_device(self.gpu_id)
        # self.device = torch.device(f"cuda:{self.gpu_id}")

        input_conversation = []
        
        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())

        # print(f'[debug] image_frame_list = {image_frame_list}')

        start_frame = bubble.frame_number
        current_frame = bubble.frame_number
        if conversation is not None and len(conversation) > 0:
            start_frame = conversation[0].frame_number
        
        # print(f'[debug] start_frame = {start_frame}')

        prev_frame = -1

        # context
        context_str = ""
        if self.no_history == False:
            # all in one context
            context_str = ""
            context_bb = {
                "role": "user", # some models don't use context as input, for example qwen2
                "content": []
            }

            for bb in conversation:
                is_user = (bb.actor == "User")
                
                if is_user and prev_frame < bb.frame_number:
                    image_content, _ = self.get_image_descriptions(images_list, image_frame_list,
                                                        prev_frame, bb.frame_number)
                    prev_frame = bb.frame_number

                    if context_str is not None and context_str != "":
                        frame_content = {
                            "type": "text",
                            "text": context_str
                        }
                        context_bb['content'].append(frame_content)
                        context_str = ""

                    context_bb['content'].extend(image_content)
                
                header = "Q" if is_user else "A"
                context_str += f"{header}(frame {bb.frame_number}): {bb.words}\n"
            
            if context_str is not None and context_str != "":
                frame_content = {
                    "type": "text",
                    "text": context_str
                }
                context_bb['content'].append(frame_content)
                context_str = ""
            
            if len(context_bb['content']) > 0:
                input_conversation.append(context_bb)
                
        # question
        bb_dict = {
            "role": "user",
            "content": []
        }
        if prev_frame < current_frame:
            image_content, _ = self.get_image_descriptions(images_list, image_frame_list,
                                                        prev_frame, current_frame)
            prev_frame = current_frame
            bb_dict['content'].extend(image_content)
        bb_dict['content'].append({
            "type": "text",
            "text": f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        })
        input_conversation.append(bb_dict)

        # print(f"[debug] conversation = {input_conversation}")

        input_image_files = []
        for frame_number in image_frame_list:
            if (frame_number < current_frame and self.no_history == False) or \
                frame_number == current_frame:
                if self.use_all_cameras:
                    for key in images_list[frame_number].keys():
                        if key in ['CAM_FRONT_CONCAT', 'CAM_BACK_CONCAT']:
                            input_image_files.append(images_list[frame_number][key])
                else:
                    input_image_files.append(images_list[frame_number]['CAM_FRONT'])
        
        # print(input_image_files)
        # print(input_conversation)

        response = get_chat_response(self.client, vlm_str, input_conversation)

        time.sleep(0.5) # control frequency for web api

        if response is not None:
            result = response.choices[0].message.content
        else:
            result = "VLM API Error."
        
        if isinstance(result, list):
            result = result[0]
        
        return result