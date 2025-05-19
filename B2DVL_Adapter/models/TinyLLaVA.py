from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import base64
from io import BytesIO

def image_template(image_path, use_base64):
    return ""

class TinyLLaVAInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing TinyLLaVA on GPU {gpu_id}...")
        # Load model, weights, and allocate resources here
        # Attention: TinyLLaVA only supports SINGLE IMAGE inference!
        # so we only use the last one.
        self.in_carla = in_carla
        self.use_bev = use_bev
        self.use_all_cameras = use_all_cameras
        self.input_window = input_window
        self.no_history = no_history
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.use_base64 = use_base64

        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        self.multi_image_flag = (self.use_all_cameras or (self.no_history == False and self.input_window > 1))

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        model_config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, 
                                                       model_max_length = model_config.tokenizer_model_max_length,
                                                       padding_side = model_config.tokenizer_padding_side)

        print(f"TinyLLaVA loaded on GPU {gpu_id} successfully")
    
    
    
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

        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        input_conversation = []
        
        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())
        input_image_files = []

        # print(f'[debug] image_frame_list = {image_frame_list}')

        start_frame = bubble.frame_number
        current_frame = bubble.frame_number
        if conversation is not None and len(conversation) > 0:
            start_frame = conversation[0].frame_number
        
        # print(f'[debug] start_frame = {start_frame}')

        prev_frame = -1

        # context
        all_context_str = ""
        if self.no_history == False:
            # all in one context
            context_str = ""

            for bb in conversation:
                is_user = (bb.actor == "User")
                
                if is_user and prev_frame < bb.frame_number:
                    image_content, image_dirs = self.get_image_descriptions(images_list, image_frame_list,
                                                                            prev_frame, bb.frame_number)
                    input_image_files.extend(image_dirs)
                    prev_frame = bb.frame_number

                    if context_str is not None and context_str != "":
                        # frame_content = {
                        #     "type": "text",
                        #     "text": context_str
                        # }
                        all_context_str += context_str
                        context_str = ""

                    all_context_str += image_content
                
                header = "Q" if is_user else "A"
                all_context_str += f"{header}(frame {bb.frame_number}): {bb.words}\n"
            
            if context_str is not None and context_str != "":
                # frame_content = {
                #     "type": "text",
                #     "text": context_str
                # }
                all_context_str += context_str
                context_str = ""
                
        # question
        if prev_frame < current_frame:
            image_content, image_dirs = self.get_image_descriptions(images_list, image_frame_list,
                                                                    prev_frame, current_frame)
            input_image_files.extend(image_dirs)
            prev_frame = current_frame
            all_context_str += image_content
        all_context_str += f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        input_conversation = all_context_str

        print(f"[debug] conversation = {input_conversation}")
        print(f'[debug] input_image_files = {input_image_files}')

        # input_images = []
        # for file in input_image_files:
        #     if self.use_base64:
        #         if file.startswith('data:'):
        #             file = file.split(',', 1)[1]
        #         image_data = base64.b64decode(file)
        #         image = Image.open(BytesIO(image_data))
        #     else:
        #         image = Image.open(file)
        #     input_images.append(image)
        input_image = input_image_files[-1] if len(input_image_files) > 0 else None
        # print(input_image_files)
        # print(input_conversation)
        output_text, genertaion_time = self.model.chat(prompt=input_conversation, 
                                                       image=input_image, 
                                                       tokenizer=self.tokenizer)

        result = output_text
        print(f'[debug] A: {result}')
        if isinstance(result, list):
            result = result[0]
        
        return result