from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

def image_template(image_path, use_base64):
    return {
        "type": "image",
        "image": f"{image_path}"
    }

class GemmaInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing Gemma on GPU {gpu_id}...")
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

        torch.cuda.set_device(self.gpu_id)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        self.multi_image_flag = (self.use_all_cameras or (self.no_history == False and self.input_window > 1))
        
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, device_map=f"cuda:{self.gpu_id}", torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        print(f"Gemma loaded on GPU {gpu_id} successfully")
    
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

        # print(f'[debug] image_frame_list = {image_frame_list}')

        start_frame = bubble.frame_number
        current_frame = bubble.frame_number
        if conversation is not None and len(conversation) > 0:
            start_frame = conversation[0].frame_number
        
        # print(f'[debug] start_frame = {start_frame}')

        prev_frame = -1

        # context
        context_str = ""
        context_bb = {
            "role": "user",
            "content": []
        }
        if self.no_history == False:
            # all in one context
            context_str = ""

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
            
            # if len(context_bb['content']) > 0:
            #     input_conversation.append(context_bb)
                
        # question
        bb_dict = context_bb
        # bb_dict = {
        #     "role": "user",
        #     "content": []
        # }
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
        # print(output)
        max_retries = 32
        retry_count = 0

        while retry_count < max_retries:
            try:
                inputs = self.processor.apply_chat_template(
                    input_conversation, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(self.model.device)

                input_len = inputs["input_ids"].shape[-1]

                generation = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
                generation = generation[0][input_len:]

                decoded = self.processor.decode(generation, skip_special_tokens=True)

                result = decoded if isinstance(decoded, str) else decoded[0]
                break

            except Exception as e:
                print(f"[Retry {retry_count+1}/{max_retries}] VLM Error: {e}")
                retry_count += 1
                
                try:
                    del self.model
                    del self.processor
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                    self.initialize(self.gpu_id, self.use_all_cameras, self.no_history,
                                    self.input_window, self.frame_rate, self.model_path,
                                    self.use_bev, self.in_carla, self.use_base64)
                except Exception as init_e:
                    print(f"[Retry {retry_count}] Reinitialization failed: {init_e}")
        
        # print(result)
        if isinstance(result, list):
            result = result[0]
        
        return result