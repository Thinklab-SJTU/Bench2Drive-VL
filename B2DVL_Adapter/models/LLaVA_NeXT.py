from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import base64
from io import BytesIO

def image_template(image_path, use_base64):
    return {
        "type": "image"
    }

class LLaVANeXTInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing LLaVANeXT on GPU {gpu_id}...")
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

        # Load LLaVA-NeXT model and processor
        # if self.multi_image_flag:
        #     # multi-image inference
        #     self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=torch.float16, device_map="auto")
        #     self.processor = AutoProcessor.from_pretrained(self.model_path)
        # else:
        #     # single-image inference
        #     self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        #     self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
        #     # self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=torch.float16, device_map="auto")
        #     # self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        #     # self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        self.processor = LlavaNextProcessor.from_pretrained(model_path)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        self.model.to(f"cuda:{gpu_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        # original
        # model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path).to(device)
        # processor = AutoProcessor.from_pretrained(args.model_path)

        print(f"LLaVANeXT loaded on GPU {gpu_id} successfully")
    
    
    
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
        image_dirs = []
        
        images_list = bubble.get_full_images()
        image_frame_list = sorted(images_list.keys())

        # print(f'[debug] image_frame_list = {image_frame_list}')

        start_frame = bubble.frame_number
        current_frame = bubble.frame_number
        if conversation is not None and len(conversation) > 0:
            start_frame = conversation[0].frame_number
        
        # print(f'[debug] start_frame = {start_frame}')

        prev_frame = -1

        context_str = ""
        if self.no_history == False:
            # all in one context
            context_str = ""
            context_bb = {
                "role": "context",
                "content": []
            }

            for bb in conversation:
                is_user = (bb.actor == "User")

                if is_user and prev_frame < bb.frame_number:
                    image_content, images = self.get_image_descriptions(images_list, image_frame_list,
                                                                        prev_frame, bb.frame_number)
                    image_dirs.append(images)
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
            
            input_conversation.append(context_bb)
        
        # question
        bb_dict = {
            "role": "user",
            "content": []
        }
        if prev_frame < current_frame:
            image_content, images = self.get_image_descriptions(images_list, image_frame_list,
                                                                prev_frame, current_frame)
            image_dirs.append(images)
            prev_frame = current_frame
            bb_dict['content'].extend(image_content)
        bb_dict['content'].append({
            "type": "text",
            "text": f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        })
        input_conversation.append(bb_dict)

        # print(f"[debug] conversation = {input_conversation}")
        # print(f'[debug] input_image_files = {input_images}')

        input_images = []
        for file in image_dirs:
            if self.use_base64:
                image_data = base64.b64decode(file)
                image = Image.open(BytesIO(image_data))
            else:
                image = Image.open(file)
            input_images.append(image)
        
        # print(input_image_files)
        # print(input_conversation)

        prompts = self.processor.apply_chat_template(input_conversation, add_generation_prompt=True)

        inputs = self.processor(images=input_images, text=prompts, 
                                padding=True, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, max_new_tokens=1024)

        # if self.multi_image_flag:
        #     result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # else:
        #     result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        result = self.processor.decode(generate_ids[0], skip_special_tokens=True)
        
        # # original
        # inputs = self.processor(images=Image.open(image_for_batch[idx]), text=f"[INST] {prompts[i]} [/INST]", return_tensors="pt").to(device)
        # generate_ids = self.model.generate(**inputs, max_new_tokens=300)
        # result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        result = result.split("[/INST]")[-1].strip()
        result = result.replace("<\s>", "")
        result = result.replace("<\s", "")
        result = result.replace("<\\", "")
        # print(f'[debug] A: {result}')
        if isinstance(result, list):
            result = result[0]
        
        return result