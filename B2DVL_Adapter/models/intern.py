import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
import base64
from io import BytesIO

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str) and image_file.startswith("data:image/"):
        header, base64_data = image_file.split(",", 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def image_template(image_path, use_base64):
    return ""

class InternVLInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing InternVL on GPU {gpu_id}...")
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

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       trust_remote_code=True,
                                                       use_fast=False)

        self.multi_image_flag = (self.use_all_cameras or (self.no_history == False and self.input_window > 1))

        print(f"InternVL loaded on GPU {gpu_id} successfully")
    
    
    
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
        # set the max number of tiles in `max_num`
        pixel_values = load_image(input_image, max_num=12).to(torch.bfloat16).to(self.device)
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        response = self.model.chat(self.tokenizer, pixel_values, input_conversation, generation_config)

        result = response
        print(f'[debug] A: {result}')
        if isinstance(result, list):
            result = result[0]
        
        return result