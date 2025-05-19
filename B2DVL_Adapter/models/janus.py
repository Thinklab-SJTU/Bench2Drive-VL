from .VLMInterface import VLMInterface
from .interact_utils import get_image_descriptions, get_carla_image_descriptions
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import torch
import base64
from io import BytesIO

def image_template(image_path, use_base64):
    return "<image_placeholder>"

class JanusProInterface(VLMInterface):
    def initialize(self, gpu_id: int, use_all_cameras: bool, no_history: bool, 
                    input_window: int, frame_rate: int, model_path: str, use_bev: bool=False, 
                    in_carla: bool=False, use_base64: bool=False):
        print(f"Initializing JanusPro on GPU {gpu_id}...")
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
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vl_gpt.to(self.device)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

        # original
        # model = LlavaNextForConditionalGeneration.from_pretrained(args.model_path).to(device)
        # processor = AutoProcessor.from_pretrained(args.model_path)

        print(f"JanusPro loaded on GPU {gpu_id} successfully")
    
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
        context_str = ""
        if self.no_history == False:
            # all in one context
            context_str = ""
            context_bb = {
                "role": "<|Context|>",
                "content": ""
            }

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
                        context_bb['content'] += context_str
                        context_str = ""

                    context_bb['content'] += image_content
                
                header = "Q" if is_user else "A"
                context_str += f"{header}(frame {bb.frame_number}): {bb.words}\n"
            
            if context_str is not None and context_str != "":
                # frame_content = {
                #     "type": "text",
                #     "text": context_str
                # }
                context_bb['content'] += context_str
                context_str = ""
            
            if context_bb['content'] != "":
                input_conversation.append(context_bb)
                
        # question
        bb_dict = {
            "role": "<|User|>",
            "content": ""
        }
        if prev_frame < current_frame:
            image_content, image_dirs = self.get_image_descriptions(images_list, image_frame_list,
                                                    prev_frame, current_frame)
            input_image_files.extend(image_dirs)
            prev_frame = current_frame
            bb_dict['content'] += image_content
        bb_dict['content'] += f"Q(frame {bubble.frame_number}): {bubble.get_full_words()}"
        input_conversation.append(bb_dict)
        input_conversation.append({"role": "<|Assistant|>", "content": ""})

        # print(f"[debug] conversation = {input_conversation}")
        # print(f'[debug] input_image_files = {input_image_files}')

        input_images = []
        for file in input_image_files:
            if self.use_base64:
                if file.startswith('data:'):
                    file = file.split(',', 1)[1]
                image_data = base64.b64decode(file)
                image = Image.open(BytesIO(image_data))
            else:
                image = Image.open(file)
            input_images.append(image)
        
        # print(input_image_files)
        # print(f"[debug] input_conversation = {input_conversation}")
        
        prepare_inputs = self.vl_chat_processor(
            conversations=input_conversation, images=input_images, force_batchify=True
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )
        
        # prompts = self.processor.apply_chat_template(input_conversation, add_generation_prompt=True)

        # inputs = self.processor(images=input_images, text=prompts, 
        #                    padding=True, return_tensors="pt").to(self.device)
        # generate_ids = self.model.generate(**inputs, max_new_tokens=1024)

        # if self.multi_image_flag:
        #     result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # else:
        #     result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        result = answer
        # print(f'[debug] A: {result}')
        
        if isinstance(result, list):
            result = result[0]
        
        return result