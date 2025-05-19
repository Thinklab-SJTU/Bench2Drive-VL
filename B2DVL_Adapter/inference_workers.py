from inference_utils import Context, create_query, create_response
import json
import os
import torch
import requests
from io_utils import *
from threading import Thread
from qa_process import generate_condition, process_qa_by_qid, process_answer_by_qid, is_good_case
from image_process import process_bubble_image
from datetime import datetime
from models import *
from tqdm import tqdm

class InferTaskDistributor:
    def __init__(self, dataset, transform, model, model_path, num_workers, outdir, wp_code, configs):

        print('========= Task Distributing  =========')
        self.dataset = dataset  # This is the VQADataset instance
        self.transform = transform
        self.model_name = model
        self.model_path = model_path
        self.model = None
        self.num_workers = num_workers
        self.outpath = outdir
        self.wp_code = wp_code
        self.configs = configs

        if not torch.cuda.is_available():
            print_error("Error: CUDA is not available! Please check.")
            raise RuntimeError("CUDA is not available on this system. Please check your setup.")

        gpu_count = torch.cuda.device_count()
        if self.num_workers > gpu_count and self.model_name != "api":
            print_error(f"Error: We only detected {gpu_count} GPU(s), " +\
                        f"which is less than the worker number {self.num_workers} you set. " +\
                        f"We will use {gpu_count} worker(s) instead.")
            self.num_workers = gpu_count

        self.workers = self.create_workers()

    def create_workers(self):
        original_scenario_list = self.dataset.get_scenario_list()

        self.do_subset = self.configs.TASK_CONFIGS['INFER_SUBSET']
        self.do_checkpoint = self.configs.TASK_CONFIGS['USE_CHECKPOINT']
        self.subset_file = self.configs.TASK_CONFIGS['SUBSET_FILE']
        self.checkpoint_file = self.configs.TASK_CONFIGS['CHECKPOINT_FILE']
        included_scenarios = []
        excluded_scenarios = []
        if self.do_subset:
            included_scenarios = read_file_lines(self.subset_file)
        if self.do_checkpoint:
            excluded_scenarios = read_file_lines(self.checkpoint_file)

        scenario_list = []
        for scenario in original_scenario_list:
            if self.do_subset and scenario not in included_scenarios:
                continue
            if self.do_checkpoint and scenario in excluded_scenarios:
                continue
            scenario_list.append(scenario)

        worker_scenario_lists = [[] for _ in range(self.num_workers)]
        
        # Distribute scenarios to workers using modulo
        for idx, scenario in enumerate(scenario_list):
            worker_id = idx % self.num_workers
            worker_scenario_lists[worker_id].append(scenario)
        
        # Create workers with their assigned scenarios
        workers = []
        for i in range(self.num_workers):
            print(f"Distributed {worker_scenario_lists[i]} to Worker {i}.")
            worker = InferenceWorker(worker_id=i, scenario_list=worker_scenario_lists[i],
                                    dataset=self.dataset, wp_code=self.wp_code, transform=self.transform,
                                    model=self.model, model_name=self.model_name, model_path=self.model_path,
                                    outpath=self.outpath, configs=self.configs)
            workers.append(worker)
        
        return workers

    def distribute_tasks(self):
        """
        Distribute tasks to workers and start processing scenarios.
        """
        print(f'Using {self.num_workers} GPU(s)')
        print('============ Task Starts  ============')
        threads = []
        for rank in range(self.num_workers):
            t = Thread(target=self.workers[rank].work_loop)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

class InferenceWorker:
    def __init__(self, worker_id, scenario_list, dataset, wp_code, transform, 
                 model, model_name, model_path, outpath, configs, in_carla=False):
        
        self.in_carla = in_carla
        self.worker_id = worker_id
        self.scenario_list = scenario_list  # List of scenarios assigned to this worker
        self.dataset = dataset
        self.transform = transform
        self.wp_code = wp_code

        self.model_name = model_name
        self.model_path = model_path
        
        self.outpath = outpath
        self.logfile = os.path.join(outpath, f'worker_{worker_id}.log')
        
        self.device = None
        self.configs = configs

        if not self.in_carla:
            self.model = get_model_interface(self.model_name)
            # flags
            self.do_subset = self.configs.TASK_CONFIGS['INFER_SUBSET']
            self.do_checkpoint = self.configs.TASK_CONFIGS['USE_CHECKPOINT']
            self.subset_file = self.configs.TASK_CONFIGS['SUBSET_FILE']
            self.checkpoint_file = self.configs.TASK_CONFIGS['CHECKPOINT_FILE']
            self.entry_exits = load_json(self.configs.TASK_CONFIGS['ENTRY_EXIT_FILE'])

            self.append_question = self.configs.INFERENCE_BASICS['APPEND_QUESTION']

            self.appendix = []
            if self.append_question:
                self.appendix = load_json(self.configs.INFERENCE_BASICS['APPENDIX_FILE'])
        
        self.use_bev = False
        if self.in_carla:
            self.use_bev = self.configs.INFERENCE_BASICS['USE_BEV']
            self.port = self.configs.port
        self.use_base64 = self.configs.use_base64
        
        self.frame_rate = self.configs.TASK_CONFIGS['FRAME_PER_SEC']

        self.input_window = self.configs.INFERENCE_BASICS['INPUT_WINDOW']
        self.conversation_window = self.configs.INFERENCE_BASICS['CONVERSATION_WINDOW']
        self.no_history = self.configs.INFERENCE_BASICS['NO_HISTORY_MODE']
        self.all_camera = self.configs.INFERENCE_BASICS['USE_ALL_CAMERAS']

        self.chain = self.configs.CHAIN

        # context
        self.context = Context(conversation_window=self.conversation_window,
                               no_history=self.no_history)
        
        self.images_window = []

        # for history traj markings
        self.location_dict = {}
    
    def find_question_and_gt_by_id(self, id, vqa_data):
        """
        Given qid, return lists of question ans gt answers.
        """
        qlist = []
        alist = []
        alllist = []

        vqa_content = vqa_data['content']['QA']
        for categories in vqa_content.values():
            for qdict in categories:
                if 'qid' in qdict and qdict['qid'] == id:
                    qlist.append(qdict['Q'])
                    alist.append(qdict['A'])
                    alllist.append(qdict)
        if not self.in_carla and self.append_question:
            for qdict in self.appendix:
                if 'qid' in qdict and qdict['qid'] == id:
                    qlist.append(qdict['Q'])
                    alist.append(qdict['A'])
                    alllist.append(qdict)
        
        if len(qlist) == 0:
            print_error(f'Error: Question with qid {id} not found. Ignored.')

        return qlist, alist, alllist
    
    def init_model(self):
        # initialize model
        if self.in_carla:
            return # should not reach here!

        self.model.initialize(gpu_id=self.worker_id,
                              use_all_cameras=self.all_camera,
                              no_history=self.no_history,
                              input_window=self.input_window,
                              frame_rate=self.frame_rate,
                              model_path=self.model_path,
                              use_bev=self.use_bev,
                              in_carla=self.in_carla)
    
    def work_loop(self):
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        
        torch.cuda.set_device(self.worker_id)
        self.device = torch.device(f"cuda:{self.worker_id}")

        self.init_model()

        included_scenarios = []
        excluded_scenarios = []
        if self.do_subset:
            included_scenarios = read_file_lines(self.subset_file)
        if self.do_checkpoint:
            excluded_scenarios = read_file_lines(self.checkpoint_file)
        
        for scenario in self.scenario_list:
            if self.do_subset and scenario not in included_scenarios:
                continue
            if self.do_checkpoint and scenario in excluded_scenarios:
                continue
            
            entry = None
            exit = None

            if scenario in self.entry_exits:
                entry = self.entry_exits[scenario]['entry']
                exit = self.entry_exits[scenario]['exit']

            self.process_scenario(scenario, entry, exit)

            if self.do_checkpoint:
                with open(self.checkpoint_file, 'a') as file:
                    file.write(scenario + '\n')
            print_green(f"Worker {self.worker_id} finished processing {scenario}")

    def process_scenario(self, scenario, entry=None, exit=None):
        """
        Process the frames of a given scenario, maintaining context.
        
        :param scenario: The scenario directory to process
        """
        print(f"Worker {self.worker_id} processing scenario {scenario}")
        self.append_log(f"[debug] Worker {self.worker_id} processing scenario {scenario}, duration = [{entry}, {exit})")
        
        # Reset the context for the new scenario
        self.context.reset()
        
        # Get all frames for this scenario from the dataset
        start_index, end_index = self.dataset.get_start_and_end_of_scenario(scenario)
        self.images_window = []

        self.location_dict = {}
        for data_index in tqdm(range(start_index, end_index), 
                                desc=f"Worker {self.worker_id}, {scenario}"):
            images, vqa_data = self.dataset[data_index]
            frame_number = vqa_data['frame_number']
            anno_file = vqa_data['anno']
            self.update_history_location_info(anno_file=anno_file,
                                              frame_number=frame_number)
        
        for data_index in tqdm(range(start_index, end_index), 
                                desc=f"Worker {self.worker_id}, {scenario}"):
            images, vqa_data = self.dataset[data_index]
            frame_number = vqa_data['frame_number']

            if entry is None:
                entry = 0
            if entry is not None and frame_number < entry:
                continue
            if exit is not None and frame_number >= exit:
                continue
            
            # if not is_good_case(vqa_data['content']):
            #     print_warning(f"Skipped frame {frame_number} of {scenario} because it is not a good case.")
            #     continue
            res = self.ask_single_frame(scenario=scenario,
                                        frame_number=frame_number,
                                        vqa_data=vqa_data,
                                        images=images,
                                        entry=entry,
                                        start_index=start_index,
                                        end_index=end_index)
            json_content, json_file_name = res
        
            write_json(json_content, json_file_name)
    
    def update_history_location_info(self, anno_file, frame_number):
        frame = int(frame_number)
        anno_data = load_json_gz(anno_file)
        for actor in anno_data['bounding_boxes']:
            if actor['class'] == 'ego_vehicle':
                ego_vehicle = actor

        for actor in anno_data['bounding_boxes']:
            if abs(actor['location'][2] - ego_vehicle['location'][2]) > 40:
                # avoid the situation that the role actor is initialized undergound
                continue
            if 'location' in actor:
                if actor['id'] not in self.location_dict:
                    self.location_dict[actor['id']] = {}
                self.location_dict[actor['id']][frame] = actor['location']
            
    def interact_with_model(self, question_bubble, qid, frame):
        """
        Interact with the large language model to get the answer for the given VQA data.
        
        :param image: The input image
        :param vqa_data: The VQA data for the current frame
        :return: The output of the model (response)
        """
        
        curr_context = self.context.get_context_for_question(qid=qid,
                                                            prev=self.chain['PREV'],
                                                            inherit=self.chain['INHERIT'],
                                                            frame_number=frame)
        for bubble in curr_context:
            self.append_log(f"{bubble}")

        response = self.ask_model(question_bubble, curr_context)

        return response

    def ask_single_frame(self, scenario, frame_number, vqa_data, images, 
                         entry, start_index, end_index, 
                         extra_prompt="", special_state_str=None):
        json_content = {
            'scenario': scenario,
            'frame_number': frame_number,
            'key_object_infos': vqa_data['content']['key_object_infos'],
            'QA': {
                "Category": []
            },
            'extra_flags': vqa_data['content']['extra_flags'],
            'Cameras': {}
        }
        for key, value in images.items():
            if 'frame' not in key:
                json_content['Cameras'][key] = value
                if self.use_base64:
                    images[key] = local_image_to_base64(value)
        
        json_file_name = os.path.join(self.outpath, scenario, f"{frame_number:05d}.json")
        
        self.images_window.append(images)
        if len(self.images_window) > self.input_window:
            self.images_window = self.images_window[1:]
        self.append_log(f"{self.images_window[:-1]}")

        self.context.fifo()
        
        qorder = self.chain['ORDER']
        # Interact with VLM
        qrank = 0
        qlen = len(qorder)
        for qid in qorder:
            qrank += 1
            qlist, alist, alllist = self.find_question_and_gt_by_id(qid, vqa_data)
            # print(f'[debug] qlist = {qlist}, alist = {alist}')
            for question, gt, original_dict in zip(qlist, alist, alllist):
                if qid not in self.chain['USE_GT']:
                    self.append_log("=======================================")
                    question, gt = process_qa_by_qid(question, gt, qid)
                    extra_condition = generate_condition(vqa_data['anno'], qid, special_state_str)
                    if extra_prompt is not None and extra_prompt != "":
                        extra_condition = extra_prompt + " " + extra_condition
                    question_bubble = create_query(words=question, images=[images],
                                                frame_number=frame_number, scenario=scenario,
                                                qid=qid, gt=gt, transform=self.transform,
                                                extra_words=extra_condition, extra_images=self.images_window[:-1])
                    if not self.in_carla:
                        question_bubble = process_bubble_image(question_bubble, self.worker_id, scenario, self.location_dict, vqa_data['anno'])
                    original_dict['actual_Q'] = question_bubble.get_full_words()
                    original_dict['actual_gt'] = question_bubble.gt
                    self.append_log(f"Q: qid = {qid}, {question_bubble.get_full_words()}")

                    response = self.interact_with_model(question_bubble, qid, frame_number)
                    response = process_answer_by_qid(response, qid, self.wp_code)
                    answer_bubble = create_response(words=response,
                                                    frame_number=frame_number, scenario=scenario,
                                                    qid=qid, gt=gt)
                    self.append_log(f"A: qid = {qid}, {response}")
                    original_dict['VLM_name'] = self.model_name
                    original_dict['VLM_answer'] = response
                    original_dict['Q_timestamp'] = question_bubble.timestamp
                    original_dict['Q_time_readable'] = datetime.fromtimestamp(question_bubble.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
                    original_dict['A_timestamp'] = answer_bubble.timestamp
                    original_dict['A_time_readable'] = datetime.fromtimestamp(answer_bubble.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # log_line = f"[debug] [Worker {self.worker_id}] Got question answer \n"
                    # log_line += f"       answer_bubble: {answer_bubble}"
                    # self.append_log(log_line)

                    self.context.update(question_bubble)
                    self.context.update(answer_bubble)

                    json_content['QA']['Category'].append(original_dict)
                else:
                    self.append_log("=======================================")
                    question, gt = process_qa_by_qid(question, gt, qid)
                    extra_condition = generate_condition(vqa_data['anno'], qid, special_state_str)
                    if extra_prompt is not None and extra_prompt != "":
                        extra_condition = extra_prompt + " " + extra_condition
                    question_bubble = create_query(words=question, images=[images],
                                                frame_number=frame_number, scenario=scenario,
                                                qid=qid, gt=gt, transform=self.dataset.transform,
                                                extra_words=extra_condition, extra_images=self.images_window[:-1])
                    if not self.in_carla:
                        question_bubble = process_bubble_image(question_bubble, self.worker_id, scenario, self.location_dict, vqa_data['anno'])
                    original_dict['actual_Q'] = question_bubble.get_full_words()
                    original_dict['actual_gt'] = question_bubble.gt
                    self.append_log(f"Q: qid = {qid}, {question_bubble.get_full_words()}")

                    response = question_bubble.gt
                    answer_bubble = create_response(words=response,
                                                    frame_number=frame_number, scenario=scenario,
                                                    qid=qid, gt=gt)
                    self.append_log(f"A: qid = {qid}, {response}")
                    original_dict['VLM_name'] = self.model_name
                    original_dict['VLM_answer'] = response
                    original_dict['Q_timestamp'] = question_bubble.timestamp
                    original_dict['Q_time_readable'] = datetime.fromtimestamp(question_bubble.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
                    original_dict['A_timestamp'] = answer_bubble.timestamp
                    original_dict['A_time_readable'] = datetime.fromtimestamp(answer_bubble.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # log_line = f"[debug] [Worker {self.worker_id}] Got question answer \n"
                    # log_line += f"       answer_bubble: {answer_bubble}"
                    # self.append_log(log_line)

                    self.context.update(question_bubble)
                    self.context.update(answer_bubble)

                    json_content['QA']['Category'].append(original_dict)
            
            print_bottom(f"\r[log] Worker {self.worker_id} answered question {qid} ({qrank} of {qlen}) " +\
                         f"at frame {frame_number} ({frame_number - entry + 1} of {end_index - start_index}) of {scenario}.")
        
        return json_content, json_file_name
    
    def ask_model(self, question_bubble, curr_context):
        if not self.in_carla:
            response = self.model.interact(question_bubble, curr_context)
        else:
            payload = {
                "bubble": question_bubble.to_dict(),
                "conversation": [b.to_dict() for b in curr_context]
            }
            try:
                print_debug(f"[debug] try asking http://localhost:{self.port}/interact")
                response = requests.post(f"http://localhost:{self.port}/interact", json=payload)
                print_debug(f"[debug] ask_response, response: {response}")
                response = response.json()['response']
                print_debug(f"[debug] final response = {response}")
            except Exception as e:
                print_error(f"Inference worker error: in ask_model, {e}")
                response = f"Error occured in inference worker: {e}"
        return response
    
    def append_log(self, line):
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
        with open(self.logfile, 'a') as file:
            file.write(line + '\n')