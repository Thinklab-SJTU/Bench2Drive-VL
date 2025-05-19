from inference_utils import Context, create_query, create_response
import json
import os
import torch
from io_utils import *
from threading import Thread
from qa_process import generate_condition, process_qa_by_qid
from image_process import process_bubble_image
from datetime import datetime
from models import *
from tqdm import tqdm

class TaskDistributor:
    def __init__(self, dataset, transform, model, model_path, num_workers, outdir, configs):

        print('========= Task Distributing  =========')
        self.dataset = dataset  # This is the VQADataset instance
        self.transform = transform
        self.model_name = model
        self.model_path = model_path
        self.model = None
        self.num_workers = num_workers
        self.outpath = outdir
        self.configs = configs

        if not torch.cuda.is_available():
            print_error("Error: CUDA is not available! Please check.")
            raise RuntimeError("CUDA is not available on this system. Please check your setup.")

        gpu_count = torch.cuda.device_count()
        if self.num_workers > gpu_count:
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
                                    dataset=self.dataset, transform=self.transform,
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
    def __init__(self, worker_id, scenario_list, dataset, transform, 
                 model, model_name, model_path, outpath, configs, in_carla=False):
        
        if not in_carla:
            self.worker_id = worker_id
            self.scenario_list = scenario_list  # List of scenarios assigned to this worker
            self.dataset = dataset
            self.transform = transform
            
            self.model_name = model_name
            self.model_path = model_path
            self.model = get_model_interface(self.model_name)

            self.outpath = outpath
            self.logfile = os.path.join(outpath, f'worker_{worker_id}.log')
            
            self.device = None
            self.configs = configs

            # flags
            self.do_subset = self.configs.TASK_CONFIGS['INFER_SUBSET']
            self.do_checkpoint = self.configs.TASK_CONFIGS['USE_CHECKPOINT']
            self.subset_file = self.configs.TASK_CONFIGS['SUBSET_FILE']
            self.checkpoint_file = self.configs.TASK_CONFIGS['CHECKPOINT_FILE']
            self.entry_exits = load_json(self.configs.TASK_CONFIGS['ENTRY_EXIT_FILE'])
            self.frame_rate = self.configs.TASK_CONFIGS['FRAME_PER_SEC']

            self.input_window = self.configs.INFERENCE_BASICS['INPUT_WINDOW']
            self.conversation_window = self.configs.INFERENCE_BASICS['CONVERSATION_WINDOW']
            self.no_history = self.configs.INFERENCE_BASICS['NO_HISTORY_MODE']
            self.append_question = self.configs.INFERENCE_BASICS['APPEND_QUESTION']
            self.all_camera = self.configs.INFERENCE_BASICS['USE_ALL_CAMERAS']

            self.chain = self.configs.CHAIN

            # context
            self.context = Context(conversation_window=self.conversation_window,
                                no_history=self.no_history)

            self.appendix = []
            if self.append_question:
                self.appendix = load_json(self.configs.INFERENCE_BASICS['APPENDIX_FILE'])
    
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
        if self.append_question:
            for qdict in self.appendix:
                if 'qid' in qdict and qdict['qid'] == id:
                    qlist.append(qdict['Q'])
                    alist.append(qdict['A'])
                    alllist.append(qdict)
        
        if len(qlist) == 0:
            print_error(f'Error: Question with qid {id} not found. Ignored.')

        return qlist, alist, alllist
    
    def work_loop(self):
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        
        torch.cuda.set_device(self.worker_id)
        self.device = torch.device(f"cuda:{self.worker_id}")

        # initialize model
        self.model.initialize(gpu_id=self.worker_id,
                              use_all_cameras=self.all_camera,
                              no_history=self.no_history,
                              input_window=self.input_window,
                              frame_rate=self.frame_rate,
                              model_path=self.model_path)

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
        images_window = []
        
        for data_index in tqdm(range(start_index, end_index), 
                                desc=f"Worker {self.worker_id}, {scenario}, frame range = [{start_index}, {end_index})"):
            images, vqa_data = self.dataset[data_index]
            frame_number = vqa_data['frame_number']
            if entry is not None and frame_number < entry:
                continue
            if exit is not None and frame_number >= exit:
                continue
            
            json_content = {
                'scenario': scenario,
                'frame_number': frame_number,
                'key_object_infos': vqa_data['content']['key_object_infos'],
                'QA': [],
                'Cameras': {}
            }
            for key, value in images.items():
                if 'frame' not in key:
                    json_content['Cameras'][key] = value
            
            json_file_name = os.path.join(self.outpath, scenario, f"{frame_number:05d}.json")
            
            images_window.append(images)
            if len(images_window) > self.input_window:
                images_window = images_window[1:]
            self.append_log(f"{images_window[:-1]}")

            self.context.fifo()
            
            # Interact with VLM
            qrank = 0
            qlen = len(self.chain)
            for qid in self.chain:
                qrank += 1
                qlist, alist, alllist = self.find_question_and_gt_by_id(qid, vqa_data)
                # print(f'[debug] qlist = {qlist}, alist = {alist}')
                for question, gt, original_dict in zip(qlist, alist, alllist):
                    question, gt = process_qa_by_qid(question, gt, qid)
                    extra_condition = generate_condition(vqa_data['anno'], qid)
                    question_bubble = create_query(words=question, images=[images],
                                                   frame_number=frame_number, scenario=scenario,
                                                   qid=qid, gt=gt, transform=self.dataset.transform,
                                                   extra_words=extra_condition, extra_images=images_window[:-1])
                    question_bubble = process_bubble_image(question_bubble, self.worker_id)
                    original_dict['actual_Q'] = question_bubble.get_full_words()
                    original_dict['actual_gt'] = question_bubble.gt
                    self.append_log(f"Q: qid = {qid}, {question_bubble.get_full_words()}")

                    response = self.interact_with_model(question_bubble)
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

                    print_bottom(f"\r[log] Worker {self.worker_id} answered question {qid} ({qrank} of {qlen}) " +\
                    f"at frame {frame_number} ({frame_number - entry + 1} of {end_index - start_index}) of {scenario}.")

                    self.append_log("============================")
                    self.append_log(f"{self.context}")

                    json_content['QA'].append(original_dict)
            
            write_json(json_content, json_file_name)
    
    def interact_with_model(self, question_bubble):
        """
        Interact with the large language model to get the answer for the given VQA data.
        
        :param image: The input image
        :param vqa_data: The VQA data for the current frame
        :return: The output of the model (response)
        """
        
        response = self.ask_model(question_bubble, self.context)

        return response
    
    def ask_model(self, question_bubble, context):
        response = self.model.interact(question_bubble, context)
        return response

    def append_log(self, line):
        with open(self.logfile, 'a') as file:
            file.write(line + '\n')