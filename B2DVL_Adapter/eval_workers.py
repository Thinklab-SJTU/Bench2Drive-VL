import json
import os
import torch
from io_utils import *
from threading import Thread
from datetime import datetime
from tqdm import tqdm
from evaluator import evaluate_question
from api_interface import *
from mytoken import *
from qa_process import is_frame_trivial, is_good_case
import numpy as np

class EvalTaskDistributor:
    def __init__(self, dataset, num_workers, outdir, configs, num_sample):

        print('========= Task Distributing  =========')
        self.dataset = dataset  # This is the VQADataset instance
        self.model = None
        self.num_workers = num_workers
        self.outpath = outdir
        self.configs = configs
        self.num_sample = num_sample
        self.workers = self.create_workers()

    def create_workers(self):
        original_scenario_list = self.dataset.get_scenario_list()

        self.do_subset = self.configs.CONFIGS['EVAL_SUBSET']
        self.do_checkpoint = self.configs.CONFIGS['USE_CHECKPOINT']
        self.subset_file = self.configs.CONFIGS['SUBSET_FILE']
        self.checkpoint_file = self.configs.CONFIGS['CHECKPOINT_FILE']
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
            worker = EvalWorker(worker_id=i, scenario_list=worker_scenario_lists[i],
                                dataset=self.dataset,
                                outpath=self.outpath, configs=self.configs,
                                num_sample=self.num_sample)
            workers.append(worker)
        
        return workers

    def distribute_tasks(self):
        """
        Distribute tasks to workers and start processing scenarios.
        """
        print('============ Task Starts  ============')
        threads = []
        for rank in range(self.num_workers):
            t = Thread(target=self.workers[rank].work_loop)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

class EvalWorker:
    def __init__(self, worker_id, scenario_list, dataset, outpath, configs, num_sample):
        self.worker_id = worker_id
        self.scenario_list = scenario_list  # List of scenarios assigned to this worker
        self.dataset = dataset

        self.outpath = outpath
        self.logfile = os.path.join(outpath, f'worker_{worker_id}.log')
        
        self.device = None
        self.configs = configs
        self.num_sample = num_sample

        # flags
        self.do_subset = self.configs.CONFIGS['EVAL_SUBSET']
        self.do_checkpoint = self.configs.CONFIGS['USE_CHECKPOINT']
        self.subset_file = self.configs.CONFIGS['SUBSET_FILE']
        self.checkpoint_file = self.configs.CONFIGS['CHECKPOINT_FILE']
        self.frame_rate = self.configs.CONFIGS['FRAME_PER_SEC']
        self.look_future = self.configs.CONFIGS['LOOK_FUTURE']

        self.llm_client = DeepseekInterface(DEEPSEEK_TOKEN, DEEPSEEK_URL)
        self.future_window = 20
    
    def find_question_and_gt_by_id(self, id, vqa_data):
        """
        Given qid, return lists of question ans gt answers.
        """
        qlist = []
        alist = []
        alllist = []

        vqa_content = vqa_data['content']['QA']
        for qdict in vqa_content:
            if 'qid' in qdict and qdict['qid'] == id:
                qlist.append(qdict['Q'])
                alist.append(qdict['A'])
                alllist.append(qdict)
        # if self.append_question:
        #     for qdict in self.appendix:
        #         if 'qid' in qdict and qdict['qid'] == id:
        #             qlist.append(qdict['Q'])
        #             alist.append(qdict['A'])
        #             alllist.append(qdict)
        
        if len(qlist) == 0:
            print_error(f'Error: Question with qid {id} not found. Ignored.')

        return qlist, alist, alllist
    
    def work_loop(self):
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

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
        
        
        # Get all frames for this scenario from the dataset
        start_index, end_index = self.dataset.get_start_and_end_of_scenario(scenario)

        valid_data_indices = []
        for data_index in range(start_index, end_index):
            _, vqa_data = self.dataset[data_index]
            frame_number = vqa_data['frame_number']
            if entry is not None and frame_number < entry:
                continue
            if exit is not None and frame_number >= exit:
                continue
            valid_data_indices.append((frame_number, data_index))  # include frame_number for sorting

        # Sort valid indices by frame_number
        valid_data_indices.sort(key=lambda x: x[0])
        frame_sorted_indices = [idx for _, idx in valid_data_indices]

        # Apply uniform sampling if needed
        if self.num_sample is not None and self.num_sample >= 0:
            sample_count = min(self.num_sample, len(frame_sorted_indices))
            sampled_indices_set = sorted(np.linspace(0, len(frame_sorted_indices) - 1, sample_count, dtype=int))
            frame_sorted_indices = [frame_sorted_indices[i] for i in sampled_indices_set]

        # Debug: print selected frame_numbers
        selected_frames = []
        for data_index in frame_sorted_indices:
            _, vqa_data = self.dataset[data_index]
            selected_frames.append(vqa_data['frame_number'])

        print_green(f"[Worker {self.worker_id}] Selected {len(selected_frames)} frame_numbers: {selected_frames}")

        for data_index in tqdm(frame_sorted_indices, 
                       desc=f"Worker {self.worker_id}, {scenario}, frame range = [{start_index}, {end_index})"):
            images, vqa_data = self.dataset[data_index]
            frame_number = vqa_data['frame_number']

            # if not is_good_case(load_json(vqa_data['original_vqa_dir'])):
            #     print_warning(f"Skipped frame {frame_number} of {scenario} because it is not a good case.")
            #     continue
            
            json_content = {
                'scenario': scenario,
                'frame_number': frame_number,
                'QA': {},
                'Cameras': {}
            }
            for key, value in images.items():
                if 'frame' not in key:
                    json_content['Cameras'][key] = value
            
            json_file_name = os.path.join(self.outpath, scenario, f"{frame_number:05d}.json")
            
            for category in vqa_data['content']['QA']:
                json_content['QA'][category] = []
                for qdict in vqa_data['content']['QA'][category]:
                    qid = qdict['qid']
                    # if qid not in [42]: # just for debug
                    #     continue
                    # get future gts of the same question
                    future_data = None
                    if self.look_future:
                        end_point = min(end_index, data_index + self.future_window)
                        future_data = []
                        for dindex in range(data_index, end_point):
                            dimages, dvqa_data =  self.dataset[dindex]
                            dframe_number = vqa_data['frame_number']
                            _, _, dalllist = self.find_question_and_gt_by_id(qid, dvqa_data)
                            data_dict = {
                                'frame_number': dframe_number,
                                'qdata': dalllist
                            }
                            future_data.append(data_dict)
                    
                    qdict['score'], qdict['eval_reason'] = evaluate_question(llm_client=self.llm_client, question_dict=qdict, 
                                                                            anno_path=vqa_data['anno_path'],
                                                                            frame_number=frame_number, future_data=future_data, 
                                                                            frame_rate=self.frame_rate, key_object_infos=vqa_data['content']['key_object_infos'])

                    # print_bottom(f"\r[log] Worker {self.worker_id} evaluated question {qid} ({qrank} of {qlen}) " +\
                    # f"at frame {frame_number} ({frame_number - entry + 1} of {end_index - start_index}) of {scenario}.")

                    # self.append_log("============================")

                    json_content['QA'][category].append(qdict)
            
            write_json(json_content, json_file_name)

    def append_log(self, line):
        log_dir = os.path.dirname(self.logfile)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        with open(self.logfile, 'a') as file:
            file.write(line + '\n')