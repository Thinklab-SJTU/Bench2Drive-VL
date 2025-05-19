import os
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from io_utils import *

class B2DVLEvalDataset(Dataset):
    def __init__(self, image_dir, vqa_dir, original_vqa_dir, transform=None):
        """
        Initialize the dataset by storing paths to image and VQA data.

        :param image_dir: Path to the directory containing images
        :param vqa_dir: Path to the directory containing VQA JSON files
        :param transform: Transform to apply on the images (e.g., normalization)
        """
        self.image_dir = image_dir
        self.vqa_dir = vqa_dir
        self.original_vqa_dir = original_vqa_dir
        self.transform = transform

        self.camera_dirs = [
            'rgb_front',
            'rgb_front_left',
            'rgb_front_right',
            'rgb_back',
            'rgb_back_left',
            'rgb_back_right'
        ]

        self.camera_tags = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
        
        # Map each scenario to its corresponding VQA frame files
        self.scenario_vqa_map = defaultdict(list)
        self.scenario_start = {}
        self.scenario_end = {}
        # Iterate over scenario folders in vqa_dir
        for scenario_folder in os.listdir(vqa_dir):
            scenario_path = os.path.join(vqa_dir, scenario_folder)

            # Ignore non-directory or directory names without underscores
            if not os.path.isdir(scenario_path) or '_' not in scenario_folder:
                continue

            # Process valid scenario folders
            for json_file in os.listdir(scenario_path):
                if json_file.endswith('.json'):  # Ensure it's a JSON file
                    frame_string = json_file.split('.')[0]  # Get the part before '.json'
                    frame_number = int(frame_string)  # Convert the frame string to an integer

                    # Store the frame information
                    info_dict = {
                        'frame_number': frame_number,
                        'frame_string': frame_string,
                        'json_path': os.path.join(scenario_path, json_file),
                    }
                    self.scenario_vqa_map[scenario_folder].append(info_dict)

        self.scenarios = []
        # Sort frames for each scenario by frame number
        for scenario in self.scenario_vqa_map:
            self.scenarios.append(scenario)
            self.scenario_vqa_map[scenario].sort(key=lambda x: x['frame_number'])
        self.scenarios.sort()
        
        self.frame_info = []
        
        for scenario in self.scenarios:
            self.scenario_start[scenario] = len(self.frame_info)
            for info in self.scenario_vqa_map[scenario]:
                self.frame_info.append((scenario, info['frame_string']))
                self.scenario_end[scenario] = len(self.frame_info)

    def __len__(self):
        """
        Return the length of the dataset, which is the number of VQA frames available.

        :return: Number of VQA frames in the dataset
        """
        return len(self.frame_info)

    def __getitem__(self, idx):
        """
        Retrieve the image and corresponding VQA data for a specific index.

        :param idx: Index of the VQA frame to retrieve
        :return: Image and its corresponding VQA data and transform
        """
        # Get the scenario and frame number for this index
        scenario, frame_string = self.frame_info[idx]
        frame_number = int(frame_string)

        # Find the image corresponding to the frame number in the scenario directory
        scenario_path = os.path.join(self.image_dir, scenario)
        camera_dirs = self.camera_dirs
        camera_tags = self.camera_tags
        
        image_dict = {}
        image_dict['frame_number'] = frame_number
        # Loop through cameras (e.g., 'rgb_front', 'rgb_front_left') to find the matching image
        for camera_dir, camera_tag in zip(camera_dirs, camera_tags):
            image_path = os.path.join(scenario_path, 'camera', camera_dir, f'{frame_string}.jpg')
            # print(f'[debug] image_path = {image_path}')
            if os.path.exists(image_path):
                try:
                    image_dict[camera_tag] = image_path
                except:
                    image_dict[camera_tag] = None
                
            else:
                pass
                # we don't need the images when evaluating.
                # print_warning(f"Warning: {camera_tag} Image for frame {frame_number} in scenario {scenario} not found.")
                # raise FileNotFoundError(f"{camera_tag} Image for frame {frame_number} in scenario {scenario} not found.")
        
        vqa_info = self.load_vqa_data(scenario, frame_number, frame_string)
                
        return image_dict, vqa_info
        
    def load_vqa_data(self, scenario, frame_number, frame_string):
        """
        Dynamically load VQA data for a specific scenario and frame number.

        :param scenario: The name of the scenario folder
        :param frame_number: The frame number for which we need to load the VQA data
        :return: VQA data for the given frame number, including scenario, frame_number, 
                frame_string, file_path
        """

        vqa_file = os.path.join(self.vqa_dir, scenario, f"{frame_number:05d}.json")
        # print(f'[debug] vqa_file_path = {vqa_file}')
        vqa_content = load_json(vqa_file)
        anno_file = os.path.join(self.image_dir, scenario, "anno", f"{frame_number:05d}.json.gz")
        anno_content = load_json_gz(anno_file)
        original_vqa_dir = os.path.join(self.original_vqa_dir, scenario, f"{frame_number:05d}.json")

        vqa_info = {
            'scenario': scenario,
            'frame_number': frame_number,
            'frame_string': frame_string,
            'file_path': vqa_file,
            'content': vqa_content,
            'anno_path': os.path.join(self.image_dir, scenario, "anno"),
            'anno': anno_content,
            'original_vqa_dir': original_vqa_dir
        }

        return vqa_info

    def get_scenario_list(self):
        """Return a list of all scenario names."""
        return self.scenarios
    
    def get_start_and_end_of_scenario(self, scenario_name):
        """Return start and end index of a scenario. Data of this scenario is in [start, end)"""
        if scenario_name not in self.scenario_start:
            raise ValueError(f"Name '{scenario_name}' is not in recorded start points.")
        if scenario_name not in self.scenario_end:
            raise ValueError(f"Name '{scenario_name}' is not in recorded end points.")
        return self.scenario_start[scenario_name], self.scenario_end[scenario_name]
