import json
import os
import gzip
import tarfile
import shutil
from collections import deque

CACHE_DIR = "./.cache"
CACHE_SIZE = 8

ANSI = {
    'RED': "\033[1;31m",
    'GREEN': "\033[32m",
    'YELLOW': "\033[33m",
    'RESET': "\033[0m"
}

def write_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

def load_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print_error(f"Error loading JSON: {json_path}")
        return "Error loading JSON"
    
def load_json_gz(file_path):
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        print_error(f"Error loading JSON file: {file_path}")
        return "Error loading JSON"

def print_warning(message):
    print(f"{ANSI['YELLOW']}{message}{ANSI['RESET']}")

def print_error(message):
    print(f"{ANSI['RED']}{message}{ANSI['RESET']}")

def print_green(message):
    print(f"{ANSI['GREEN']}{message}{ANSI['RESET']}")

def print_bottom(message):
    print(f"\r\033[42m\033[37m {message} \033[0m\033[K", end="")

def print_debug(message):
    DEBUG = int(os.environ.get("DEBUG", 0))
    if DEBUG:
        print(message)

def read_file_lines(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]  # Remove trailing newline and whitespace
    except FileNotFoundError:
        print_error(f"Error when reading lines: The file '{file_path}' does not exist, returning empty list.")
        return []
    except Exception as e:
        print_error(f"Error when reading lines, returning empty list: {e}")
        return []

cache_queue = deque()

os.makedirs(CACHE_DIR, exist_ok=True)

def clean_cache():
    """
    Cleans up all folders in the cache directory.
    """
    for folder in os.listdir(CACHE_DIR):
        folder_path = os.path.join(CACHE_DIR, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)  # Delete folder
            print(f"Cleaned up cache folder: {folder_path}")
    
    # Clear the cache queue
    cache_queue.clear()


def extract_tar_gz(tar_gz_path):
    """
    Extracts a .tar.gz file and returns the path of the extracted folder.
    """
    folder_name = os.path.basename(tar_gz_path).replace('.tar.gz', '')
    extract_path = os.path.join(CACHE_DIR, folder_name)

    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(path=CACHE_DIR)
    # print_warning(f"Extracted: {tar_gz_path} to {CACHE_DIR}")
    
    return extract_path


def get_real_path(path):
    abs_path = os.path.abspath(path)
    path_parts = abs_path.strip(os.sep).split(os.sep)
    current_dir = os.sep

    for part in path_parts:
        current_dir = os.path.join(current_dir, part)
        if not os.path.exists(current_dir):
            tar_gz_path = current_dir + '.tar.gz'
            
            if os.path.exists(tar_gz_path):
                extract_path = None
                for cached_folder in cache_queue:
                    if os.path.basename(cached_folder) == part:
                        extract_path = cached_folder
                        break
                
                if not extract_path:
                    extract_path = extract_tar_gz(tar_gz_path)

                    if len(cache_queue) >= CACHE_SIZE:
                        old_folder = cache_queue.popleft()
                        shutil.rmtree(old_folder)
                        print_warning(f"Removed from cache: {old_folder}")

                    cache_queue.append(extract_path)
                
                current_dir = os.path.abspath(extract_path)
            else:
                raise FileNotFoundError(f"Path {part} or its .tar.gz file not found.")
    
    return current_dir

def filter_serializable(obj):
    """
    Recursively filters out non-serializable elements from a dictionary or list.
    
    Parameters:
    - obj: The input object (dict, list, or other types).
    
    Returns:
    - A filtered version of the input object with only JSON-serializable elements.
    """
    if isinstance(obj, dict):
        return {k: filter_serializable(v) for k, v in obj.items() if is_serializable(v)}
    elif isinstance(obj, list):
        return [filter_serializable(v) for v in obj if is_serializable(v)]
    else:
        return obj if is_serializable(obj) else None  # Return None for non-serializable elements

def is_serializable(value):
    """
    Checks if a value can be JSON serialized.
    
    Parameters:
    - value: Any Python object.
    
    Returns:
    - True if the value is serializable, False otherwise.
    """
    if isinstance(value, dict) or isinstance(value, list):
        return value 
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
    
def clean_json_string(input_string):
    if input_string.startswith('```json'):
        input_string = input_string[7:]
    
    if input_string.endswith('```'):
        input_string = input_string[:-3]
    
    trim_braces(input_string)
    
    return input_string

def trim_braces(s):
    start_index = 0
    while start_index < len(s) and s[start_index] != '{':
        start_index += 1

    end_index = len(s) - 1
    while end_index >= 0 and s[end_index] != '}':
        end_index -= 1

    if start_index <= end_index:
        return s[start_index:end_index + 1]
    else:
        return ""


TARGET_FOLDERS = {
    "rgb_back", "rgb_back_left", "rgb_back_right",
    "rgb_front", "rgb_front_left", "rgb_front_right",
    "lidar", "radar"
}

def delete_target_folders(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            if dirname in TARGET_FOLDERS:
                full_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(full_path)
                    print_debug(f"Deleted: {full_path}")
                except Exception as e:
                    print_debug(f"Failed to delete {full_path}: {e}")

def clear_target_folders(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            if dirname in TARGET_FOLDERS:
                full_path = os.path.join(dirpath, dirname)
                try:
                    for item in os.listdir(full_path):
                        item_path = os.path.join(full_path, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    print_debug(f"Cleared contents: {full_path}")
                except Exception as e:
                    print_debug(f"Failed to clear {full_path}: {e}")