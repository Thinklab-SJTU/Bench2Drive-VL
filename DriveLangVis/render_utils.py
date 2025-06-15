import json
import html
from datetime import datetime
import gzip
import copy
import os
import re

def load_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return "Error loading JSON"

def save_json(json_path, data):
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        return "Error saving JSON"

def load_gzip_json(gzip_path):
    try:
        with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return "Error loading GZ JSON"
    
def load_status(STATUS_FILE):
    with open(STATUS_FILE, "r") as f:
        return json.load(f)

def save_status(status_data, STATUS_FILE):
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=4)

def get_first_level_dirs(path):
    try:
        return [
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]
    except Exception as e:
        print(f"Error accessing path {path}: {e}")
        return []


def get_key_style(key_path, path_dict, VALUE_STATUS_FILE):
    status = "raw" # default ststus is raw
    json_name = key_path.split('/')[0]
    file_path = path_dict[json_name]
    try:
        with open(VALUE_STATUS_FILE, "r") as f:
            status_data = json.load(f)
            if file_path in status_data:
                status = status_data[file_path].get(key_path, "raw")
    except FileNotFoundError:
        pass
    
    style = ""
    if key_path.split('/')[-1] == "Q":
        style = "background-color: black; color: white; font-weight: bold;"
    if status == "controversy":
        style = "background-color: orange; color: white;"
    if key_path.split('/')[-1] == "qid":
        style += "font-weight: bold;"
    return style

def json_to_html(data, path_dict, VALUE_STATUS_FILE, current_view, vqa_filter, show_key_object_info, key_path=""):
    new_data = copy.deepcopy(data)

    if not show_key_object_info:
        if key_path is not None and key_path == "qa_json":
            if 'key_object_infos' in new_data:
                del new_data['key_object_infos']
    
    if vqa_filter and len(vqa_filter) > 0:
        if key_path is not None and key_path == "qa_json":
            for key, qa_list in new_data["QA"].items():
                if qa_list is not None:
                    new_data["QA"][key] = [qa for qa in qa_list if str(qa.get("qid")) in vqa_filter]

    if current_view == "VQA View":
        if key_path is not None and key_path != "qa_json":
            return "<p>collapsed</p>"
        if "key_object_infos" in new_data and new_data["key_object_infos"] is not None:
            allowed_keys = ["id", "Category", "Status", "Visual_description", "Detailed_description", "dict_id"]
            new_data["key_object_infos"] = {
                key: {k: v for k, v in value.items() if k in allowed_keys}
                for key, value in new_data["key_object_infos"].items()
            }
        if "QA" in new_data:
            allowed_qa_keys = ["Q", "A", "qid", "object_tags", "dict_id", "VLM_name", "VLM_answer"]
            for key, qa_list in new_data["QA"].items():
                if isinstance(qa_list, list):
                    new_data["QA"][key] = [
                        {k: qa[k] for k in allowed_qa_keys if k in qa}
                        for qa in qa_list if isinstance(qa, dict)
                    ]
                
    return json_to_html_with_keys(new_data, path_dict, VALUE_STATUS_FILE, current_view, key_path)
    
def json_to_html_with_keys(data, path_dict, VALUE_STATUS_FILE, current_view, key_path="", qid=-1):
    
    def highlight_object_tags(value):
        pattern = r"\(&lt;.*?&gt;\)"
        # use <span> to generate formated html
        return re.sub(
            pattern,
            lambda match: f'<span style="color: #007BFF;">{match.group(0)}</span>',
            value
        )
    
    if isinstance(data, dict):
        html_output = '<table border="1">'
        for key, value in data.items():
            if key == 'dict_id':
                continue
            full_key = f"{key_path}/{key}" if key_path else key
            key_style = get_key_style(full_key, path_dict, VALUE_STATUS_FILE)
            if isinstance(value, dict) and value.get('qid', -1) > 0:
                qid = value.get('qid')
            html_output += f'<tr><td>{highlight_object_tags(html.escape(str(key)))}</td><td style="{key_style}">{json_to_html_with_keys(value, path_dict, VALUE_STATUS_FILE, current_view, full_key, qid)}'
            if not isinstance(value, (dict, list)) and ((current_view == 'Full View') or (key in ['Q', 'A', 'object_tags']) or ('description' in key)):
                # only leave nodes show edit button
                html_output += f'<button onclick="editValue(\'{full_key}\', \'{html.escape(str(value))}\')">Edit</button></td>'
            else:
                html_output += f'</td>'
            html_output += '</tr>'
        html_output += '</table>'
        return html_output
    elif isinstance(data, list):
        html_output = '<ul>'
        child_type = None
        for i, value in enumerate(data):
            k = i
            if isinstance(value, dict):
                k = value['dict_id']
            full_key = f"{key_path}/{k}" if key_path else str(k)
            key_style = get_key_style(full_key, path_dict, VALUE_STATUS_FILE)
            html_output += f'<li style="{key_style}">'
            html_output += json_to_html_with_keys(value, path_dict, VALUE_STATUS_FILE, current_view, full_key, qid)
            if not isinstance(value, (dict, list)):
                child_type = type(value)
                # only leaf nodes show edit button
                html_output += f' <button onclick="editValue(\'{full_key}\', \'{html.escape(str(value))}\')">Edit</button>'
                html_output += f' <button onclick="deleteValue(\'{full_key}\')">Delete</button>'
            html_output += '</li>'
        html_output += '</ul>'
        if child_type:
            html_output += f'<button onclick="addValue(\'{key_path}\', \'{child_type.__name__}\')">Append</button>'
        return html_output
    else:
        escaped_value = highlight_object_tags(html.escape(str(data)))
        html_output = f'<span data-key="{key_path}" qid="{qid}">{escaped_value}</span>'
        return html_output


def convert_value_type(old_value, new_value):
    """
    Change new_value's datatype according to old_value.
    if it fails, raise ValueError.
    """
    if isinstance(old_value, bool):
        if new_value.lower() in ["true", "false"]:
            return new_value.lower() == "true"
        raise ValueError("Cannot convert to bool")
    elif isinstance(old_value, int):
        return int(new_value)
    elif isinstance(old_value, float):
        return float(new_value)
    elif isinstance(old_value, list):
        return json.loads(new_value)
    elif isinstance(old_value, dict):
        return json.loads(new_value)
    elif isinstance(old_value, str):
        return str(new_value)
    else:
        raise ValueError(f"Unsupported type: {type(old_value)}")

def get_nested_value(d, path):
    for key in path:
        final_key = key
        if isinstance(d, list):
            key = int(key)  # index of list
            final_key = key
            if len(d) > 0 and isinstance(d[0], dict):
                for i in range(len(d)):
                    if d[i]['dict_id'] == key:
                        final_key = i
        d = d[final_key]
    return d

def log_edit(action, nodepath, filepath, logpath, old_value=None, new_value=None):
    """
    Record edit histories in logs.
    
    Args:
        action (str): action type ("add", "delete", "modify")
        path (str): path of edit log
        new_value (Any): new value given
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "path": nodepath,
        "file": filepath,
        "old_value": old_value,
        "new_value": new_value
    }
    
    log_line = f"{log_entry}\n"
    
    try:
        with open(logpath, "a", encoding="utf-8") as log_file:
            log_file.write(log_line)
    except Exception as e:
        print(f"Failed to write to log: {e}")

def get_status_count(qa_root, folder, file_status):
    qa_dir = os.path.join(qa_root, folder)
    json_files = [f for f in os.listdir(qa_dir) if f.endswith('.json')]
    all_count = len(json_files)

    status_data = load_status(file_status)
    dir_str = str(qa_dir)
    count_dict = {
        'raw': 0,
        'controversy': 0,
        'verified': 0,
    }
    for key, value in status_data.items():
        if dir_str in key:
            count_dict[value] += 1

    count_dict['raw'] = all_count - count_dict['controversy'] - count_dict['verified']
    return count_dict['raw'], count_dict['controversy'], count_dict['verified']

def append_list_id(obj):
    '''
    append dict_id for dicts under lists
    '''
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = append_list_id(value)
        return result if result else None
    elif isinstance(obj, list):
        result = []
        for i in range(len(obj)):
            if isinstance(obj[i], dict):
                obj[i]['dict_id'] = i
            appended_item = append_list_id(obj[i])
            if appended_item is not None:
                result.append(appended_item)
        return result if result else None
    else:
        return obj

def delete_list_id(obj):
    '''
    delete dict_id for dicts under lists
    '''
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key != 'dict_id':
                result[key] = delete_list_id(value)
        return result if result else None
    elif isinstance(obj, list):
        result = []
        for i in range(len(obj)):
            appended_item = delete_list_id(obj[i])
            if appended_item is not None:
                result.append(appended_item)
        return result if result else None
    else:
        return obj