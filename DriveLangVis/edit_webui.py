from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired
import os
import json
import gzip
import re
from render_utils import *
import threading
import time
import platform
import yaml
system = platform.system()

app = Flask(__name__)
app.secret_key = "defaultkey"

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

b2d_data_path = config["b2d_data_path"]
appendix_path = config["appendix_path"]
qa_path = config["qa_path"]
anno_image_path = config["anno_image_path"]
b2d_rel_path = config["b2d_rel_path"]
b2d_subfix = config["b2d_subfix"]
anno_rel_path = config["anno_rel_path"]
anno_subfix = config["anno_subfix"]

EDIT_LOG_PATH = './edit_log'
STATUS_FILE = "file_status.json"
VALUE_STATUS_FILE = "value_status.json"
COMMON_OPTION_FILE = "common_options.json"
ENTRY_EXIT_FILE = "entry_exit.json"
USER_DATA_FILE = "user_data.json"
USER_DATA_SAVE_INTERVAL = 30
DEFAULT_USER = "default user"

default_user_data = {
    "current_view": "VQA View",
    "vqa_filter": [],
    "show_key_object_info": True,
    "rel_path": "",
    "selected_number": None,
    "json_content": {
        "anno_json": "not exist",
        "appendix_json": "not exist",
        "qa_json": "not exist",
    },
    "content": {
        "rgb_front": "not exist",
        "rgb_top_down": "not exist",
        "anno_json": "not exist",
        "appendix_json": "not exist",
        "qa_json": "not exist",
    },
    "filter_dict": {
        "anno_json": [],
        "appendix_json": [],
        "qa_json": [],
    },
    "json_files": [],
    "json_numbers": [],
}

user_data = load_json(USER_DATA_FILE)
if not isinstance(USER_DATA_FILE, dict):
    user_data = {}
user_data[DEFAULT_USER] = default_user_data

if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "w") as f:
        json.dump({}, f)
if not os.path.exists(VALUE_STATUS_FILE):
    with open(VALUE_STATUS_FILE, "w") as f:
        json.dump({}, f)
if not os.path.exists(COMMON_OPTION_FILE):
    with open(COMMON_OPTION_FILE, "w") as f:
        json.dump({}, f)
if not os.path.exists(ENTRY_EXIT_FILE):
    with open(ENTRY_EXIT_FILE, "w") as f:
        json.dump({}, f)
if not os.path.exists(EDIT_LOG_PATH):
    with open(EDIT_LOG_PATH, "w") as f:
        pass # empty file

@app.route(f'/media/<path:filename>')
def media_files(filename):
    if system == "Linux":
        filename = os.path.join("/", filename)
    if not os.path.isabs(filename):
        return "Not an absolute path", 400

    if not os.path.isfile(filename):
        return "File not found", 404

    return send_file(filename)

class RelPathForm(FlaskForm):
    rel_path = StringField("Relative Path", validators=[DataRequired()])
    json_number = SelectField("Choose JSON Number", choices=[], coerce=str)
    submit = SubmitField("Load")

def get_path_dict(username):
    global user_data
    print(f"[debug] get_path_dict = {username}")
    selected_number = user_data[username]['selected_number']
    rel_path = user_data[username]['rel_path']
    rgb_front_path = os.path.join(anno_image_path, rel_path, anno_rel_path, f"{selected_number}.{anno_subfix}")
    rgb_top_down_path = os.path.join(b2d_data_path, rel_path, b2d_rel_path, f"{selected_number}.{b2d_subfix}")
    anno_json_path = os.path.join(b2d_data_path, rel_path, "anno", f"{selected_number}.json.gz")
    appendix_json_path = os.path.join(appendix_path, rel_path, f"{selected_number}.json")
    qa_json_path = os.path.join(qa_path, rel_path, f"{selected_number}.json")
    return {
        'rgb_front': rgb_front_path,
        'rgb_top_down': rgb_top_down_path,
        'anno_json': anno_json_path,
        'appendix_json': appendix_json_path,
        'qa_json': qa_json_path
    }

def update_content(username):
    global user_data
    path_dict = get_path_dict(username)
    print(f"[debug] update_content username = {username}")
    rgb_front_path = path_dict['rgb_front']
    rgb_top_down_path = path_dict['rgb_top_down']
    anno_json_path = path_dict['anno_json']
    appendix_json_path = path_dict['appendix_json']
    qa_json_path = path_dict['qa_json']

    user_data[username]['content']["rgb_front"] = rgb_front_path if os.path.exists(rgb_front_path) else "not exist"
    user_data[username]['content']["rgb_top_down"] = rgb_top_down_path if os.path.exists(rgb_top_down_path) else "not exist"
    user_data[username]['json_content']["anno_json"] = (
        append_list_id(load_gzip_json(anno_json_path)) if os.path.exists(anno_json_path) else "not exist"
    )
    user_data[username]['json_content']["appendix_json"] = append_list_id(load_json(appendix_json_path)) if os.path.exists(appendix_json_path) else "not exist"
    user_data[username]['json_content']["qa_json"] = append_list_id(load_json(qa_json_path)) if os.path.exists(qa_json_path) else "not exist"

    for key in ["anno_json", "appendix_json", "qa_json"]:
        if isinstance(user_data[username]['json_content'][key], (dict, list)):
            user_data[username]['content'][key] = filter_json_tree(user_data[username]['json_content'][key], 
                                                                   user_data[username]['filter_dict'][key])
            user_data[username]['content'][key] = json_to_html(user_data[username]['content'][key], 
                                                               path_dict, VALUE_STATUS_FILE, 
                                                               user_data[username]['current_view'], 
                                                               user_data[username]['vqa_filter'], 
                                                               user_data[username]['show_key_object_info'], 
                                                               key)
    
    save_json(USER_DATA_FILE, user_data)
    with open('./debug.json', "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)

@app.route("/", methods=["GET", "POST"])
def index():
    form = RelPathForm()
    username = request.args.get("username", DEFAULT_USER)
    print(f"[debug] index username = {username}")
    global user_data
    if username not in user_data:
        user_data[username] = default_user_data
    # print(f"[debug] user_data['username'] = {user_data}")

    if request.method == "POST":
        user_data[username]['rel_path'] = form.rel_path.data
        qa_dir = os.path.join(qa_path, user_data[username]['rel_path'])
        # print(f"[debug] user_data['username'] = {user_data['username']}")

        user_data[username]['json_files'] = []
        user_data[username]['json_numbers'] = []
        # Populate the JSON number choices if directory exists
        if os.path.exists(qa_dir):
            user_data[username]['json_files'] = [f for f in os.listdir(qa_dir) if f.endswith(".json")]
            user_data[username]['json_numbers'] = sorted(
                [f.split('.')[0] for f in user_data[username]['json_files'] if f.split('.')[0].isdigit() and len(f.split('.')[0]) == 5]
            )
            form.json_number.choices = [(num, num) for num in user_data[username]['json_numbers']]
        else:
            form.json_number.choices = []

        user_data[username]['selected_number'] = form.json_number.data

        # Load the selected data if a number is chosen
        if user_data[username]['selected_number']:
            update_content(username)
    save_json(USER_DATA_FILE, user_data)

    return render_template("index.html", form=form, selected_number=user_data[username]['selected_number'], 
                           content=user_data[username]['content'])

@app.route("/edit_json", methods=["POST"])
def edit_json():
    global user_data
    data = request.json
    key_path = data.get("key_path")
    new_value = data.get("new_value")
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] edit_json username = {username}")

    if not key_path or new_value is None:
        return jsonify({"error": "Invalid request"}), 400
    
    json_name = key_path[0]
    if json_name not in user_data[username]['json_content'].keys():
        return "Invalid JSON name", 400

    try:
        # get the old value
        old_value = get_nested_value(user_data[username]['json_content'], key_path)
        # change to new value's datatype
        parsed_value = convert_value_type(old_value, new_value)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    # modify JSON content
    def set_nested_value(d, path, value):
        for i, key in enumerate(path[:-1]):
            if isinstance(d, list):
                # if list, index should be in int
                try:
                    final_key = int(key)
                    if int(key) >= len(d):
                        raise ValueError(f"Invalid list index '{key}'(out of range) at path position {i}.")
                    if len(d) > 0 and isinstance(d[0], dict):
                        for j in range(len(d)):
                            if d[j]['dict_id'] == key:
                                final_key = j
                    d = d[final_key]
                except ValueError:
                    raise TypeError(f"Invalid list index '{key}' at path position {i}.")
            elif isinstance(d, dict):
                # if dict, setdefault
                d = d.setdefault(key, {})
            else:
                raise TypeError(f"Unexpected type {type(d)} encountered at path position {i}.")
        
        # final value
        if isinstance(d, list):
            try:
                key = int(path[-1])
                if key >= len(d):
                    # if index out of range, append the list
                    d.extend([None] * (key - len(d) + 1))
                d[key] = value
            except ValueError:
                raise TypeError(f"Invalid list index '{path[-1]}' at final path position.")
        elif isinstance(d, dict):
            d[path[-1]] = value
        else:
            raise TypeError(f"Unexpected type {type(d)} encountered at final path position.")

    set_nested_value(user_data[username]['json_content'], key_path, parsed_value)

    path_dict = get_path_dict(username)
    for key in user_data[username]['json_content'].keys():
        if key == json_name:
            if key == 'anno_json':
                with gzip.open(path_dict[key], "wt", encoding="utf-8") as gz_file:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), gz_file, 
                                                ensure_ascii=False, indent=4)
            else:
                with open(path_dict[key], "w", encoding="utf-8") as f:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), f, ensure_ascii=False, indent=4)
                with open("debug.json", "w", encoding="utf-8") as f:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), f, ensure_ascii=False, indent=4)

    log_edit(action="modify", nodepath="/".join(key_path), filepath=path_dict[json_name], 
             logpath=EDIT_LOG_PATH, old_value=old_value, new_value=parsed_value)
    return jsonify({"success": True})

@app.route("/delete_json", methods=["POST"])
def delete_json():
    global user_data
    data = request.json
    key_path = data.get("key_path")
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] delete_json username = {username}")

    if not key_path:
        return jsonify({"error": "Invalid request"}), 400
    
    json_name = key_path[0]
    if json_name not in user_data[username]['json_content'].keys():
        return "Invalid JSON name", 400
    
    old_value = get_nested_value(user_data[username]['json_content'], key_path)

    # 删除节点
    def delete_nested_value(d, path):
        for i, key in enumerate(path[:-1]):
            if isinstance(d, list):
                final_key = int(key)
                if len(d) > 0 and isinstance(d[0], dict):
                    for j in range(len(d)):
                        if d[j]['dict_id'] == key:
                            final_key = j
                d = d[final_key]
            elif isinstance(d, dict):
                d = d[key]
            else:
                raise TypeError(f"Unexpected type {type(d)} at path position {i}.")
        if isinstance(d, list):
            del d[int(path[-1])]
        elif isinstance(d, dict):
            del d[path[-1]]
        else:
            raise TypeError("Cannot delete from non-dict or non-list object.")

    try:
        delete_nested_value(user_data[username]['json_content'], key_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    path_dict = get_path_dict(username)
    for key in user_data[username]['json_content'].keys():
        if key == json_name:
            if key == 'anno_json':
                with gzip.open(path_dict[key], "wt", encoding="utf-8") as gz_file:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), gz_file, 
                              ensure_ascii=False, indent=4)
            else:
                with open(path_dict[key], "w", encoding="utf-8") as f:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), f, ensure_ascii=False, indent=4)

    log_edit(action="delete", nodepath="/".join(key_path), filepath=path_dict[json_name],
             logpath=EDIT_LOG_PATH, old_value=old_value)
    return jsonify({"success": True})

@app.route("/add_json", methods=["POST"])
def add_json():
    global user_data
    data = request.json
    key_path = data.get("key_path")
    child_type = data.get("child_type")
    new_value = data.get("new_value")
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] add_json username = {username}")

    # print(f'[debug] request.json = {request.json}')

    if not key_path or not child_type or new_value is None:
        return jsonify({"error": "Invalid request"}), 400
    
    json_name = key_path[0]
    if json_name not in user_data[username]['json_content'].keys():
        return "Invalid JSON name", 400

    try:
        if child_type == "int":
            new_value = int(new_value)
        elif child_type == "float":
            new_value = float(new_value)
        elif child_type == "bool":
            new_value = new_value.lower() in ("true", "1")
        elif child_type == "str":
            new_value = str(new_value)
        else:
            raise ValueError(f"Unsupported type: {child_type}")
    except ValueError:
        return jsonify({"error": "Invalid value type"}), 400

    def add_nested_value(d, path, value):
        for i, key in enumerate(path):
            if isinstance(d, list):
                final_key = int(key)
                if len(d) > 0 and isinstance(d[0], dict):
                    for j in range(len(d)):
                        if d[j]['dict_id'] == key:
                            final_key = j
                d = d[final_key]
            elif isinstance(d, dict):
                d = d[key]
            else:
                raise TypeError(f"Unexpected type {type(d)} at path position {i}.")
        if isinstance(d, list):
            if isinstance(value, dict):
                value['dict_id'] = len(d)
            d.append(value)
        else:
            raise TypeError("Can only add value to a list.")

    try:
        add_nested_value(user_data[username]['json_content'], key_path, new_value)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    path_dict = get_path_dict(username)
    for key in user_data[username]['json_content'].keys():
        if key == json_name:
            if key == 'anno_json':
                with gzip.open(path_dict[key], "wt", encoding="utf-8") as gz_file:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), gz_file, 
                              ensure_ascii=False, indent=4)
            else:
                with open(path_dict[key], "w", encoding="utf-8") as f:
                    json.dump(delete_list_id(user_data[username]['json_content'][json_name]), f, ensure_ascii=False, indent=4)

    log_edit(action="append", nodepath="/".join(key_path), filepath=path_dict[json_name], logpath=EDIT_LOG_PATH, 
             old_value=None, new_value=new_value)
    return jsonify({"success": True})

@app.route("/history", methods=["POST"])
def get_history():
    data = request.json
    path = data.get("path")
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] history username = {username}")
    if not path:
        return jsonify({"error": "Path is required"}), 400
    
    path_dict = get_path_dict(username)
    try:
        history_entries = []
        with open(EDIT_LOG_PATH, "r", encoding="utf-8") as log_file:
            for line in log_file:
                entry = eval(line.strip())
                if entry.get("path"):
                    key = entry["path"].split('/')[0]
                    # print(f'[debug] path_dict[{key}] = {path_dict[key]}, entry["file"] = {entry["file"]}, path = {path}, entry["path"] = {entry["path"]}')
                    if path_dict[key] == entry["file"] and path == entry["path"]:
                        if entry['old_value'] is not None:
                            entry['old_value'] = html.escape(entry['old_value'])
                        if entry['new_value'] is not None:
                            entry['new_value'] = html.escape(entry['new_value'])
                        history_entries.append(entry)
        
        return jsonify(history_entries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_all_history", methods=["GET"])
def get_all_history():
    username = request.args.get("username", DEFAULT_USER)
    path_dict = get_path_dict(username)
    print(f"[debug] get_all_history username = {username}")
    try:
        history = []
        with open(EDIT_LOG_PATH, "r") as log_file:
            for line in log_file:
                entry = eval(line.strip())
                for key in path_dict.keys():
                    if path_dict[key] == entry['file']:
                        if entry['old_value'] is not None:
                            entry['old_value'] = html.escape(entry['old_value'])
                        if entry['new_value'] is not None:
                            entry['new_value'] = html.escape(entry['new_value'])
                        history.append(entry)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mark", methods=["POST"])
def mark_status():
    status = request.args.get("status")
    username = request.args.get("username", DEFAULT_USER)
    path = get_path_dict(username)['qa_json']
    print(f"[debug] mark username = {username}")

    if not path or not status:
        return jsonify({"success": False, "error": "Missing path or status"}), 400

    status_data = load_status(STATUS_FILE)
    if status != 'controversy' and check_controversy() is True:
        return jsonify({"success": False, "error": f"Can not mark {status} when controversial value exists in this file, please check."}), 200
    
    status_data[path] = status
    save_status(status_data, STATUS_FILE)

    return jsonify({"success": True})

@app.route("/mark_entry_exit", methods=["POST"])
def mark_entry_exit():
    global user_data
    side = request.args.get("side")
    frame = request.args.get("frame")
    username = request.args.get("username", DEFAULT_USER)
    path = get_path_dict(username)['qa_json'].split('/')[-2]
    print(f"[debug] mark_entry_exit username = {username}")

    if not path or not side or not frame or side not in ['entry', 'exit']:
        return jsonify({"success": False, "error": "Missing args"}), 400

    entry_exit_data = load_status(ENTRY_EXIT_FILE)

    if path not in entry_exit_data:
        entry_exit_data[path] = {}
        entry_exit_data[path]['entry'] = int(user_data[username]['json_numbers'][0])
        entry_exit_data[path]['exit'] = int(user_data[username]['json_numbers'][-1])
    
    entry_exit_data[path][side] = int(frame)

    save_status(entry_exit_data, ENTRY_EXIT_FILE)

    return jsonify({"success": True})

@app.route("/get_entry_exit", methods=["POST"])
def get_entry_exit():
    global user_data
    username = request.args.get("username", DEFAULT_USER)
    path = get_path_dict(username)['qa_json'].split('/')[-2]
    print(f"[debug] mark_entry_exit username = {username}")

    if not path:
        return jsonify({"success": False, "error": "Missing paths"}), 400

    entry_exit_data = load_status(ENTRY_EXIT_FILE)

    if path not in entry_exit_data:
        entry_exit_data[path] = {}
        entry_exit_data[path]['entry'] = int(user_data[username]['json_numbers'][0])
        entry_exit_data[path]['exit'] = int(user_data[username]['json_numbers'][-1])
        # save_status(entry_exit_data, STATUS_FILE) # not need to save

    return jsonify({"success": True, 
                    "entry": entry_exit_data[path]['entry'],
                    "exit": entry_exit_data[path]['exit'],
                    })

@app.route("/mark_value", methods=["POST"])
def mark_value():
    data = request.json
    path = data.get("path")
    status = data.get("status")
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] mark_value username = {username}")
    path_dict = get_path_dict(username)

    if not path or not status:
        return jsonify({"success": False, "error": "Missing path or status"}), 400

    status_data = load_status(VALUE_STATUS_FILE)
    json_name = path[0]
    path_str = '/'.join(path)
    file_path = path_dict[json_name]
    if file_path not in status_data.keys():
        status_data[file_path] = {}
    status_data[file_path][path_str] = status

    save_status(status_data, VALUE_STATUS_FILE)

    if status == "controversy":
        status_data = load_status(STATUS_FILE)
        status_data[path_dict['qa_json']] = status
        save_status(status_data, STATUS_FILE)

    return jsonify({"success": True})

def check_controversy():
    username = request.args.get("username", DEFAULT_USER)
    path_dict = get_path_dict(username)
    status_data = load_status(VALUE_STATUS_FILE)

    for path in path_dict.values():
        # print(f'[debug] checking path: {path}')
        if path in status_data.keys():
            # print(f'[debug] checking path: {path}')
            for key in status_data[path].keys():
                # print(f'[debug] checking key: {key}')
                if status_data[path][key] == 'controversy':
                    return True
    return False

@app.route("/get_status", methods=["GET"])
def get_status():
    username = request.args.get("username", DEFAULT_USER)
    path = get_path_dict(username)['qa_json']
    print(f"[debug] get_status username = {username}")
    key_path = request.args.get('path', None)
    if key_path is not None:
        status = "raw"  # default set to raw
        json_name = key_path.split('/')[0]
        file_path = get_path_dict(username)[json_name]
        with open(VALUE_STATUS_FILE, "r") as f:
            status_data = json.load(f)
            if file_path in status_data:
                status = status_data[file_path].get(key_path, "raw")
    else:
        status_data = load_status(STATUS_FILE)
        status = status_data.get(path, "raw")

    return jsonify({"success": True, "status": status})

def filter_json_tree(json_data, filter_keywords):
    if not filter_keywords:
        return json_data

    def matches_keyword(value, keyword):
        if keyword.startswith("*") and keyword.endswith("*"):
            return keyword[1:-1] in value
        elif keyword.startswith("*"):
            return value.endswith(keyword[1:])
        elif keyword.endswith("*"):
            return value.startswith(keyword[:-1])
        else:
            return value == keyword

    def matches_any_keyword(value):
        return any(matches_keyword(value, keyword) for keyword in filter_keywords)

    def recursive_filter(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if matches_any_keyword(key) or key == "dict_id":
                    result[key] = value
                else:
                    # recursive filter
                    filtered_value = recursive_filter(value)
                    if filtered_value is not None:
                        result[key] = filtered_value
            return result if result else None
        elif isinstance(obj, list):
            result = []
            for item in obj:
                filtered_item = recursive_filter(item)
                if filtered_item is not None:
                    result.append(filtered_item)
            return result if result else None
        else:
            if matches_any_keyword(str(obj)):
                return obj
            else:
                return None

    return recursive_filter(json_data)

@app.route("/change_number", methods=["POST"])
def change_number():
    data = request.json
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] change_number username = {username}")
    global user_data

    if not user_data[username]['selected_number']:
        return jsonify({"error": "No selected number"}), 400

    direction = data.get("direction")

    if not direction or direction not in ["prev", "next"]:
        return jsonify({"error": "Invalid direction"}), 400

    json_numbers = user_data[username]['json_numbers']

    if user_data[username]['selected_number'] not in json_numbers:
        return jsonify({"error": "Selected number not in list"}), 404

    current_index = json_numbers.index(user_data[username]['selected_number'])

    if direction == "prev":
        new_index = (current_index - 1) % len(json_numbers)
    elif direction == "next":
        new_index = (current_index + 1) % len(json_numbers)

    user_data[username]['selected_number'] = json_numbers[new_index]
    save_json(USER_DATA_FILE, user_data)

    update_content(username)

    return jsonify({"selected_number": user_data[username]['selected_number'], 
                    "content": user_data[username]['content']})

@app.route("/filter", methods=["POST"])
def filter_json():
    # global json_content, filter_dict, vqa_filter, show_key_object_info
    global user_data
    data = request.json
    filter_keywords = data.get("keywords", [])
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] filter_json username = {username}")
    json_key = data.get("key")

    if json_key not in user_data[username]['json_content']:
        return "Invalid JSON key", 400

    # filter JSON
    if isinstance(user_data[username]['filter_dict'][json_key], (dict, list)):
        user_data[username]['filter_dict'][json_key] = filter_keywords
        filtered_json = filter_json_tree(user_data[username]['json_content'][json_key], filter_keywords)
        html_response = json_to_html(filtered_json, get_path_dict(username), VALUE_STATUS_FILE, 
                                     user_data[username]['current_view'], 
                                     user_data[username]['vqa_filter'], 
                                     user_data[username]['show_key_object_info'], json_key)
        save_json(USER_DATA_FILE, user_data)
    else:
        html_response = "<p>not exist</p>"

    # print(json_key)
    # print(html_response)
    return html_response

@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    username = request.args.get("username", DEFAULT_USER)
    print(f"[debug] toggle_view username = {username}")
    global user_data
    toggle = request.args.get('toggle', 0)
    if toggle == '1':
        if user_data[username]['current_view'] == "VQA View":
            user_data[username]['current_view'] = "Full View"
        else:
            user_data[username]['current_view'] = "VQA View"
    save_json(USER_DATA_FILE, user_data)
    return jsonify({"current_view": user_data[username]['current_view']})

@app.route('/toggle_key_object_info', methods=['POST'])
def toggle_key_object_info():
    global user_data
    data = request.json
    toggle = data.get("toggle", 0)
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] toggle_key_object_info username = {username}")

    if toggle == 1:
        user_data[username]['show_key_object_info'] = not user_data[username]['show_key_object_info']
        save_json(USER_DATA_FILE, user_data)
    return jsonify({"flag": user_data[username]['show_key_object_info']})

@app.route('/overview')
def overview():
    global qa_path
    if not os.path.exists(qa_path):
        return render_template('overview.html', error='QA Path does not exist.')
    
    folders = get_first_level_dirs(qa_path)
    folders = sorted([d for d in folders if '_' in d])
    folder_status = {}

    for folder in folders:
        raw_count, controversial_count, verified_count = get_status_count(qa_path, folder, STATUS_FILE)
        folder_status[folder] = {
            "raw": raw_count,
            "controversial": controversial_count,
            "verified": verified_count
        }

    return render_template('overview.html', folders=folders, folder_status=folder_status)

@app.route("/qa_filter", methods=["POST"])
def update_vqa_filter():
    global user_data

    data = request.json
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] updata_vqa_filter username = {username}")
    filter_qids = data.get("ids", [])
    user_data[username]['vqa_filter'] = filter_qids
    save_json(USER_DATA_FILE, user_data)

    return jsonify({"success": True})

@app.route('/get_options', methods=['POST'])
def get_options():
    global user_data
    data = request.json
    # print(f'[debug] data = {data}')
    path = data['path']
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] get_options username = {username}")
    path[-1] = 'qid'
    # print(f'[debug] path = {path}')
    common_options = load_status(COMMON_OPTION_FILE)
    key = str(get_nested_value(user_data[username]['json_content'], path))
    options = common_options.get(key, [])
    for value in options:
        value = html.escape(value)
    # print(f'[debug] key = {key}, options = {options}')
    return jsonify(options)

@app.route('/add_option', methods=['POST'])
def add_to_options():
    global user_data
    data = request.json
    path = data['path']
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] add_to_options username = {username}")
    path[-1] = 'qid'
    new_option = data['value']
    common_options = load_status(COMMON_OPTION_FILE)
    key = str(get_nested_value(user_data[username]['json_content'], path))

    if key not in common_options:
        common_options[key] = []
    
    if new_option not in common_options[key]:
        common_options[key].append(new_option)
    
    save_status(common_options, COMMON_OPTION_FILE)
    
    return jsonify({"success": True})

@app.route('/remove_option', methods=['POST'])
def remove_from_options():
    global user_data
    data = request.json
    path = data['path']
    username = data.get("username", DEFAULT_USER)
    print(f"[debug] remove_option username = {username}")
    path[-1] = 'qid'
    option = data['value']
    # print(f'[debug] option = {option}')
    common_options = load_status(COMMON_OPTION_FILE)
    key = str(get_nested_value(user_data[username]['json_content'], path))
    
    if key in common_options and option in common_options[key]:
        common_options[key].remove(option)

    save_status(common_options, COMMON_OPTION_FILE)
    
    return jsonify({"success": True})


@app.route('/check_user', methods=['GET'])
def check_user():
    global user_data
    username = request.args.get('username')
    if username in user_data:
        return jsonify({"exists": True})
    return jsonify({"exists": False})

@app.route('/create_user', methods=['POST'])
def create_user():
    global user_data
    data = request.json
    username = data.get("username")
    if username in user_data:
        return jsonify({"success": False, "error": "User already exists."})
    user_data[username] = default_user_data
    save_json(USER_DATA_FILE, user_data)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)
