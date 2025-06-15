from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired
from json2html import json2html
import os
import json
import gzip
import re

app = Flask(__name__)
app.secret_key = "okitasouji"

b2d_data_path = "/media/telkwevr/22729A30729A08A5/Project_/Bench2Drive-rep"
appendix_path = "/media/telkwevr/22729A30729A08A5/Project_/Carla_Chain_QA/carla_vqa_gen/vqa_dataset/outgraph/appendix"
qa_path = "/media/telkwevr/22729A30729A08A5/Project_/Carla_Chain_QA/carla_vqa_gen/vqa_dataset/outgraph"
anno_image_path = "/media/telkwevr/22729A30729A08A5/Project_/Carla_Chain_QA/carla_vqa_gen/vqa_dataset/outexample/anno_images"

rel_path = ""
selected_number = None
content = {}
json_content = {}
filter_dict = {
    "anno_json": [],
    "appendix_json": [],
    "qa_json": [],
}
json_files = []
json_numbers = []

@app.route(f'/media/<path:filename>')
def media_files(filename):
    return send_from_directory('/.', filename)

class RelPathForm(FlaskForm):
    rel_path = StringField("Relative Path", validators=[DataRequired()])
    json_number = SelectField("Choose JSON Number", choices=[], coerce=str)
    submit = SubmitField("Load")

def update_content():
    global json_content, content, filter_dict, selected_number, rel_path
    rgb_front_path = os.path.join(anno_image_path, rel_path, f"{selected_number}.png")
    rgb_top_down_path = os.path.join(b2d_data_path, rel_path, "camera/rgb_top_down", f"{selected_number}.jpg")
    anno_json_path = os.path.join(b2d_data_path, rel_path, "anno", f"{selected_number}.json.gz")
    appendix_json_path = os.path.join(appendix_path, rel_path, f"{selected_number}.json")
    qa_json_path = os.path.join(qa_path, rel_path, f"{selected_number}.json")

    content["rgb_front"] = rgb_front_path if os.path.exists(rgb_front_path) else "not exist"
    content["rgb_top_down"] = rgb_top_down_path if os.path.exists(rgb_top_down_path) else "not exist"
    json_content["anno_json"] = (
        load_gzip_json(anno_json_path) if os.path.exists(anno_json_path) else "not exist"
    )
    json_content["appendix_json"] = load_json(appendix_json_path) if os.path.exists(appendix_json_path) else "not exist"
    json_content["qa_json"] = load_json(qa_json_path) if os.path.exists(qa_json_path) else "not exist"

    for key in ["anno_json", "appendix_json", "qa_json"]:
        if isinstance(json_content[key], (dict, list)):
            content[key] = filter_json_tree(json_content[key], filter_dict[key])
            content[key] = json2html.convert(json=content[key])

@app.route("/", methods=["GET", "POST"])
def index():
    form = RelPathForm()
    global rel_path, selected_number, content, json_content, filter_dict
    
    json_content = {
        "anno_json": "not exist",
        "appendix_json": "not exist",
        "qa_json": "not exist",
    }
    content = {
        "rgb_front": "not exist",
        "rgb_top_down": "not exist",
        "anno_json": "not exist",
        "appendix_json": "not exist",
        "qa_json": "not exist",
    }

    if request.method == "POST":
        rel_path = form.rel_path.data
        qa_dir = os.path.join(qa_path, rel_path)

        global json_files, json_numbers
        json_files = []
        json_numbers = []
        # Populate the JSON number choices if directory exists
        if os.path.exists(qa_dir):
            json_files = [f for f in os.listdir(qa_dir) if f.endswith(".json")]
            json_numbers = sorted(
                [f.split('.')[0] for f in json_files if f.split('.')[0].isdigit() and len(f.split('.')[0]) == 5]
            )
            form.json_number.choices = [(num, num) for num in json_numbers]
        else:
            form.json_number.choices = []

        selected_number = form.json_number.data

        # Load the selected data if a number is chosen
        if selected_number:
            update_content()

    return render_template("index.html", form=form, selected_number=selected_number, content=content)


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
                if matches_any_keyword(key):
                    result[key] = value
                else:
                    # 递归过滤值
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
    global selected_number

    if not selected_number:
        return jsonify({"error": "No selected number"}), 400

    data = request.json
    direction = data.get("direction")

    if not direction or direction not in ["prev", "next"]:
        return jsonify({"error": "Invalid direction"}), 400

    global json_files, json_numbers

    if selected_number not in json_numbers:
        return jsonify({"error": "Selected number not in list"}), 404

    current_index = json_numbers.index(selected_number)

    if direction == "prev":
        new_index = (current_index - 1) % len(json_numbers)
    elif direction == "next":
        new_index = (current_index + 1) % len(json_numbers)

    selected_number = json_numbers[new_index]

    update_content()

    return jsonify({"selected_number": selected_number, "content": content})

@app.route("/filter", methods=["POST"])
def filter_json():
    global json_content, filter_dict

    data = request.json
    filter_keywords = data.get("keywords", [])
    json_key = data.get("key")
    # print(json_content)

    if json_key not in json_content:
        return "Invalid JSON key", 400

    # 过滤指定的 JSON
    if isinstance(filter_dict[json_key], (dict, list)):
        filter_dict[json_key] = filter_keywords
        filtered_json = filter_json_tree(json_content[json_key], filter_keywords)
        html_response = json2html.convert(json=filtered_json)
    else:
        html_response = "<p>not exist</p>"

    # print(json_key)
    # print(html_response)
    return html_response

def load_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return "Error loading JSON"


def load_gzip_json(gzip_path):
    try:
        with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return "Error loading GZ JSON"


if __name__ == "__main__":
    app.run(debug=True)
