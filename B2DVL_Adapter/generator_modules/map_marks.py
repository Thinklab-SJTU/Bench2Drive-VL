import json
import os

util_path = os.path.join(os.environ.get('WORK_DIR', '.'), 'B2DVL_Adapter/generator_modules/util')

with open(os.path.join(util_path, "tunnels.json"), "r") as file:
    TUNNELS = json.load(file)