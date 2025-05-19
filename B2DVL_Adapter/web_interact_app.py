import argparse
import yaml
import json
from fastapi import FastAPI
from pydantic import BaseModel
from models import get_model_interface
from typing import List
from inference_utils import Bubble
import uvicorn
import os

def load_config(config_path):
    _, ext = os.path.splitext(config_path)
    with open(config_path, "r") as f:
        if ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format: use .yaml/.yml or .json")

def create_app(model_interface):
    app = FastAPI()

    @app.post("/interact")
    def interact_route(request: dict):
        bubble = Bubble.from_dict(request["bubble"])
        conversation = [Bubble.from_dict(bb) for bb in request.get("conversation", [])]
        output = model_interface.interact(bubble, conversation)
        # print(f"[debug] VLM output: {output}")
        return {"response": output}

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)

    def get_cfg(path: List[str], default=None):
        ref = config
        try:
            for key in path:
                ref = ref[key]
            return ref
        except KeyError:
            return default

    model_interface = get_model_interface(config.get("MODEL_NAME"))

    model_interface.initialize(
        gpu_id=config.get("GPU_ID", 0),
        use_all_cameras=get_cfg(["INFERENCE_BASICS", "USE_ALL_CAMERAS"], False),
        no_history=get_cfg(["INFERENCE_BASICS", "NO_HISTORY_MODE"], False),
        input_window=get_cfg(["INFERENCE_BASICS", "INPUT_WINDOW"], 1),
        frame_rate=get_cfg(["TASK_CONFIGS", "FRAME_PER_SEC"], 10),
        model_path=config.get("MODEL_PATH", ""),
        use_bev=get_cfg(["INFERENCE_BASICS", "USE_BEV"], False),
        in_carla=config.get("IN_CARLA", False),
        use_base64=config.get("USE_BASE64", False)
    )

    app = create_app(model_interface)
    host = "0.0.0.0"
    port = config.get("PORT", 7023)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
