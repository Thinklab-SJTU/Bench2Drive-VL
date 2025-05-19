from inference_workers import InferenceWorker
from infer_configure import InferConfig
from io_utils import *

def get_carla_inference_worker(config_dir, model_name, model_path, outpath, gpu_id):
    print("======== Configuration Starts ========")
    config = InferConfig(config_dir, in_carla=True)
    config.display_config()
    print("========= Configuration Ends =========")

    worker = InferenceWorker(worker_id=gpu_id, scenario_list=None,
                             dataset=None, wp_code=None, transform=None,
                             model=None, model_name=model_name, model_path=model_path,
                             outpath=outpath, configs=config, in_carla=True)
    
    # worker.init_model() 
    # The model needs to be hosted separately using a web app 
    # because Transformer and CARLA are difficult 
    # to run in the same Python environment due to compatibility issues.
                            
    return worker