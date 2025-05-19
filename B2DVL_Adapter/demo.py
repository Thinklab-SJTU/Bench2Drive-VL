from io_utils import *

DEFAULT_EVAL_CONFIGS = {
    'INFER_SUBSET': True, # True if just inference a subset mentioned in configs/subset.txt
    'USE_CHECKPOINT': True, # True if record the finished scenarios and do not infer them next time
    'SUBSET_FILE': './configs/subset.txt',
    'CHECKPOINT_FILE': './configs/finished_scenarios.txt',
    'INFERENCE_RESULT_DIR': '../B2DVL-Chain-Inference/outputs',
    'B2D_DIR': '../Bench2Drive-rep',
    'FRAME_PER_SEC': 10 # the collection rate of data
}

write_json(DEFAULT_EVAL_CONFIGS, './configs/config.json')