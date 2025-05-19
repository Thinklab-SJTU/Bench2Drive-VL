MODEL_MAP = {
    "gt": "VLMInterface",
    "LLaVANeXT": "LLaVANeXTInterface",
    "Qwen2.5VL": "Qwen25Interface",
    "api": "VLMAPIInterface",
    "Gemma": "GemmaInterface",
    "Janus-Pro": "JanusProInterface",
    "InternVL": "InternVLInterface"
    # Add other models as you need
}

def get_model_interface(model_name):
    """
    Retrieve the appropriate model interface class based on the model name.
    :param model_name: Name of the model.
    :return: An instance of the corresponding model interface.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} is not supported. Available models: {list(MODEL_MAP.keys())}")
    
    class_name = MODEL_MAP[model_name]
    
    # Lazy import based on model_name
    if model_name == "gt":
        from .VLMInterface import VLMInterface
        return VLMInterface()
    elif model_name == "LLaVANeXT":
        from .LLaVA_NeXT import LLaVANeXTInterface
        return LLaVANeXTInterface()
    elif model_name == "Qwen2.5VL":
        from .qwen25 import Qwen25Interface
        return Qwen25Interface()
    elif model_name == "api":
        from .vlm_api import VLMAPIInterface
        return VLMAPIInterface()
    elif model_name == "Gemma":
        from .gemma import GemmaInterface
        return GemmaInterface()
    elif model_name == "Janus-Pro":
        from .janus import JanusProInterface
        return JanusProInterface()
    elif model_name == "InternVL":
        from .intern import InternVLInterface
        return InternVLInterface()