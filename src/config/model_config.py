from transformers import AutoConfig

def get_model_config(model_name_or_path: str):
    return AutoConfig.from_pretrained(model_name_or_path)
