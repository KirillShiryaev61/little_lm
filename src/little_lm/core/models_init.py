from little_lm.model.base_model import BaseModel
from little_lm.model.base_model_v2 import BaseModelV2
from little_lm.core.schema import BaseModelConfig

MODELS = {
    'test_base_model': (BaseModel, BaseModelConfig),
    'test_base_model_v2': (BaseModelV2, BaseModelConfig),
    
    'base_model': (BaseModel, BaseModelConfig),
    'base_model_v2': (BaseModelV2, BaseModelConfig),
}

def get_model_n_config_class(model_name: str):
    if model_name not in MODELS.keys():
        raise ValueError(f"Нет доступной модели для проверки гиперпараметров, доступны {MODELS.keys()}")
    return MODELS[model_name]
