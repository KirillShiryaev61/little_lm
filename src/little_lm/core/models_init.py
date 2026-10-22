from little_lm.model.base_model import BaseModel
from little_lm.model.model_v2 import ModelV2
from little_lm.core.schemas import BaseModelConfig

MODELS = {
    'test_base_model': (BaseModel, BaseModelConfig),
    'base_model': (BaseModel, BaseModelConfig),
    'model_v2': (ModelV2, BaseModelConfig)
}

def get_model_n_config_class(model_name: str):
    if model_name not in MODELS.keys():
        raise ValueError(f"Нет доступной модели для проверки гиперпараметров, доступны {MODELS.keys()}")
    return MODELS[model_name]
