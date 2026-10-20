from pydantic import BaseModel, ConfigDict

class BaseModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    data_path: str
    context_size: int
    hidden_dim: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    lenght_answer: int
    temperature: float
    random_state: int
    device: str | None