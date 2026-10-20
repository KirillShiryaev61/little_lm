import argparse
import yaml
from src.base_model import BaseModel
from core.constant import MODELS_DIR, SLASH
from core.schemas import BaseModelConfig

def start_session() -> None:
    print(SLASH, "\nЧат с моделью запущен:\n", SLASH, sep="")
    print("Для выхода введите 'exit', 'quit' или нажмите Ctrl+C\n", SLASH, sep="")

def end_session() -> None:
    print("\n", SLASH, "\nЗавершение сессии\n", SLASH, sep="")

def get_model_n_config_class(model_name: str):
    configs = {
        'base_model': (BaseModel, BaseModelConfig),
    }
    if model_name not in configs.keys():
        raise ValueError(f"Нет доступной модели для проверки гиперпараметров, доступны {configs.keys()}")
    return configs[model_name]

def main(model_name: str) -> None:

    weights_path = MODELS_DIR / model_name / f"{model_name}.pt"
    config_path = MODELS_DIR / model_name / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    Model, ModelConfig = get_model_n_config_class(model_name)
    model_params = ModelConfig(**config['model'])

    # Инициализация модели
    model = Model(
        model_name=model_name, 
        config=model_params
    )
    if weights_path.is_file():
        print(f"Веса найдены: {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"Веса для модели '{model_name}' не найдены в {MODELS_DIR}")
        model.train()

    # Инференс модели
    start_session()
    
    try:
        while True:
            prompt = input("Промпт: ").strip()

            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                end_session()
                break

            print(SLASH)
            model.generate(prompt=prompt)
            print("\n", SLASH, sep="")

    except (KeyboardInterrupt, EOFError):
        end_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default='base_model',
        help="Имя модели для инференса",
    )
    args = parser.parse_args()
    main(model_name=args.model_name)
