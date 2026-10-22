import argparse
from contextlib import redirect_stdout
import io
import polars as pl
import yaml
from little_lm.core.constant import MODELS_DIR, PROMPTS
from little_lm.core.models_init import get_model_n_config_class

def main(model_name: str):
    Model, ModelConfig = get_model_n_config_class(model_name)
    weights_path = MODELS_DIR / model_name / f"{model_name}.pt"
    config_path = MODELS_DIR / model_name / "config.yaml"
    data_path = MODELS_DIR / model_name / "generate_model_text.csv"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_params = ModelConfig(**config["model"])
    model = Model(model_name, model_params)
    if weights_path.is_file():
        model.load_weights(weights_path)
    else:
        model.train()

    data = []
    for prompt in PROMPTS:
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            model.generate(prompt)
        generated_text = buffer.getvalue()
        row = {
            "prompt": prompt,
            "generated_text": generated_text
        }
        data.append(row)
    data = pl.DataFrame(data)
    data.write_csv(data_path)
    print("Сбор данных для анализа закончен.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сбор генеративного текста для оценки")
    parser.add_argument(
            "--model_name", 
            type=str, 
            default='base_model',
            help="Имя модели для сбора данных",
    )
    args = parser.parse_args()
    main(model_name=args.model_name)
