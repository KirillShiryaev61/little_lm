import argparse
from little_lm.core.constant import ACTUAL_MODEL
from little_lm.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=ACTUAL_MODEL,
        help="Имя модели для инференса",
    )
    args = parser.parse_args()
    main(model_name=args.model_name)