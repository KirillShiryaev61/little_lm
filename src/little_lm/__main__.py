import argparse
from little_lm.cli import main

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