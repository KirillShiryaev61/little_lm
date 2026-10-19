import argparse
from pathlib import Path
from src.model import Base_Model

SLASH = '=' * 65

def start_session() -> None:
    print(SLASH, "\nЧат с моделью запущен:\n", SLASH, sep="")
    print("Для выхода введите 'exit', 'quit' или нажмите Ctrl+C\n", SLASH, sep="")

def end_session() -> None:
    print("\n", SLASH, "\nЗавершение сессии\n", SLASH, sep="")

def main(model_name: str) -> None:

    models_dir = Path(__file__).parent / "models"
    weights_path = models_dir / f"{model_name}.pt"

    # Инициализация модели
    model = Base_Model(model_name=model_name)
    if weights_path.is_file():
        print(f"Веса найдены: {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"Веса для модели '{model_name}' не найдены в {models_dir}")
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
        required=True,
        help="Имя модели для инференса",
    )
    args = parser.parse_args()
    main(model_name=args.model_name)
