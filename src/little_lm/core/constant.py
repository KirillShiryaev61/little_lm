from pathlib import Path

ACTUAL_MODEL = 'base_model_v2'
SLASH = '=' * 65
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
PROMPTS = (
    "Мой дядя самых",
    "Учил его всему шутя",
    "Как признаться в любви девушке?",
    "Какая сейчас погода в Санкт-Петербурге?"
)
