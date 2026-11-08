from pathlib import Path

ACTUAL_MODEL = 'base_model_v2'
SLASH = '=' * 100
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
PROMPTS = (
    "Мой дядя самых",
    "Учил его всему шутя",
    "Как признаться в любви девушке?",
    "Какая сейчас погода в Санкт-Петербурге?",
    "Мой день совсем не очень... А как твой день?",
)
