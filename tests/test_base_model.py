import yaml
from core.constant import MODELS_DIR
from core.model_types import PreparedData
from core.models_init import get_model_n_config_class

MODEL_NAME = 'test_base_model'
MODEL, MODEL_CONFIG = get_model_n_config_class(MODEL_NAME)
CONFIG_PATH = MODELS_DIR / MODEL_NAME / "config.yaml"
WEIGHTS_PATH = MODELS_DIR / MODEL_NAME / f"{MODEL_NAME}.pt"
PLOT_PATH = MODELS_DIR / MODEL_NAME / "train_loss_plot.png"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
MODEL_PARAMS = MODEL_CONFIG(**config["model"])

class TestBaseModel:

    def test_prepare_data(self):
        """Базовая проверка корректности работы prepare_data"""
        model = MODEL(MODEL_NAME, MODEL_PARAMS)
        model._prepare_data()

        assert isinstance(model.data, PreparedData)
        assert isinstance(model.char_to_id, dict)
        assert isinstance(model.id_to_char, dict)
        assert model.data.x_train.ndim == 2
        assert model.data.x_train.shape[1] == 12
        assert model.data.y_train.ndim == 1
        assert model.data.x_train.shape[0] == model.data.y_train.shape[0]

    def test_train_model(self):
        """Тестирование обучения и сохранения весов"""
        model = MODEL(MODEL_NAME, MODEL_PARAMS)
        assert not model.trained
        
        model.train()
        assert model.trained
        assert WEIGHTS_PATH.is_file()

    def test_load_weights(self):
        """Тестирование загрузки весов модели"""
        model = MODEL(MODEL_NAME, MODEL_PARAMS)
        
        assert WEIGHTS_PATH.is_file()
        model.load_weights(WEIGHTS_PATH)
        assert model.trained

    def test_generate_model(self, capsys):
        """Тестирование генерации текста модели"""
        model = MODEL(MODEL_NAME, MODEL_PARAMS)
        model.load_weights(WEIGHTS_PATH)
        capsys.readouterr()

        model.generate("Мой дядя самых")
        generated = capsys.readouterr().out
        assert isinstance(generated, str)
        assert len(generated) == model.lenght_answer
        assert all(char in model.char_to_id for char in generated)

    def tests_clear(self):
        """Очистка файлов после тестирования"""
        WEIGHTS_PATH.unlink(missing_ok=True)
        PLOT_PATH.unlink(missing_ok=True)
        assert not WEIGHTS_PATH.is_file()
        assert not PLOT_PATH.is_file()
        