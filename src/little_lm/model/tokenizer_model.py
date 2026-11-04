import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from little_lm.core.constant import MODELS_DIR
from little_lm.core.model_types import PreparedData
from little_lm.core.schema import BaseModelConfig
from little_lm.model.base_model_v2 import BaseModelV2, Net
from little_lm.tokenizers.simple_tokenizer_v1 import SimpleTokenizerV1

class TokenizerModel(BaseModelV2):

    def __init__(
        self, 
        model_name: str, 
        config: BaseModelConfig,
    ) -> None:
        self.model_name = model_name
        
        self.data_path = config.data_path
        self.context_size = config.context_size
        self.hidden_dim = config.hidden_dim
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.lenght_answer = config.lenght_answer
        self.temperature = config.temperature
        self.random_state = config.random_state
        if config.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = config.device

        self.tokenizer = None
        self.net = None
        self.optimizer = None
        self.loss = None
        self.dataloader = None
        self.trained = False
        self.vocab = None
        self.vocab_size = None
        self.data = None

        # Фиксирование рандома
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True)

    def _prepare_data(
        self,
        data_path: str | None = None,
    ) -> None:
        """Подготовка данных к обучению модели"""
        if data_path is None:
            data_path = self.data_path

        # Загрузка данных
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Токены
        preprocessed = re.split(r'(--|[,.:;?_!"()«»\'—\-\s])', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_tokens = sorted(set(preprocessed))

        self.vocab = {token:integer for integer, token in enumerate(all_tokens)}
        self.tokenizer = SimpleTokenizerV1(self.vocab)
        self.vocab_size = len(self.tokenizer.char_to_id)

        # Формирование обучающей выборки
        ids = torch.tensor(self.tokenizer.encode(text), device=self.device)

        x_train = torch.stack(
            [ids[i:i - self.context_size] for i in range(self.context_size)], 
            dim=1
        )
        y_train = ids[self.context_size:]

        self.data = PreparedData(
            x_train=x_train.to(self.device), 
            y_train=y_train.to(self.device),
        )
        return None

    def _init_models(self) -> None:
        """Инициализация модели"""
        self.net = Net(
            self.vocab_size, 
            self.hidden_dim, 
            self.context_size,
        ).to(self.device)
        self.net.apply(self._init_weights)

        self.optimizer = torch.optim.SGD(
            self.net.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-6
        )
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        return None


    def save_weights(
        self,
        filepath: Path,
    ) -> None:
        """Сохранение весов и метаданных модели на диск"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "vocab": self.vocab,
            },
            filepath,
        )
        print(f"Веса успешно сохранены в {filepath}")
        return None

    def load_weights(
        self,
        filepath: Path,
    ) -> None:
        """Загрузка сохраненных весов и токенов"""
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
            weights_only=True
        )
        self.vocab = checkpoint["vocab"]
        self.tokenizer = SimpleTokenizerV1(self.vocab)
        self.vocab_size = len(self.tokenizer.char_to_id)

        self._init_models()
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
        self.trained = True
        print(f"Веса успешно загружены из {filepath}")
        return None

    def train(self) -> None:
        print("Запуск обучения модели...")

        self._prepare_data()
        self._init_models()

        report_loss = []
        dataset = TensorDataset(self.data.x_train, self.data.y_train)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            self.net.train()
            epochs_loss = 0.0
            num_batches = 0
            for x_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                preds = self.net(x_batch)
                loss_value = self.loss(preds.view(-1, self.vocab_size), y_batch.view(-1))
                loss_value.backward()
                self.optimizer.step()

                num_batches += 1
                epochs_loss += loss_value.item()

            # Блок для оценки
            # with torch.no_grad():

            report_loss.append(epochs_loss / num_batches)

        self.trained = True
        print(f"Выполнилось {epoch:02d} эпох | Средняя ошибка последней эпохи: {report_loss[-1]:.4f}")
        print("Обучение модели закончено")
        self.save_weights(filepath=MODELS_DIR / self.model_name / f"{self.model_name}.pt")
        self.get_loss_plot(report_loss)
        return None

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Генерация ответа языковой модели"""
        if not self.trained:
            raise RuntimeError("Модель еще не обучена")

        current_ids = self.tokenizer.encode(prompt)[-self.context_size:]
        if len(current_ids) < self.context_size:
            pad_needs = self.context_size - len(current_ids)
            current_ids = [self.tokenizer.pad_token_id] * pad_needs + current_ids

        for _ in range(self.lenght_answer):
            x_in = torch.tensor([current_ids[-self.context_size:]], device=self.device)
            preds = self.net(x_in)

            # Применение температуры к вероятностям
            temp = max(self.temperature, 1e-4)
            probs = torch.softmax(preds[0] / temp, dim=-1)

            # Семплируем следующий символ
            next_id = torch.multinomial(
                probs, 
                num_samples=1, 
                replacement=True,
            ).item()

            current_ids.append(next_id)
        
        generated = self.tokenizer.decode(current_ids)
        print(generated)
        return generated
