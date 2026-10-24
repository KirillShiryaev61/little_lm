from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from little_lm.core.constant import MODELS_DIR
from little_lm.core.schema import BaseModelConfig
from little_lm.core.model_types import PreparedData

class Net(nn.Module):
    def __init__(
            self,
            vocab_size, 
            hidden_dim,
            context_size,
        ) -> None:
        
        super(Net, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        self.linear_1 = nn.Linear(self.context_size * self.vocab_size, self.hidden_dim)
        self.act_1 = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.eye(self.vocab_size, device=x.device)[x].reshape(len(x), -1)
        
        x = self.linear_1(x)
        x = self.act_1(x)
        x = self.linear_2(x)

        return x

class BaseModelV2():
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

        self.net = None
        self.optimizer = None
        self.loss = None
        self.dataloader = None
        self.trained = False
        self.vocab_size = None
        self.char_to_id = {}
        self.id_to_char = {}
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
        vocab = sorted(set(text))

        # Словари для преобразования токенов в текст и обратно
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.id_to_char = {i: c for i, c in enumerate(vocab)}

        self.vocab_size = len(vocab)
        ids = torch.tensor([self.char_to_id[c] for c in text], device=self.device)

        # Формирование обучающей выборки
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

    def _init_weights(self, layer: nn.Module) -> None:
        """Установление весов по умолчанию"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

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
        self.loss = torch.nn.CrossEntropyLoss()
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
                "vocab_size": self.vocab_size,
                "char_to_id": self.char_to_id,
                "id_to_char": self.id_to_char,
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
        self.vocab_size = checkpoint["vocab_size"]
        self.char_to_id = checkpoint["char_to_id"]
        self.id_to_char = checkpoint["id_to_char"]

        self._init_models()
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
        self.trained = True
        print(f"Веса успешно загружены из {filepath}")
        return None

    def get_loss_plot(
            self,
            loss: list,
        ) -> None:
        """Отрисовка и сохранение графика динамики train_loss"""
        epochs = [i for i in range(1, self.num_epochs + 1)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs, loss)
        ax.axhline(loss[-1], color="red", linestyle="--")
        ax.set_xlim(1, max(epochs))

        plt.title("График динамики train_loss модели", fontweight="bold", fontsize=12)
        plt.xlabel("Эпохи", fontsize=10)
        plt.ylabel("train_loss", fontsize=10)
        plt.legend([f"train_loss", f"last_epoch_loss: {loss[-1]:.3f}"], fontsize=10)
        plt.tight_layout()
        fig.savefig(MODELS_DIR / self.model_name / "train_loss_plot.png")
        print("График динамики train_loss модели сохранен.")
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
                loss_value = self.loss(preds, y_batch)
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
        if len(prompt) < self.context_size:
            prompt = prompt.rjust(self.context_size)
                
        current_ids = [self.char_to_id[c] for c in prompt[-self.context_size:]]

        generated = []
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
            generated.append(self.id_to_char[next_id])
            print(self.id_to_char[next_id], sep="", end="")
        return "".join(generated)
