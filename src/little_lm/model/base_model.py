import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from little_lm.core.constant import MODELS_DIR
from little_lm.core.schema import BaseModelConfig
from little_lm.core.model_types import (
    PreparedData,
    ResultForwardNet,
    Weights,
)

class BaseModel():
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
        
        self.trained = False
        self.vocab_size = None
        self.char_to_id = {}
        self.id_to_char = {}
        self.gen = torch.Generator(device=self.device).manual_seed(self.random_state)

    def _prepare_data(
            self,
            data_path: str | None = None,
    ) -> None:
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
            x_train=x_train, 
            y_train=y_train,
        )
        return None

    def _init_weights(
        self,
        vocab_size: int | None = None,
        context_size: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        if vocab_size is None:
            vocab_size = self.vocab_size
        if context_size is None:
            context_size = self.context_size
        if hidden_dim is None:
            hidden_dim = self.hidden_dim

        # Инициализация весов
        W1 = torch.normal(
            0, 
            0.1, 
            (context_size * vocab_size, hidden_dim), 
            generator=self.gen, 
            device=self.device
        )
        W2 = torch.normal(
            0, 
            0.1, 
            (hidden_dim, vocab_size), 
            generator=self.gen, 
            device=self.device
        )
        self.weights = Weights(w1=W1, w2=W2)
        return None

    def _net(
        self, 
        x: torch.Tensor, 
    ) -> ResultForwardNet:
        
        # One-hot encoding
        x_one_hot = torch.eye(self.vocab_size, device=self.device)[x].reshape(len(x), -1)
        # Скрытый слой + ReLU
        hidden_states = torch.relu(x_one_hot @ self.weights.w1)
        # Логиты
        logits = hidden_states @ self.weights.w2
        # Стабильный Softmax: вычитаем max по строке во избежании overflow
        logits_shifted = logits - torch.amax(logits, dim=1, keepdims=True)
        exp_logits = torch.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(dim=1, keepdims=True)

        return ResultForwardNet(
            x_one_hot=x_one_hot, 
            hidden_states=hidden_states, 
            probs=probs
        )

    def save_weights(
        self, 
        filepath: str, 
    ) -> None:
        """Сохранение весов и метаданных модели на диск."""
        checkpoint = {
            "w1": self.weights.w1.cpu(),
            "w2": self.weights.w2.cpu(),
            "vocab_size": self.vocab_size, 
            "context_size": self.context_size,
            "hidden_dim": self.hidden_dim,
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char,
        }
        torch.save(checkpoint, filepath)
        print(f"Веса успешно сохранены в {filepath}")

    def load_weights(
        self, 
        filepath: str,
    ) -> None:
        """Загрузка сохраненных весов и токенов."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.weights = Weights(
            w1=checkpoint["w1"].to(self.device),
            w2=checkpoint["w2"].to(self.device),
        )

        # Обновляем параметры модели под сохраненные веса
        self.vocab_size = checkpoint.get("vocab_size", self.vocab_size)
        self.context_size = checkpoint.get("context_size", self.context_size)
        self.hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
        self.char_to_id = checkpoint.get("char_to_id", self.char_to_id)
        self.id_to_char = checkpoint.get("id_to_char", self.id_to_char)
        self.trained = True

        print(f"Веса успешно загружены из {filepath}")
        return None

    def get_loss_plot(
            self,
            loss: list,
    ) -> None:
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
        self._init_weights()

        ar_batch = torch.arange(self.batch_size, device=self.device)
        report_loss = []
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            epochs_loss = 0.0
            num_batches = 0

            # Случайный порядок батчей
            starts = torch.arange(
                0, 
                self.data.y_train.size(0) - self.batch_size + 1, 
                self.batch_size, 
                device=self.device
            )
            batch_indices = starts[
                torch.randperm(
                    len(starts), 
                    generator=self.gen, 
                    device=self.device,
                )
            ]

            for b in batch_indices:
                b = b.item()
                batch_X = self.data.x_train[b : b + self.batch_size]
                batch_Y = self.data.y_train[b : b + self.batch_size]

                # Прямой проход
                x_one_hot, hidden_states, probs = self._net(
                    x=batch_X,
                )

                # Считаем Cross-Entropy Loss для батча
                correct_probs = probs[ar_batch, batch_Y]
                loss = -torch.mean(torch.log(torch.clamp(correct_probs, min=1e-12)))
                epochs_loss += loss.item()
                num_batches += 1

                # Градиент выхода
                probs[ar_batch, batch_Y] -= 1.0

                # Обратное распространение ошибки
                dh = (probs @ self.weights.w2.T) * (hidden_states > 0)

                # Обновление весов с шагом learning_rate
                self.weights.w2 -= self.learning_rate * (hidden_states.T @ probs) / self.batch_size
                self.weights.w1 -= self.learning_rate * (x_one_hot.T @ dh) / self.batch_size
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
    ) -> None:
        """
        prompt: строка
        temperature: степень креативности (чем меньше, тем консервативнее выбор)
        """
        if not self.trained:
            raise RuntimeError("Модель еще не обучена")
        if len(prompt) < self.context_size:
            prompt = prompt.rjust(self.context_size)
        
        current_ids = [self.char_to_id[c] for c in prompt[-self.context_size:]]
    
        for _ in range(self.lenght_answer):
            x_in = torch.tensor([current_ids[-self.context_size:]], device=self.device)
            _, _, p = self._net(x_in)
    
            # Применение температуры к вероятностям
            temp = max(self.temperature, 1e-4)
            log_p = torch.log(torch.clamp(p[0], min=1e-12)) / temp
            probs = torch.softmax(log_p, dim=0)
    
            # Семплируем следующий символ
            next_id = torch.multinomial(
                probs, 
                num_samples=1, 
                replacement=True, 
                generator=self.gen
            ).item()
            
            current_ids.append(next_id)
            print(self.id_to_char[next_id], sep="", end="")
        return None
