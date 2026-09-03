import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size

        # Инициализируем decoder случайными единичными векторами-столбцами.
        # "Единичными" значит длина каждого вектора = 1 — это стандартная
        # практика в SAE-литературе, чтобы штраф L1 был честно сопоставим
        # между разными фичами (иначе фича с "длинным" вектором могла бы
        # давать большую реконструкцию при том же значении кода).
        decoder_weight = torch.randn(dict_size, d_model)
        decoder_weight = decoder_weight / decoder_weight.norm(dim=1, keepdim=True)

        self.W_dec = nn.Parameter(decoder_weight)          # [dict_size, d_model]
        self.b_dec = nn.Parameter(torch.zeros(d_model))    # [d_model]

        # Encoder стартует как "зеркало" decoder (транспонированная копия) —
        # простой трюк, который на практике ускоряет первые шаги обучения.
        self.W_enc = nn.Parameter(decoder_weight.t().clone())  # [d_model, dict_size]
        self.b_enc = nn.Parameter(torch.zeros(dict_size))       # [dict_size]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Вычитаем b_dec перед encoder — устоявшийся трюк (Anthropic, SAELens):
        # так encoder видит активацию "относительно" среднего смещения, которое
        # и так восстановит decoder через свой b_dec.
        x_centered = x - self.b_dec
        pre_activation = x_centered @ self.W_enc + self.b_enc
        return torch.relu(pre_activation)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    @torch.no_grad()
    def normalize_decoder_(self):
        """
        После каждого шага возвращаем длину decoder-векторов к 1.
        Без этого модель может "жульничать": делать decoder-векторы очень
        длинными, чтобы даже маленькое (дёшево штрафуемое) значение кода
        давало большую реконструкцию.
        """
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.data /= norms