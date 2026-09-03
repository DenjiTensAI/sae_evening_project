"""
Читает файлы-чанки, которые создал collect_activations.py, и отдаёт
перемешанные батчи для обучения SAE — не загружая все чанки в память
одновременно.
"""

import glob
import os
import random
import torch


class ActivationShuffleBuffer:
    def __init__(self, activations_dir, batch_size=4096, buffer_size=200_000, device="cpu"):
        self.paths = sorted(glob.glob(os.path.join(activations_dir, "acts_chunk_*.pt")))
        if not self.paths:
            raise FileNotFoundError(
                f"Не нашёл ни одного файла acts_chunk_*.pt в {activations_dir}. "
                f"Сначала запусти collect_activations.py."
            )
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

        # Бесконечный перемешанный "конвейер" путей к чанкам: когда файлы
        # закончились, тасуем список заново и идём по кругу. Это позволяет
        # тренироваться сколько угодно шагов, даже если чанков немного.
        self._path_iter = self._infinite_shuffled_paths()
        self.buffer = torch.empty(0, 0)
        self._refill()

    def _infinite_shuffled_paths(self):
        while True:
            paths = self.paths[:]
            random.shuffle(paths)
            for p in paths:
                yield p

    def _refill(self):
        """Дочитывает новые чанки, пока в буфере не наберётся buffer_size
        токенов, затем перемешивает буфер целиком."""
        pieces = [self.buffer] if self.buffer.numel() > 0 else []
        total = sum(p.shape[0] for p in pieces)
        while total < self.buffer_size:
            path = next(self._path_iter)
            chunk = torch.load(path).float()  # приводим float16 -> float32 для обучения
            pieces.append(chunk)
            total += chunk.shape[0]
        self.buffer = torch.cat(pieces, dim=0)
        perm = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[perm]

    def next_batch(self) -> torch.Tensor:
        if self.buffer.shape[0] < self.batch_size:
            self._refill()
        batch = self.buffer[: self.batch_size]
        self.buffer = self.buffer[self.batch_size:]
        return batch.to(self.device)