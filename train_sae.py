import os
import argparse
import torch

from sae_model import SparseAutoencoder
from sae_dataset import ActivationShuffleBuffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", type=str, default="./activations")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--dict_size", type=int, default=2048,
                    help="Ширина словаря фич. 2-4k при d_model=768 — как в плане.")
    p.add_argument("--l1_coeff", type=float, default=3e-3,
                    help="λ — сила штрафа за неразреженность. Больше -> меньше L0, хуже реконструкция.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--n_steps", type=int, default=3000)
    p.add_argument("--dead_feature_window", type=int, default=200,
                    help="Через сколько шагов без единого срабатывания фича считается мёртвой")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def variance_explained(batch: torch.Tensor, reconstruction: torch.Tensor) -> float:
    """
    Упрощённая прокси-версия метрики "loss recovered"

    Честная версия ("сколько % CE loss модели восстановилось") требует
    заново прогонять текст через GPT-2 с подменёнными активациями —
    это дороже и по-хорошему делается один раз в конце, на отчёте
    (Блок 4), а не на каждом шаге обучения. Здесь вместо этого быстрая
    прокси: насколько ошибка реконструкции мала по сравнению с тем,
    если бы мы вообще ничего не предсказывали, кроме среднего значения.
    1.0 = идеальная реконструкция, 0.0 = как будто предсказали просто среднее.
    Явно фиксируем это упрощение в README как компромисс вечерней версии.
    """
    mse = ((batch - reconstruction) ** 2).sum(dim=-1).mean()
    baseline = ((batch - batch.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean()
    return (1 - (mse / baseline).item())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    sae = SparseAutoencoder(args.d_model, args.dict_size).to(args.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    buffer = ActivationShuffleBuffer(
        args.activations_dir, batch_size=args.batch_size, device=args.device
    )

    # Считаем, сколько шагов подряд каждая фича молчала (не сработала ни разу).
    steps_since_fired = torch.zeros(args.dict_size, device=args.device)

    print(f"Обучаю SAE: d_model={args.d_model}, dict_size={args.dict_size}, "
          f"l1_coeff={args.l1_coeff}, device={args.device}")

    for step in range(1, args.n_steps + 1):
        batch = buffer.next_batch()  # [batch_size, d_model]

        reconstruction, features = sae(batch)

        mse_loss = ((batch - reconstruction) ** 2).sum(dim=-1).mean()
        l1_loss = features.abs().sum(dim=-1).mean()
        loss = mse_loss + args.l1_coeff * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder_()

        # Обновляем счётчик "молчания" фич.
        fired_this_step = (features > 0).any(dim=0)          # [dict_size], bool
        steps_since_fired[fired_this_step] = 0
        steps_since_fired[~fired_this_step] += 1

        if step == 1 or step % args.log_every == 0:
            l0 = (features > 0).float().sum(dim=-1).mean().item()
            dead_frac = (steps_since_fired > args.dead_feature_window).float().mean().item()
            var_explained = variance_explained(batch, reconstruction)
            print(
                f"step {step:5d} | loss {loss.item():8.4f} | "
                f"L0 {l0:6.1f} | variance_explained {var_explained:5.3f} | "
                f"dead_features {dead_frac * 100:5.1f}%"
            )

    ckpt_path = os.path.join(
        args.out_dir, f"sae_dict{args.dict_size}_l1_{args.l1_coeff}.pt"
    )
    torch.save({"state_dict": sae.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Готово. Чекпоинт сохранён в {ckpt_path}")


if __name__ == "__main__":
    main()