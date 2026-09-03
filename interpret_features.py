"""
Для каждой выбранной фичи находим токены (в свежем, не участвовавшем
в обучении тексте), на которых эта фича активируется сильнее всего,
и показываем текстовый контекст вокруг них — чтобы прочитать глазами
и придумать короткую гипотезу вида "фича реагирует на X".
"""

import argparse
import random
import torch

from transformer_lens.model_bridge import TransformerBridge
from datasets import load_dataset

from sae_model import SparseAutoencoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--hook_point", type=str, default="resid_post")
    p.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--n_texts", type=int, default=300,
                    help="Сколько свежих текстов прогнать в поисках примеров")
    p.add_argument("--ctx_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_features", type=int, default=8,
                    help="Сколько всего фич интерпретировать")
    p.add_argument("--n_random_features", type=int, default=3,
                    help="Из них — случайных живых фич (остальные — самые частые)")
    p.add_argument("--top_k", type=int, default=8,
                    help="Сколько top-примеров показывать на фичу")
    p.add_argument("--context_window", type=int, default=10,
                    help="Сколько токенов слева/справа показывать вокруг найденного")
    p.add_argument("--out_file", type=str, default="./feature_report.md")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_sae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    sae_args = ckpt["args"]
    sae = SparseAutoencoder(sae_args["d_model"], sae_args["dict_size"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, sae_args


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Загружаю обученный SAE...")
    sae, sae_args = load_sae(args.checkpoint, args.device)
    dict_size = sae_args["dict_size"]
    hook_name = f"blocks.{args.layer}.hook_{args.hook_point}"

    print("Загружаю GPT-2-small...")
    model = TransformerBridge.boot_transformers("gpt2", device=args.device)
    model.enable_compatibility_mode()
    model.eval()
    eot_id = model.tokenizer.eos_token_id  # тот же <|endoftext|>: BOS и padding

    print(f"Стримлю {args.n_texts} свежих текстов (split=validation, "
          f"SAE их не видел при обучении)...")
    ds = load_dataset(args.dataset_name, args.dataset_config, split="validation", streaming=True)
    ds_iter = iter(ds)

    # Для каждого текста храним пару (токены, значения_фич_на_каждый_токен).
    # float16 для экономии памяти — те же соображения, что и в Блоке 1.
    seqs_features = []
    activation_counts = torch.zeros(dict_size)
    seen_texts = 0

    while seen_texts < args.n_texts:
        text_batch = []
        for _ in range(args.batch_size):
            try:
                row = next(ds_iter)
            except StopIteration:
                break
            txt = row.get("text", "").strip()
            if len(txt) > 20:
                text_batch.append(txt)
        if not text_batch:
            break

        tokens = model.to_tokens(text_batch, truncate=True, move_to_device=True)
        tokens = tokens[:, : args.ctx_len]

        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]  # [batch, seq, d_model]

        flat_features = sae.encode(acts.reshape(-1, acts.shape[-1]))
        features = flat_features.reshape(acts.shape[0], acts.shape[1], dict_size)

        valid_mask = (tokens != eot_id)  # [batch, seq]
        activation_counts += ((features > 0) & valid_mask.unsqueeze(-1)).float().sum(dim=(0, 1)).cpu()

        for b in range(tokens.shape[0]):
            seqs_features.append((tokens[b].cpu(), features[b].to(torch.float16).cpu()))
        seen_texts += tokens.shape[0]

    total_tokens = sum(int((t != eot_id).sum()) for t, _ in seqs_features)
    print(f"Собрал активации фич на {seen_texts} текстах ({total_tokens} токенов, без BOS/padding).")

    # --- Выбор фич для интерпретации ---
    # "Живая" фича — та, что хотя бы раз сработала в нашей выборке.
    active_feature_ids = (activation_counts > 0).nonzero(as_tuple=True)[0].tolist()
    if not active_feature_ids:
        print("Не нашлось ни одной живой фичи в этой выборке — либо "
              "SAE недообучен, либо текстов слишком мало. Попробуй "
              "увеличить --n_texts.")
        return

    n_top = max(args.n_features - args.n_random_features, 0)
    most_frequent = sorted(active_feature_ids, key=lambda f: -activation_counts[f])[:n_top]
    remaining = [f for f in active_feature_ids if f not in most_frequent]
    random_selection = random.sample(remaining, min(args.n_random_features, len(remaining)))
    selected_features = most_frequent + random_selection
    print(f"Выбраны фичи: {selected_features} "
          f"(первые {len(most_frequent)} — самые частые, остальные — случайные)")

    # --- Поиск top-k примеров и запись отчёта ---
    report_lines = ["# Отчёт по интерпретации фич SAE\n"]

    for feat in selected_features:
        candidates = []  # (значение_активации, индекс_текста, позиция_токена)
        for seq_idx, (tok_ids, feats) in enumerate(seqs_features):
            values = feats[:, feat].float()
            nonzero_positions = (values > 0).nonzero(as_tuple=True)[0]
            for pos in nonzero_positions.tolist():
                if tok_ids[pos].item() == eot_id:
                    continue  # BOS или padding - не содержательный токен
                candidates.append((values[pos].item(), seq_idx, pos))

        candidates.sort(key=lambda c: -c[0])
        top_candidates = candidates[: args.top_k]

        report_lines.append(f"\n## Фича #{feat}\n")
        report_lines.append(
            f"Сработала на {int(activation_counts[feat])} токенах из {total_tokens} "
            f"просмотренных.\n"
        )
        if not top_candidates:
            report_lines.append("_Нет примеров с положительной активацией._\n")
            continue

        for val, seq_idx, pos in top_candidates:
            tok_ids, _ = seqs_features[seq_idx]
            lo = max(0, pos - args.context_window)
            hi = min(len(tok_ids), pos + args.context_window + 1)

            prefix = model.to_string(tok_ids[lo:pos].tolist()) if pos > lo else ""
            token_str = model.to_string([tok_ids[pos].item()])
            suffix = model.to_string(tok_ids[pos + 1: hi].tolist()) if pos + 1 < hi else ""

            clean = lambda s: s.replace("\n", " ")
            report_lines.append(
                f"- (act={val:.2f}) ...{clean(prefix)}**[{clean(token_str)}]**{clean(suffix)}...\n"
            )

    with open(args.out_file, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\nГотово. Отчёт сохранён в {args.out_file}.")
    print("Открой файл, прочитай примеры для каждой фичи и придумай короткую гипотезу.")


if __name__ == "__main__":
    main()