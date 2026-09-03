"""
Сбор активаций residual stream одного слоя
GPT-2-small на подвыборке текстового корпуса, запись на диск чанками.
"""

import os
import argparse
import torch
from transformer_lens.model_bridge import TransformerBridge
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=6,
                    help="Номер слоя (0-11 для gpt2-small). 6 — типичный 'средний' выбор.")
    p.add_argument("--hook_point", type=str, default="resid_post",
                    choices=["resid_post", "resid_pre", "resid_mid"],
                    help="Какую точку residual stream брать")
    p.add_argument("--n_tokens", type=int, default=2_000_000,
                    help="Сколько токенов активаций собрать всего")
    p.add_argument("--chunk_tokens", type=int, default=100_000,
                    help="Размер одного файла-чанка в токенах")
    p.add_argument("--ctx_len", type=int, default=256,
                    help="Длина контекста при прогоне через модель")
    p.add_argument("--batch_size", type=int, default=16,
                    help="Батч текстов за один forward pass")
    p.add_argument("--dataset_name", type=str, default="Salesforce/wikitext",
                    help="Новые версии huggingface_hub требуют формат 'namespace/name'")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--out_dir", type=str, default="./activations")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float32", "float16", "bfloat16"],
                    help="float16 экономит диск в 2 раза, для этой задачи точности хватает с запасом")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    hook_name = f"blocks.{args.layer}.hook_{args.hook_point}"
    dtype = DTYPE_MAP[args.dtype]

    print(f"[1/3] Загружаю GPT-2-small на {args.device} (TransformerBridge)...")
    model = TransformerBridge.boot_transformers("gpt2", device=args.device)
    # boot_transformers по умолчанию отдаёт "сырые" HF-веса (без folding LayerNorm
    # и центрирования весов). enable_compatibility_mode() включает ту же обработку,
    # что раньше делал HookedTransformer.from_pretrained по умолчанию — это важно,
    # чтобы численные значения residual stream совпадали с тем, что видели в
    # большинстве SAE-туториалов/Neuronpedia, и чтобы не удивляться другим цифрам
    # при сравнении. Имена хуков вида "blocks.N.hook_resid_post" сохраняются через
    # alias-слой, так что остальной код ниже не меняется.
    model.enable_compatibility_mode()
    model.eval()
    print(f"      hook: {hook_name}, d_model={model.cfg.d_model}")

    # У GPT-2 нет отдельного padding-токена - когда тексты в батче разной
    # длины, короткие дозаполняются тем же самым <|endoftext|>, который
    # используется и как BOS. Поэтому фильтровать нужно по значению токена,
    # а не только по индексу 0 (см. предыдущий фикс - он ловил только BOS).
    eot_id = model.tokenizer.eos_token_id

    print(f"[2/3] Стримлю датасет {args.dataset_name}/{args.dataset_config}...")
    ds = load_dataset(args.dataset_name, args.dataset_config, split="train", streaming=True)
    ds_iter = iter(ds)

    print("[3/3] Собираю активации...")
    tokens_collected = 0
    chunk_idx = 0
    chunk_buffer = []
    chunk_tokens_count = 0

    def flush_chunk():
        nonlocal chunk_idx, chunk_buffer, chunk_tokens_count
        if not chunk_buffer:
            return
        acts = torch.cat(chunk_buffer, dim=0)  # [n_tokens_in_chunk, d_model]
        path = os.path.join(args.out_dir, f"acts_chunk_{chunk_idx:04d}.pt")
        torch.save(acts.to(dtype).contiguous(), path)
        print(f"      -> {path}  shape={tuple(acts.shape)}  ({acts.element_size() * acts.nelement() / 1e6:.1f} МБ)")
        chunk_idx += 1
        chunk_buffer = []
        chunk_tokens_count = 0

    while tokens_collected < args.n_tokens:
        text_batch = []
        for _ in range(args.batch_size):
            try:
                row = next(ds_iter)
            except StopIteration:
                ds_iter = iter(load_dataset(
                    args.dataset_name, args.dataset_config, split="train", streaming=True
                ))
                row = next(ds_iter)
            txt = row.get("text", "").strip()
            if len(txt) > 20:  # отсекаем пустые/служебные строки wikitext
                text_batch.append(txt)
        if not text_batch:
            continue

        tokens = model.to_tokens(text_batch, truncate=True, move_to_device=True)
        tokens = tokens[:, : args.ctx_len]

        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]  # [batch, seq, d_model]

        # <|endoftext|> встречается в трёх ролях: BOS в начале, padding в
        # конце коротких текстов (когда тексты в батче разной длины), и
        # изредка настоящий разделитель между статьями. Ни одна из этих
        # позиций не несёт содержательного смысла слова - исключаем их все
        # разом по значению токена (attention sink + Gemma Scope, см. выше).
        mask = tokens != eot_id  # [batch, seq], True = оставляем
        acts = acts[mask]  # булева индексация сама схлопывает [batch, seq] -> [n_valid_tokens]

        chunk_buffer.append(acts.cpu())
        chunk_tokens_count += acts.shape[0]
        tokens_collected += acts.shape[0]

        if chunk_tokens_count >= args.chunk_tokens:
            flush_chunk()
            print(f"      прогресс: {tokens_collected}/{args.n_tokens} токенов")

    flush_chunk()
    print(f"Готово: {tokens_collected} токенов активаций сохранено в {args.out_dir}/")


if __name__ == "__main__":
    main()