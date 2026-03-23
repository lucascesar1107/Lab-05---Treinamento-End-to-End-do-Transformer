import torch
from torch.utils.data import TensorDataset


# TAREFA 1 — Dataset Real (HuggingFace)


def load_translation_pairs(subset_size=1000):
    from datasets import load_dataset

    print(f"[Tarefa 1] Carregando dataset multi30k ({subset_size} pares)...")
    dataset = load_dataset("bentrevett/multi30k", split="train", trust_remote_code=True)
    dataset = dataset.select(range(min(subset_size, len(dataset))))

    src = [item["en"] for item in dataset]
    tgt = [item["de"] for item in dataset]

    print(f"[Tarefa 1] {len(src)} pares carregados.")
    print(f"  Ex. EN: {src[0]}")
    print(f"  Ex. DE: {tgt[0]}\n")
    return src, tgt



# TAREFA 2 — Tokenização Básica


def build_tokenizer(model_name="bert-base-multilingual-cased"):
    from transformers import AutoTokenizer

    print(f"[Tarefa 2] Carregando tokenizador: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[Tarefa 2] Vocabulario: {tokenizer.vocab_size:,} tokens\n")
    return tokenizer


def tokenize_pairs(src_sentences, tgt_sentences, tokenizer, max_len=32):
    START = tokenizer.cls_token_id   # <START>
    EOS   = tokenizer.sep_token_id   # <EOS>
    PAD   = tokenizer.pad_token_id   # <PAD>

    print("[Tarefa 2] Tokenizando pares de frases...")

    src_ids_list = []
    tgt_ids_list = []

    for src, tgt in zip(src_sentences, tgt_sentences):
        src_ids = tokenizer.encode(src, add_special_tokens=False)[:max_len]
        tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)[:max_len - 2]
        src_ids_list.append(src_ids)
        tgt_ids_list.append([START] + tgt_ids + [EOS])  # <START> tokens <EOS>

    src_max = max(len(s) for s in src_ids_list)
    tgt_max = max(len(t) for t in tgt_ids_list)

    src_padded = [s + [PAD] * (src_max - len(s)) for s in src_ids_list]
    tgt_padded = [t + [PAD] * (tgt_max - len(t)) for t in tgt_ids_list]

    print(f"[Tarefa 2] Tokenizacao concluida.")
    print(f"  Comprimento src: {src_max} | tgt: {tgt_max}")
    print(f"  START={START} | EOS={EOS} | PAD={PAD}\n")

    return src_padded, tgt_padded, START, EOS, PAD


def build_tensors(src_padded, tgt_padded):
    src_tensor = torch.tensor(src_padded, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)
    return src_tensor, tgt_tensor


def build_dataset(src_tensor, tgt_tensor):
    return TensorDataset(src_tensor, tgt_tensor)