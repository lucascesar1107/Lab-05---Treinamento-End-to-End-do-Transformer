import torch
from transformers import AutoTokenizer

from transformer import Transformer
from dataset import load_translation_pairs, tokenize_pairs, build_tensors
from train import build_model, train
from inference import overfitting_test, plot_loss


# CONFIGURAÇÕES


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSET_SIZE = 1000
MAX_LEN     = 32
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 1e-3
D_MODEL     = 128
D_FF        = 256
NUM_LAYERS  = 2

print(f"Dispositivo: {DEVICE}\n")


# TAREFA 1 — Dataset


src_sentences, tgt_sentences = load_translation_pairs(SUBSET_SIZE)

PROBE_SRC = src_sentences[0]   # frase guardada para o Overfitting Test
PROBE_TGT = tgt_sentences[0]



# TAREFA 2 — Tokenização


print("[Tarefa 2] Carregando tokenizador bert-base-multilingual-cased...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

src_padded, tgt_padded, START_ID, EOS_ID, PAD_ID = tokenize_pairs(
    src_sentences, tgt_sentences, tokenizer, max_len=MAX_LEN
)

src_tensor, tgt_tensor = build_tensors(src_padded, tgt_padded)

VOCAB_SIZE = tokenizer.vocab_size
MAX_SEQ    = max(src_tensor.size(1), tgt_tensor.size(1)) + 10



# TAREFA 3 — Training Loop


model = build_model(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    max_len=MAX_SEQ,
    device=DEVICE
)

loss_history = train(
    model=model,
    src_tensor=src_tensor,
    tgt_tensor=tgt_tensor,
    pad_idx=PAD_ID,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=DEVICE
)

plot_loss(loss_history)


# TAREFA 4 — Prova de Fogo


overfitting_test(
    model=model,
    tokenizer=tokenizer,
    src_sentence=PROBE_SRC,
    tgt_sentence=PROBE_TGT,
    start_id=START_ID,
    eos_id=EOS_ID,
    pad_id=PAD_ID,
    max_len=MAX_LEN,
    device=DEVICE
)