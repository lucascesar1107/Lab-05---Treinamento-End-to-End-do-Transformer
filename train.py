import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformer import Transformer


def build_model(vocab_size, d_model=128, d_ff=256, num_layers=2,
                max_len=200, device=torch.device("cpu")):

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Tarefa 3] Modelo instanciado: {n_params:,} parametros treinaveis.")
    return model


def train(model, src_tensor, tgt_tensor, pad_idx,
          epochs=15, batch_size=32, lr=1e-3, device=torch.device("cpu")):

    # Funcao de perda — ignore_index ignora tokens de padding
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Otimizador Adam (mesmo do paper original)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(src_tensor, tgt_tensor),
                        batch_size=batch_size, shuffle=True)

    loss_history = []

    print("\n[Tarefa 3] Iniciando Training Loop...")
    print(f"  Epocas: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Exemplos: {len(src_tensor)}")
    print("-" * 40)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Teacher Forcing
            # dec_input  = [START, tok1, tok2]
            # dec_target = [tok1, tok2, EOS]
            dec_input  = tgt_batch[:, :-1]
            dec_target = tgt_batch[:, 1:]

            optimizer.zero_grad()
            logits = model(src_batch, dec_input)   # [B, T-1, vocab]

            B, S, V = logits.shape
            loss = criterion(logits.reshape(B * S, V),
                             dec_target.reshape(B * S))

            loss.backward()     # calcula gradientes (WQ, WK, WV...)
            optimizer.step()    # atualiza os pesos

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        loss_history.append(avg)
        print(f"  Epoca {epoch:02d}/{epochs} | Loss: {avg:.4f}")

    print("-" * 40)
    print(f"[Tarefa 3] Loss inicial: {loss_history[0]:.4f} -> Loss final: {loss_history[-1]:.4f}\n")
    return loss_history