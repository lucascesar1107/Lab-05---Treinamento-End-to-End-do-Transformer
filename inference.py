import torch
from transformer import greedy_decode


def overfitting_test(model, tokenizer, src_sentence, tgt_sentence,
                     start_id, eos_id, pad_id, max_len=32,
                     device=torch.device("cpu")):

    print("=" * 50)
    print("TAREFA 4 — PROVA DE FOGO (OVERFITTING TEST)")
    print("=" * 50)
    print(f"  Entrada (EN): {src_sentence}")
    print(f"  Esperado (DE): {tgt_sentence}")

    src_ids   = tokenizer.encode(src_sentence, add_special_tokens=False)[:max_len]
    enc_input = torch.tensor([src_ids], dtype=torch.long, device=device)

    generated_ids = greedy_decode(model, enc_input, start_id, eos_id, max_steps=max_len)

    clean_ids      = [t for t in generated_ids if t not in (start_id, eos_id, pad_id)]
    generated_text = tokenizer.decode(clean_ids, skip_special_tokens=True)

    print(f"  Gerado pelo modelo: {generated_text}")

    if tgt_sentence.strip().lower() == generated_text.strip().lower():
        print("  RESULTADO: PASSOU — traducao exata memorizada!")
    elif all(w in generated_text.lower() for w in tgt_sentence.lower().split()):
        print("  RESULTADO: PASSOU (parcial) — todas as palavras presentes.")
    else:
        print("  RESULTADO: ATENCAO — saida difere. Verifique se o Loss caiu.")

    print("=" * 50)


def plot_loss(loss_history):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
        plt.xlabel("Epoca")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Convergencia do Transformer — Lab 05")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=120)
        plt.show()
        print("\n[Grafico] loss_curve.png salvo.")
    except ImportError:
        print("Historico de loss:", [f"{v:.4f}" for v in loss_history])