# Lab 05 – Treinamento Fim-a-Fim do Transformer
Este projeto implementa o treinamento completo de um modelo Transformer conforme descrito no artigo científico *Attention Is All You Need*.
O objetivo é conectar a arquitetura construída nos laboratórios anteriores a um dataset real do HuggingFace e executar o Loop de Treinamento (Forward, Loss, Backward, Step), demonstrando convergência da função de perda.

As bibliotecas utilizadas foram PyTorch, HuggingFace Datasets e HuggingFace Transformers.

---

## Requisitos
- Python 3.x
- PyTorch
- Datasets
- Transformers
- Matplotlib

Instalação das dependências:
```
pip install torch datasets transformers matplotlib
```

---

## Execução
Para executar o treinamento completo, utilize:
```
python main.py
```
O script carrega o dataset, tokeniza os pares de frases, treina o modelo por 15 épocas e executa o Overfitting Test ao final.

---

## Estrutura do Projeto
```
Lab 05/
├── transformer.py   # Arquitetura Transformer completa (base do Lab 04)
├── dataset.py       # Tarefa 1 e 2: carregamento do dataset e tokenização
├── train.py         # Tarefa 3: Training Loop
├── inference.py     # Tarefa 4: Overfitting Test e gráfico de Loss
└── main.py          # Arquivo principal — execute este
```

---

## Tarefas

**Tarefa 1 – Dataset Real**  
Carrega o dataset `bentrevett/multi30k` do HuggingFace com 1000 pares de frases em inglês e alemão.

**Tarefa 2 – Tokenização Básica**  
Tokenização com `bert-base-multilingual-cased`. As frases de destino recebem os tokens especiais `<START>` e `<EOS>`, e padding é aplicado para uniformizar o comprimento do batch.

**Tarefa 3 – Training Loop**  
O modelo Transformer é treinado com `CrossEntropyLoss` (ignore_index=PAD) e otimizador `Adam` por 15 épocas. A cada época o valor do Loss é impresso.

**Tarefa 4 – Prova de Fogo**  
Após o treinamento, o modelo tenta reproduzir a tradução de uma frase que estava no conjunto de treino, utilizando o Loop Auto-regressivo (`greedy_decode`) construído no Lab 04.

---

## Resultado Obtido
```
Loss inicial: 8.8578
Loss final:   1.3638
```
A curva de convergência é salva automaticamente em `loss_curve.png`.

---

## Ferramentas de IA Utilizadas
| Tarefa | Ferramenta | Uso |
|--------|-----------|-----|
| Tarefa 1 | HuggingFace `datasets` | Carregamento do multi30k |
| Tarefa 2 | HuggingFace `transformers` | Tokenização multilingual |
| Tarefa 3 | — | Implementação própria sobre classes do Lab 04 |
| Tarefa 4 | — | Loop auto-regressivo do Lab 04 |

---

## Referência
```
Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
```
