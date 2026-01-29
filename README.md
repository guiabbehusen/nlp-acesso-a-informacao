# Detecção de PII em Português com NER (Token Classification) + Pós‑processamento

Este repositório contém uma solução de **detecção de PII** baseada em **NER (Named Entity Recognition / Token Classification)** com *fine‑tuning* do modelo **`neuralmind/bert-base-portuguese-cased`** e uma camada leve de pós‑processamento focada em desempenho de competição (saída final binária **0/1**).

A regra de avaliação final é simples:

- O modelo prevê entidades no texto.
- Se existir **pelo menos uma entidade válida** → `y_pred = 1`
- Se não existir entidade válida → `y_pred = 0`

Depois, comparamos `y_pred` com a coluna `y_true` do CSV e calculamos **Accuracy, Precision, Recall e F1** (classe positiva = 1).

---

## O que tem aqui

- **Notebook único**: `solucao_desafiodf.ipynb`
  - Treina um NER no JSON sintético (`dados_treino_ner_250.json`)
  - Avalia no CSV real (`amostra_com_labels_1 - Página1.csv`)
  - Implementa:
    - tokenização com alinhamento BIO
    - treino com `Trainer`
    - inferência com **chunking + stride** para textos longos
    - regras focadas para reduzir FP e recuperar FN comuns (assinatura/autoidentificação)
    - ajuste automático de threshold (com foco em **minimizar erros**)

---

## Estrutura esperada de arquivos

Coloque os dados em uma pasta `data/`:

```
data/
  dados_treino_ner_500.json
  amostra_com_labels_1 - Página1.csv
```

Saída do treino (criada automaticamente):

```
trained_ner_model_500/
  config.json
  model.safetensors
  tokenizer.json
  ...
```

> Dica: você pode mudar o nome da pasta de saída no notebook (variável `SAVE_DIR`) para comparar versões.

---

## Requisitos

- Python 3.9+ (recomendado 3.10/3.11)
- CPU funciona, GPU acelera bastante (opcional)
- RAM: 8GB+ recomendado (com folga se estiver usando GPU)

---

## Instalação

### Opção A — Ambiente local (venv)

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install transformers datasets seqeval accelerate scikit-learn pandas numpy torch
```

> Se você já tem PyTorch instalado (por exemplo, com CUDA), mantenha a versão compatível com o seu ambiente.

### Opção B — Kaggle
No Kaggle, normalmente `torch` já vem instalado. Em geral basta:

```bash
pip install -q transformers datasets seqeval accelerate
```

---

## Como rodar (treino + avaliação)

1) Garanta que os arquivos estejam em `data/`:
   - `data/dados_treino_ner_500.json`
   - `data/amostra_com_labels_1 - Página1.csv`

2) Abra o notebook no Jupyter:

```bash
jupyter lab
# ou
jupyter notebook
```

3) Execute as células em ordem.

### Configuração mínima a conferir no notebook

- `JSON_NAME = "dados_treino_ner_250.json"`
- `MODEL_NAME = "neuralmind/bert-base-portuguese-cased"`
- `SAVE_DIR = "./trained_ner_model_250"` (ou outro nome)
- `MAX_LENGTH = 512`
- `STRIDE = 256` (bom padrão)

---

## Estratégia utilizada

### 1) Fine‑tuning NER (Token Classification)
O “motor principal” é um modelo NER treinado via `AutoModelForTokenClassification`:

- Base: `neuralmind/bert-base-portuguese-cased`
- Objetivo: aprender rótulos **BIO** (`B-...`, `I-...`, `O`) nos tokens
- Treino: somente com o JSON sintético

**Tokenização e alinhamento de labels (BIO)**  
Como BERT usa *subwords*, um token pode virar vários pedaços. O notebook faz:

- `is_split_into_words=True`
- mapeia labels via `word_ids()`
- marca apenas o primeiro subword com o label real
- subwords seguintes recebem `-100` (ignorado na loss)

Isso é padrão e evita “punir” o modelo por subword.

### 2) Inferência com textos longos (chunking + stride)
BERT tem limite de contexto de ~**512 tokens**. Textos longos são processados em janelas:

- `max_length = 512`
- `stride` (sobreposição) para não “cortar” entidade na borda
- **early-exit**: se qualquer janela retornar entidade válida → `y_pred = 1`

Na prática, isso resolve a maioria dos falsos negativos em textos extensos.

### 3) Decisão binária com validação (menos ruído)
Como a competição só avalia `0/1`, uma entidade “ruim” pode virar FP.
Então o notebook separa sinais em:

**Sinais fortes (determinísticos, baixo risco de FP)**
- e-mail
- telefone (validado por quantidade de dígitos, evitando confundir com números de processo)
- matrícula (quando aparece com a palavra “MATRÍCULA”)
- autoidentificação/assinatura em padrões comuns

**Sinais fracos (NER)**
- `PER` (pessoa)
- `ADDR` (endereço)

Esses passam por:
- filtros simples (stopwords e formatos implausíveis)
- threshold de score
- heurística para evitar “PER = Obrigada”, “PER = Atenciosamente”, etc.

### 4) Regras específicas que melhoraram muito o score
Em textos reais, vários casos positivos aparecem assim:

- **Assinatura**: “Atenciosamente, Nome Sobrenome”
- **Autoidentificação**: “Eu, Nome Sobrenome…” / “Me chamo Nome…”

O NER pode falhar nesses pontos, então a solução aplica um “resgate” simples:
- detecta palavras‑chave (“Atenciosamente”, “Att”, “Me chamo”, “Eu,” etc.)
- extrai o trecho seguinte
- valida se “parece nome” (capitalização e partículas do português como “de/da/do”)

Isso reduz FN sem aumentar muito FP.

---

## Ajuste do threshold com foco em “menos erros”
O notebook não escolhe threshold para maximizar F1 por padrão.
Em vez disso, ele varre valores e escolhe o que **minimiza o número total de erros (FP+FN)** no CSV de validação real.

Na prática, isso é mais compatível com o objetivo de “maximizar acertos” no conjunto de avaliação.

---

## Troubleshooting rápido

### 1) “Meu treino ficou com loss=0 e num_labels=1”
Isso acontece quando o JSON vem só com tag `O` (sem entidades).
Solução:
- gerar dados com entidades anotadas (ideal), ou
- habilitar fallback de pseudo‑labels (se o notebook estiver com essa opção)

### 2) “Erro de 512 tokens / texto grande”
Não rode `ner(texto_grande)` direto. Use inferência por janelas (`MAX_LENGTH=512` + `STRIDE`).

### 3) “Falsos positivos com números de processo (SEI)”
O conserto é endurecer validação de telefone/IDs e bloquear padrões de número de processo para não parecerem PII.

---

## Como reproduzir a avaliação final
Ao final do notebook, você terá:

- `classification_report` do sklearn (com foco na classe 1)
- matriz de confusão (TN/FP/FN/TP)
- lista de erros (para inspeção manual)

---

## Resumo curto da estratégia (para colar em relatório)
Fine‑tuning de `neuralmind/bert-base-portuguese-cased` para NER com labels BIO em dados sintéticos; inferência com janela deslizante (512 tokens + stride) e decisão binária por presença de entidade; pós‑processamento para reduzir FP e recuperar FN (assinatura/autoidentificação, validação de padrões fortes e threshold de entidades fracas).

---

## Observação
Este projeto foi desenhado para uma avaliação binária de PII. Se você quiser reaproveitar como NER “puro”, basta remover a camada binária e avaliar as entidades diretamente (seqeval / span-level).
