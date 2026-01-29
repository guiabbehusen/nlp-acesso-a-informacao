# solucao_desafiodf — Detecção de PII com NER (Português) + Conversão para Binário

Este projeto implementa uma solução de **detecção de PII** usando **NER (Named Entity Recognition / Token Classification)** e uma lógica de avaliação **binária (0/1)**, conforme o edital:

- O modelo prevê entidades no texto.
- Se existir **pelo menos uma entidade válida** → `y_pred = 1`
- Se não existir entidade válida → `y_pred = 0`

A avaliação final compara `y_pred` com `y_true` no CSV e calcula **Accuracy, Precision, Recall e F1**.

---

## Datasets usados

Os arquivos ficam na pasta `data/`:

- **Treino (NER sintético)**: `dados_treino_ner_250.json`  
  Formato: lista de exemplos com `tokens` e `ner_tags` (labels BIO).

- **Validação (real, binário)**: `amostra_com_labels_1 - Página1.csv`  
  Colunas: `Texto Mascarado`, `y_true`.

---

## Resultados obtidos (validação final)

No arquivo `amostra_com_labels_1 - Página1.csv` (99 linhas), o modelo atingiu:

- **Accuracy**: `0.9899`
- **Precision**: `1.0000`
- **Recall (Sensibilidade)**: `0.9714`
- **F1**: `0.9855`

### Nota P1 (conforme edital)

O edital define a nota **P1** como:

\[
P1 = \frac{2 \cdot (Precis\~ao \times Sensibilidade)}{Precis\~ao + Sensibilidade}
\]

Substituindo os valores:

- Precisão = `1.0000`
- Sensibilidade (Recall) = `0.9714`

\[
P1 = \frac{2 \cdot (1.0000 \times 0.9714)}{1.0000 + 0.9714} = 0.9855072463768115
\]

✅ **P1 obtida:** `0.9855072463768115`

---

## Matriz de confusão (NER → Binário)

Abaixo está a matriz de confusão usada na avaliação final (0/1):

![Matriz de confusão (NER → Binário)](assets/matriz_confusao.png)

---

## Estratégia (o que foi feito na prática)

### 1) Fine-tuning de NER (Token Classification)
O “motor” é um modelo NER treinado a partir do **BERT português cased**:

- Modelo base: `neuralmind/bert-base-portuguese-cased`
- Arquitetura: `AutoModelForTokenClassification`
- Labels: esquema **BIO** (`B-...`, `I-...`, `O`)

**Por que NER?**  
NER permite capturar PII em texto livre (nomes, endereços, e-mails, telefones etc.) sem depender apenas de regex. Isso dá mais robustez para variações do mundo real.

### 2) Tokenização com alinhamento BIO (subwords)
Como o BERT tokeniza em *subwords*, o mesmo token pode virar vários pedaços. O treinamento alinha corretamente os labels:

- `is_split_into_words=True`
- `word_ids()` para mapear tokens → subwords
- o primeiro subword recebe o label original
- os demais recebem `-100` (ignora na loss)

Isso evita treinar errado por causa da tokenização.

### 3) Textos longos: **chunking + stride**
BERT trabalha com limite de ~**512 tokens** por janela. Para textos longos, a inferência é feita com **janelas sobrepostas** (stride), garantindo que o modelo “leia” tudo sem truncar.

Na decisão binária, é aplicado *early exit*: se qualquer janela gerar entidade válida, o texto vira `1`.

### 4) Conversão NER → Binário com validação
Como o objetivo final é binário (0/1), uma entidade espúria pode virar **falso positivo**. Por isso, a saída do NER passa por validações simples:

- evita aceitar “PER” em palavras genéricas (ex.: “Obrigada”, “Atenciosamente”)
- usa heurísticas para assinatura/autoidentificação (ex.: “Atenciosamente, Fulano de Tal”, “Eu, Fulano…”)
- filtra casos onde números de processo/protocolo podem confundir o modelo

Essa camada foi decisiva para reduzir erros no conjunto real.

---

## Como instalar e rodar (tutorial)

### 1) Crie o ambiente e instale dependências

Recomendado: Python 3.10+.

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Coloque os dados em `data/`

Estrutura esperada:

```
data/
  dados_treino_ner_250.json
  amostra_com_labels_1 - Página1.csv
```

### 3) Execute o notebook
Abra o Jupyter e rode as células em ordem:

```bash
jupyter lab
# ou
jupyter notebook
```

No notebook, confira se as variáveis de caminho apontam para os arquivos corretos em `data/`.

---

## Requirements

O arquivo `requirements.txt` inclui as bibliotecas usadas no treino e avaliação:

- `transformers`
- `datasets`
- `seqeval`
- `accelerate`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `tqdm`

---

## Observações finais

- A estratégia foi pensada para **competição com saída binária**, então o pós-processamento é tão importante quanto o NER.
- Se você quiser evoluir ainda mais:
  - aumente diversidade do sintético (nomes/endereços/assinaturas)
  - inclua negativos difíceis (processos SEI, números de NF, empenhos) para reduzir confusões
  - mantenha validação conservadora para evitar FPs

