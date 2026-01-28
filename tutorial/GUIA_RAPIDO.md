# 🚀 AutoRAG - Guia Completo de Uso e Teste

> **AutoRAG** é uma ferramenta de AutoML para encontrar automaticamente o melhor pipeline RAG para seus dados.

---

## 📋 Índice

1. [Instalação](#instalação)
2. [Teste Rápido (5 minutos)](#teste-rápido-5-minutos)
3. [Teste com Modelos Locais](#teste-com-modelos-locais)
4. [Estrutura dos Dados](#estrutura-dos-dados)
5. [Configuração YAML](#configuração-yaml)
6. [Comandos CLI](#comandos-cli)
7. [Métricas Disponíveis](#métricas-disponíveis)
8. [Módulos Disponíveis](#módulos-disponíveis)
9. [Criando Seus Próprios Dados](#criando-seus-próprios-dados)
10. [Resultados de Exemplo](#resultados-de-exemplo)

---

## Instalação

### Pré-requisitos
- Python 3.10+ (3.12 recomendado)
- pip ou UV (gerenciador de pacotes rápido)

### Instalação no Windows

```powershell
# 1. Navegar para o diretório do projeto
cd C:\Users\martinianogomes\OneDrive\Documentos\Projetos\AutoRAG

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar ambiente (PowerShell)
.\.venv\Scripts\Activate.ps1

# Ou no CMD:
# .venv\Scripts\activate.bat

# Ou no Git Bash:
# source .venv/Scripts/activate

# 4. Instalar AutoRAG com todas as dependências (pode levar 15-30 min)
pip install -e ".[all]" --timeout 3600

# 5. (Opcional) Instalar suporte a modelos locais
pip install torch sentence-transformers llama-index-embeddings-huggingface
```

### Instalação no Linux

```bash
# 1. Navegar para o diretório do projeto
cd /home/martinianogomes/Documentos/Projetos/Academico/TCC/AutoRAG

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar ambiente
source .venv/bin/activate

# 4. Instalar AutoRAG básico
pip install -e ".[all]" --timeout 3600

# 5. (Opcional) Instalar suporte a modelos locais
pip install torch sentence-transformers llama-index-embeddings-huggingface
```

### Verificar Instalação

```bash
# Windows (Git Bash): source .venv/Scripts/activate
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate

autorag --help
```

---

## Teste Rápido (5 minutos)

### 1. Teste mais simples (apenas BM25, sem API keys)

```bash
cd /home/martinianogomes/Documentos/Projetos/Academico/TCC/AutoRAG
source .venv/bin/activate

# Executar avaliação com dados de exemplo
autorag evaluate \
    --config tutorial/config_simples.yaml \
    --qa_data_path tests/resources/qa_data_sample.parquet \
    --corpus_data_path tests/resources/corpus_data_sample.parquet \
    --project_dir tutorial/meu_teste
```

### 2. Ver resultados

```bash
# Via terminal
cat tutorial/meu_teste/0/retrieve_node_line/lexical_retrieval/summary.csv

# Via Dashboard (interface visual)
autorag dashboard --trial_dir tutorial/meu_teste/0 --port 7690
# Acesse: http://localhost:7690
```

---

## Teste com Modelos Locais

### Configuração para Vector DB em Memória (sem API keys)

O arquivo `tutorial/config_local.yaml` já está configurado para usar:
- **BM25**: Busca léxica
- **ChromaDB em memória**: Busca semântica com modelo local (all-mpnet-base-v2)
- **Híbrido**: Combinação de ambos (RRF e CC)

### Executar Teste Completo

```bash
cd /home/martinianogomes/Documentos/Projetos/Academico/TCC/AutoRAG
source .venv/bin/activate

# Executar avaliação com modelos locais
autorag evaluate \
    --config tutorial/config_local.yaml \
    --qa_data_path tests/resources/qa_data_sample.parquet \
    --corpus_data_path tests/resources/corpus_data_sample.parquet \
    --project_dir tutorial/projeto_local
```

### Comparar Resultados

```bash
# Ver resultados de cada método
cat tutorial/projeto_local/0/retrieve_node_line/lexical_retrieval/summary.csv
cat tutorial/projeto_local/0/retrieve_node_line/semantic_retrieval/summary.csv
cat tutorial/projeto_local/0/retrieve_node_line/hybrid_retrieval/summary.csv
```

### Abrir Dashboard

```bash
autorag dashboard --trial_dir tutorial/projeto_local/0 --port 7690
```

---

## Estrutura dos Dados

### QA Dataset (qa.parquet)

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `qid` | str | ID único da pergunta | "q_001" |
| `query` | str | A pergunta | "O que é Python?" |
| `retrieval_gt` | List[List[str]] | IDs dos documentos relevantes | [["doc_001"]] |
| `generation_gt` | List[str] | Respostas esperadas | ["Python é uma linguagem..."] |

### Corpus Dataset (corpus.parquet)

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `doc_id` | str | ID único do documento | "doc_001" |
| `contents` | str | Conteúdo textual | "Python é uma linguagem de programação..." |
| `metadata` | dict | Metadados (opcional) | {"source": "wikipedia"} |

---

## Configuração YAML

### Configuração Mínima (BM25 apenas)

```yaml
# config_simples.yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 3
        modules:
          - module_type: bm25
            bm25_tokenizer: [porter_stemmer, space]
```

### Configuração com Vector DB Local (em memória)

```yaml
# config_local.yaml
vectordb:
  - name: chroma_local
    db_type: chroma
    client_type: ephemeral  # Em memória - não persiste
    embedding_model: huggingface_all_mpnet_base_v2  # Modelo LOCAL
    collection_name: memoria_local
    embedding_batch: 50

node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      # 1. Busca Léxica
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision, retrieval_ndcg]
        top_k: 5
        modules:
          - module_type: bm25
            bm25_tokenizer: [porter_stemmer, space]

      # 2. Busca Semântica
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision, retrieval_ndcg]
        top_k: 5
        modules:
          - module_type: vectordb
            vectordb: chroma_local

      # 3. Busca Híbrida
      - node_type: hybrid_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision, retrieval_ndcg]
        top_k: 5
        modules:
          - module_type: hybrid_rrf
            weight_range: (4, 80)
          - module_type: hybrid_cc
            normalize_method: [mm, tmm, z]
            weight_range: (0.0, 1.0)
            test_weight_size: 21
```

### Configuração com OpenAI (requer API key)

```yaml
# Requer: export OPENAI_API_KEY="sk-sua-chave"
vectordb:
  - name: chroma_openai
    db_type: chroma
    client_type: ephemeral
    embedding_model: openai_embed_3_small
    collection_name: openai_collection

node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall]
        top_k: 5
        modules:
          - module_type: vectordb
            vectordb: chroma_openai
```

---

## Comandos CLI

### Comandos Principais

```bash
# Ver ajuda
autorag --help

# Validar configuração (sem executar)
autorag validate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet

# Executar avaliação completa
autorag evaluate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet \
    --project_dir ./projeto

# Pular validação (mais rápido)
autorag evaluate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet \
    --project_dir ./projeto \
    --skip_validation true
```

### Visualização e Deploy

```bash
# Dashboard interativo
autorag dashboard --trial_dir ./projeto/0 --port 7690

# API Server REST
autorag run_api --trial_dir ./projeto/0 --port 8000

# Interface Web (Streamlit)
autorag run_web --trial_path ./projeto/0

# Extrair melhor configuração
autorag extract_best_config --trial_path ./projeto/0 --output_path best_config.yaml
```

---

## Métricas Disponíveis

### Métricas de Retrieval

| Métrica | Descrição | Quando usar |
|---------|-----------|-------------|
| `retrieval_f1` | F1 Score (harmônica de precision e recall) | Métrica balanceada geral |
| `retrieval_recall` | Proporção de docs relevantes encontrados | Quando não pode perder nenhum doc |
| `retrieval_precision` | Proporção de docs retornados que são relevantes | Quando quer resultados precisos |
| `retrieval_ndcg` | Normalized Discounted Cumulative Gain | Avalia a ordem do ranking |
| `retrieval_mrr` | Mean Reciprocal Rank | Posição do primeiro doc relevante |
| `retrieval_map` | Mean Average Precision | Média da precisão em cada posição |

### Métricas de Geração

| Métrica | Descrição |
|---------|-----------|
| `bleu` | BLEU Score - correspondência de n-gramas |
| `meteor` | METEOR Score - sinônimos e stemming |
| `rouge` | ROUGE Score - sobreposição de sequências |
| `sem_score` | Similaridade semântica |
| `bert_score` | BERTScore - similaridade contextual |
| `g_eval` | Avaliação por GPT |

---

## Módulos Disponíveis

### Retrieval

| Módulo | Tipo | Descrição | Requer API? |
|--------|------|-----------|-------------|
| `bm25` | Léxico | BM25 com diferentes tokenizadores | ❌ |
| `vectordb` | Semântico | Busca por embeddings | Depende do modelo |
| `hybrid_rrf` | Híbrido | Reciprocal Rank Fusion | Depende |
| `hybrid_cc` | Híbrido | Convex Combination | Depende |

### Modelos de Embedding

| Nome | Tipo | Descrição |
|------|------|-----------|
| `huggingface_all_mpnet_base_v2` | Local | Sentence Transformers (768 dim) |
| `huggingface_baai_bge_small` | Local | BGE Small (384 dim) |
| `huggingface_bge_m3` | Local | BGE M3 multilingual |
| `openai_embed_3_small` | API | OpenAI text-embedding-3-small |
| `openai_embed_3_large` | API | OpenAI text-embedding-3-large |
| `mock` | Teste | Embeddings aleatórios (para teste) |

### Tokenizadores BM25

| Tokenizador | Descrição |
|-------------|-----------|
| `porter_stemmer` | Stemmer Porter (inglês) |
| `space` | Tokenização por espaço |
| `gpt2` | Tokenizador GPT-2 |
| `ko_kiwi` | Tokenizador coreano |

### Vector Databases

| DB | Tipo | Uso |
|----|------|-----|
| `chroma` (ephemeral) | Em memória | Testes rápidos |
| `chroma` (persistent) | Em disco | Desenvolvimento |
| `milvus` | Servidor | Produção |
| `pinecone` | Cloud | Produção escalável |
| `qdrant` | Servidor | Produção |
| `weaviate` | Servidor | Produção |

---

## Criando Seus Próprios Dados

### Exemplo Python

```python
import pandas as pd

# 1. CRIAR CORPUS (seus documentos)
corpus_data = [
    {
        "doc_id": "doc_001",
        "contents": "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
        "metadata": {"source": "wikipedia", "topic": "programming"}
    },
    {
        "doc_id": "doc_002",
        "contents": "Machine Learning é um subcampo da inteligência artificial que permite sistemas aprenderem com dados.",
        "metadata": {"source": "wikipedia", "topic": "ai"}
    },
    {
        "doc_id": "doc_003",
        "contents": "RAG (Retrieval-Augmented Generation) combina busca de documentos com geração de texto por LLMs.",
        "metadata": {"source": "paper", "topic": "nlp"}
    },
]
corpus_df = pd.DataFrame(corpus_data)
corpus_df.to_parquet("meu_corpus.parquet", index=False)

# 2. CRIAR QA DATASET (perguntas e respostas)
qa_data = [
    {
        "qid": "q_001",
        "query": "O que é Python?",
        "retrieval_gt": [["doc_001"]],  # IDs dos docs relevantes
        "generation_gt": ["Python é uma linguagem de programação de alto nível"]
    },
    {
        "qid": "q_002",
        "query": "O que é Machine Learning?",
        "retrieval_gt": [["doc_002"]],
        "generation_gt": ["Machine Learning é um subcampo da inteligência artificial"]
    },
    {
        "qid": "q_003",
        "query": "Como funciona RAG?",
        "retrieval_gt": [["doc_003"]],
        "generation_gt": ["RAG combina busca de documentos com geração de texto"]
    },
]
qa_df = pd.DataFrame(qa_data)
qa_df.to_parquet("meu_qa.parquet", index=False)

print("✅ Dados criados!")
print(f"   Corpus: {len(corpus_df)} documentos")
print(f"   QA: {len(qa_df)} perguntas")
```

### Executar com Seus Dados

```bash
autorag evaluate \
    --config tutorial/config_local.yaml \
    --qa_data_path meu_qa.parquet \
    --corpus_data_path meu_corpus.parquet \
    --project_dir meu_projeto
```

---

## Resultados de Exemplo

### Comparação: BM25 vs Semântico vs Híbrido

Resultados obtidos com os dados de exemplo (`tests/resources/`):

| Método | F1 Score | Recall | Precision | NDCG |
|--------|----------|--------|-----------|------|
| **BM25 (porter_stemmer)** | 0.333 | 100% | 20% | **1.00** |
| **VectorDB (all-mpnet)** | 0.333 | 100% | 20% | 0.94 |
| **Híbrido (CC)** | 0.333 | 100% | 20% | **1.00** |

### Interpretação

- **Recall 100%**: Todos os métodos encontram o documento relevante
- **NDCG**: BM25 e Híbrido ordenam melhor o ranking (1.0 vs 0.94)
- **Conclusão**: Para este dataset, BM25 é suficiente e mais eficiente

---

## Estrutura de Arquivos Gerados

Após executar `autorag evaluate`:

```
projeto/
├── data/
│   ├── qa.parquet           # Cópia dos dados QA
│   └── corpus.parquet       # Cópia do corpus
├── resources/
│   ├── bm25_*.pkl           # Índices BM25 
│   └── chroma/              # Vector DB (se persistente)
├── 0/                       # Trial 0
│   ├── config.yaml          # Configuração usada
│   ├── summary.csv          # Resumo geral
│   └── retrieve_node_line/
│       ├── summary.csv
│       ├── lexical_retrieval/
│       │   ├── summary.csv  # Métricas por módulo
│       │   ├── 0.parquet    # Resultados módulo 0
│       │   └── best_0.parquet # Melhor resultado
│       ├── semantic_retrieval/
│       │   └── ...
│       └── hybrid_retrieval/
│           └── ...
└── trial.json               # Metadados do trial
```

---

## Dicas e Troubleshooting

### Erros Comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `Directory does not exist` | Caminho errado | Verifique o caminho com `ls` |
| `No module named torch` | PyTorch não instalado | `uv pip install torch` |
| `OPENAI_API_KEY` | API key não configurada | `export OPENAI_API_KEY="sk-..."` |

### Dicas de Performance

1. **Comece simples**: Use BM25 primeiro para validar seus dados
2. **top_k menor**: top_k=3 geralmente dá melhor F1 que top_k=10
3. **Modelos locais**: Use `huggingface_all_mpnet_base_v2` para testes sem API
4. **Ephemeral vs Persistent**: Use `ephemeral` para testes rápidos

### Arquivos de Configuração de Exemplo

| Arquivo | Descrição |
|---------|-----------|
| `tutorial/config_simples.yaml` | Apenas BM25 |
| `tutorial/config_local.yaml` | BM25 + VectorDB local + Híbrido |
| `tutorial/config_comparacao_bm25.yaml` | Comparação de tokenizadores |
| `sample_config/rag/full.yaml` | Configuração completa (todos módulos) |

---

## Links Úteis

- **Documentação oficial**: https://marker-inc-korea.github.io/AutoRAG/
- **GitHub**: https://github.com/Marker-Inc-Korea/AutoRAG
- **Colab Tutorial**: [Step 1: Básico](https://colab.research.google.com/drive/19OEQXO_pHN6gnn2WdfPd4hjnS-4GurVd)
