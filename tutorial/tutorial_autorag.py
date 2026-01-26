#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        🚀 TUTORIAL COMPLETO DO AUTORAG                        ║
║                                                                               ║
║   AutoRAG: Ferramenta AutoML para encontrar o melhor pipeline RAG             ║
║   Repositório: https://github.com/Marker-Inc-Korea/AutoRAG                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Este script é um tutorial interativo e completo do AutoRAG que demonstra:

1. 📊 ESTRUTURA DOS DADOS
   - Formato dos arquivos QA e Corpus (parquet)
   - Campos obrigatórios e opcionais
   - Exemplos de dados de entrada

2. 📄 CONFIGURAÇÕES YAML
   - Explicação detalhada de cada arquivo de configuração
   - Exemplos comentados de YAML
   - VectorDB, nodes, modules e estratégias

3. 🖥️ COMANDOS CLI
   - Todos os comandos disponíveis no AutoRAG CLI
   - Opções e parâmetros de cada comando
   - Exemplos práticos de uso

4. 🚀 EXECUÇÃO DE AVALIAÇÕES
   - Via Python programaticamente
   - Via CLI (linha de comando)
   - Configurações simples e avançadas

5. 📈 ANÁLISE DE RESULTADOS
   - Leitura e interpretação das métricas
   - Comparação entre diferentes métodos
   - Identificação do melhor pipeline

6. 🌐 API E DASHBOARD
   - Servidor REST API
   - Dashboard interativo
   - Interface Web Streamlit

7. 📝 CRIAÇÃO DE DADOS
   - Como criar seu próprio dataset
   - Formato correto dos arquivos
   - Boas práticas

8. 🔧 MÓDULOS E MÉTRICAS
   - Lista completa de módulos disponíveis
   - Métricas de retrieval e geração
   - Embedding models e tokenizadores

═══════════════════════════════════════════════════════════════════════════════

ARQUIVOS DE CONFIGURAÇÃO INCLUÍDOS:
- tutorial/config_simples.yaml       → Apenas BM25 (sem API keys)
- tutorial/config_local.yaml         → BM25 + VectorDB local + Híbrido
- tutorial/config_comparacao_bm25.yaml → Comparação de tokenizadores
- tutorial/config_memoria_completo.yaml → Com OpenAI (requer API key)

═══════════════════════════════════════════════════════════════════════════════

INSTALAÇÃO E REQUISITOS:

# 1. Instalação básica
pip install -e .

# 2. Para modelos locais (sem API keys)
pip install torch sentence-transformers llama-index-embeddings-huggingface

# 3. Para OpenAI (opcional)
export OPENAI_API_KEY="sk-sua-chave"

═══════════════════════════════════════════════════════════════════════════════

USO DESTE SCRIPT:

    # Tutorial informativo (mostra tudo)
    python tutorial_autorag.py

    # Execuções
    python tutorial_autorag.py --run-simple      # Apenas BM25
    python tutorial_autorag.py --run-local       # BM25 + VectorDB + Híbrido
    python tutorial_autorag.py --run-bm25-compare # Comparação de tokenizadores
    python tutorial_autorag.py --run-all         # Todas as avaliações

    # Análise
    python tutorial_autorag.py --analyze         # Analisa resultados
    python tutorial_autorag.py --compare         # Compara métodos

    # Documentação
    python tutorial_autorag.py --cli-help        # Todos os comandos CLI
    python tutorial_autorag.py --metrics         # Lista métricas
    python tutorial_autorag.py --modules         # Lista módulos
    python tutorial_autorag.py --configs         # Mostra configs YAML
    python tutorial_autorag.py --create-data     # Exemplo de criação de dados

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import glob
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Adicionar o diretório raiz ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# CAMINHOS E CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

# Caminhos dos dados de teste
QA_PATH = os.path.join(ROOT_DIR, "tests/resources/qa_data_sample.parquet")
CORPUS_PATH = os.path.join(ROOT_DIR, "tests/resources/corpus_data_sample.parquet")

# Diretório do tutorial
TUTORIAL_DIR = os.path.join(ROOT_DIR, "tutorial")

# Arquivos de configuração do tutorial
CONFIG_FILES = {
    "simples": os.path.join(TUTORIAL_DIR, "config_simples.yaml"),
    "local": os.path.join(TUTORIAL_DIR, "config_local.yaml"),
    "bm25_compare": os.path.join(TUTORIAL_DIR, "config_comparacao_bm25.yaml"),
    "openai": os.path.join(TUTORIAL_DIR, "config_memoria_completo.yaml"),
}

# Diretórios de saída
PROJECT_DIRS = {
    "simples": os.path.join(TUTORIAL_DIR, "projeto_simples"),
    "local": os.path.join(TUTORIAL_DIR, "projeto_local"),
    "bm25_compare": os.path.join(TUTORIAL_DIR, "projeto_bm25_compare"),
    "openai": os.path.join(TUTORIAL_DIR, "projeto_openai"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 1: ESTRUTURA DOS DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def explorar_dados():
    """
    Explora e documenta a estrutura dos dados de entrada do AutoRAG.
    
    O AutoRAG requer dois arquivos no formato Parquet:
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1. QA DATASET (qa.parquet) - Perguntas e respostas para avaliação          │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ Colunas:                                                                    │
    │   • qid           : str        - ID único da pergunta                       │
    │   • query         : str        - Texto da pergunta                          │
    │   • retrieval_gt  : List[List] - Ground truth: IDs dos docs relevantes      │
    │   • generation_gt : List[str]  - Ground truth: respostas esperadas          │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 2. CORPUS DATASET (corpus.parquet) - Base de documentos para busca         │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ Colunas:                                                                    │
    │   • doc_id   : str  - ID único do documento                                 │
    │   • contents : str  - Conteúdo textual do documento                         │
    │   • metadata : dict - Metadados opcionais (fonte, data, etc.)               │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
    try:
        import pandas as pd
    except ImportError:
        print("❌ Pandas não instalado. Execute: pip install pandas")
        return None, None
    
    print("=" * 80)
    print("📊 PARTE 1: ESTRUTURA DOS DADOS DO AUTORAG")
    print("=" * 80)
    
    # Verificar se os arquivos existem
    if not os.path.exists(QA_PATH):
        print(f"❌ Arquivo QA não encontrado: {QA_PATH}")
        return None, None
    
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Arquivo Corpus não encontrado: {CORPUS_PATH}")
        return None, None
    
    qa_df = pd.read_parquet(QA_PATH)
    corpus_df = pd.read_parquet(CORPUS_PATH)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 📁 QA DATASET - Perguntas e Ground Truth                                    │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    print(f"   📂 Arquivo: {QA_PATH}")
    print(f"   📊 Linhas: {len(qa_df)}")
    print(f"   📋 Colunas: {list(qa_df.columns)}")
    print(f"\n   📝 Tipos de dados:")
    for col in qa_df.columns:
        print(f"      • {col}: {qa_df[col].dtype}")
    
    print(f"\n   🔍 Exemplo (primeira linha):")
    print(f"      • qid: {qa_df['qid'].iloc[0]}")
    print(f"      • query: {qa_df['query'].iloc[0][:60]}...")
    print(f"      • retrieval_gt: {qa_df['retrieval_gt'].iloc[0]}")
    if 'generation_gt' in qa_df.columns:
        gen_gt = qa_df['generation_gt'].iloc[0]
        if gen_gt:
            print(f"      • generation_gt: {str(gen_gt)[:60]}...")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 📁 CORPUS DATASET - Base de Documentos                                      │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    print(f"   📂 Arquivo: {CORPUS_PATH}")
    print(f"   📊 Linhas: {len(corpus_df)}")
    print(f"   📋 Colunas: {list(corpus_df.columns)}")
    print(f"\n   📝 Tipos de dados:")
    for col in corpus_df.columns:
        print(f"      • {col}: {corpus_df[col].dtype}")
    
    print(f"\n   🔍 Exemplo (primeira linha):")
    print(f"      • doc_id: {corpus_df['doc_id'].iloc[0]}")
    print(f"      • contents: {corpus_df['contents'].iloc[0][:100]}...")
    if 'metadata' in corpus_df.columns:
        print(f"      • metadata: {corpus_df['metadata'].iloc[0]}")
    
    return qa_df, corpus_df


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 2: CONFIGURAÇÕES YAML
# ═══════════════════════════════════════════════════════════════════════════════

def listar_configuracoes():
    """
    Lista e explica todas as configurações YAML disponíveis no tutorial.
    """
    print("\n" + "=" * 80)
    print("📄 PARTE 2: ARQUIVOS DE CONFIGURAÇÃO YAML")
    print("=" * 80)
    
    configs_info = {
        "config_simples.yaml": {
            "descricao": "Configuração mínima apenas com BM25",
            "api_keys": "❌ Não requer",
            "modulos": ["BM25 com porter_stemmer e space"],
            "metricas": ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
            "uso": "Testes rápidos e validação de dados",
            "tempo": "~30 segundos"
        },
        "config_local.yaml": {
            "descricao": "Comparação completa com modelos LOCAIS",
            "api_keys": "❌ Não requer (usa sentence-transformers)",
            "modulos": [
                "BM25 (léxico)",
                "ChromaDB ephemeral + all-mpnet-base-v2 (semântico)",
                "Híbrido RRF e CC"
            ],
            "metricas": ["retrieval_f1", "retrieval_recall", "retrieval_precision", "retrieval_ndcg", "retrieval_mrr"],
            "uso": "Comparação BM25 vs Semântico vs Híbrido",
            "tempo": "~2-5 minutos (inclui download do modelo)"
        },
        "config_comparacao_bm25.yaml": {
            "descricao": "Comparação de tokenizadores BM25",
            "api_keys": "❌ Não requer",
            "modulos": ["BM25 com porter_stemmer, space, gpt2"],
            "metricas": ["retrieval_f1", "retrieval_recall", "retrieval_precision", "retrieval_ndcg", "retrieval_mrr"],
            "uso": "Otimização de parâmetros BM25",
            "tempo": "~1-2 minutos"
        },
        "config_memoria_completo.yaml": {
            "descricao": "Configuração completa com OpenAI",
            "api_keys": "✅ Requer OPENAI_API_KEY",
            "modulos": [
                "BM25 (léxico)",
                "ChromaDB + OpenAI embed-3-small (semântico)",
                "Híbrido RRF e CC"
            ],
            "metricas": ["retrieval_f1", "retrieval_recall", "retrieval_precision", "retrieval_ndcg", "retrieval_mrr", "retrieval_map"],
            "uso": "Avaliação com embeddings de alta qualidade",
            "tempo": "~3-5 minutos"
        }
    }
    
    for config_name, info in configs_info.items():
        config_path = os.path.join(TUTORIAL_DIR, config_name)
        exists = "✅" if os.path.exists(config_path) else "❌"
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ {exists} {config_name:<70} │
└─────────────────────────────────────────────────────────────────────────────┘
   📝 Descrição: {info['descricao']}
   🔑 API Keys: {info['api_keys']}
   ⏱️  Tempo estimado: {info['tempo']}
   
   🔧 Módulos:""")
        for mod in info['modulos']:
            print(f"      • {mod}")
        print(f"\n   📊 Métricas: {', '.join(info['metricas'])}")
        print(f"   💡 Uso recomendado: {info['uso']}")


def mostrar_conteudo_configs():
    """
    Mostra o conteúdo real dos arquivos de configuração YAML.
    """
    print("\n" + "=" * 80)
    print("📝 CONTEÚDO DOS ARQUIVOS DE CONFIGURAÇÃO")
    print("=" * 80)
    
    for name, path in CONFIG_FILES.items():
        if os.path.exists(path):
            print(f"\n{'─'*80}")
            print(f"📄 {os.path.basename(path)}")
            print(f"   Caminho: {path}")
            print("─" * 80)
            with open(path, 'r') as f:
                content = f.read()
            print(content)
        else:
            print(f"\n❌ {name}: Arquivo não encontrado em {path}")


def mostrar_exemplo_yaml_completo():
    """
    Mostra exemplos detalhados de configuração YAML com explicações.
    """
    print("\n" + "=" * 80)
    print("📖 GUIA COMPLETO DE CONFIGURAÇÃO YAML")
    print("=" * 80)
    
    guia = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        ESTRUTURA DO ARQUIVO YAML                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Um arquivo de configuração YAML do AutoRAG tem duas seções principais:

1. vectordb (opcional): Define as bases de dados vetoriais
2. node_lines (obrigatório): Define os pipelines de processamento

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 SEÇÃO: vectordb
──────────────────
Define configurações de Vector Databases para busca semântica.

```yaml
vectordb:
  - name: meu_vectordb              # Nome único para referenciar
    db_type: chroma                 # Tipo: chroma, milvus, pinecone, qdrant, weaviate
    client_type: ephemeral          # ephemeral (memória) ou persistent (disco)
    embedding_model: huggingface_all_mpnet_base_v2  # Modelo de embedding
    collection_name: minha_colecao  # Nome da coleção
    embedding_batch: 50             # Tamanho do batch para embeddings
    
    # Para persistent, adicione:
    # path: ${PROJECT_DIR}/resources/chroma
```

MODELOS DE EMBEDDING DISPONÍVEIS:
┌────────────────────────────────────┬─────────┬──────────────────────────────┐
│ Nome                               │ Tipo    │ Dimensões                    │
├────────────────────────────────────┼─────────┼──────────────────────────────┤
│ huggingface_all_mpnet_base_v2      │ Local   │ 768                          │
│ huggingface_baai_bge_small         │ Local   │ 384                          │
│ huggingface_bge_m3                 │ Local   │ 1024 (multilingual)          │
│ openai_embed_3_small               │ API     │ 1536                         │
│ openai_embed_3_large               │ API     │ 3072                         │
│ mock                               │ Teste   │ 768 (aleatório)              │
└────────────────────────────────────┴─────────┴──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 SEÇÃO: node_lines
────────────────────
Define pipelines de processamento sequencial.

```yaml
node_lines:
  - node_line_name: retrieve_node_line    # Nome do pipeline
    nodes:                                 # Lista de nodes (etapas)
      - node_type: lexical_retrieval       # Tipo do node
        strategy:                          # Configuração da estratégia
          metrics: [retrieval_f1, retrieval_recall]  # Métricas para avaliar
          speed_threshold: 10              # Threshold de velocidade (opcional)
        top_k: 5                           # Número de resultados
        modules:                           # Módulos a testar
          - module_type: bm25
            bm25_tokenizer: [porter_stemmer, space]
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 TIPOS DE NODE DISPONÍVEIS:
────────────────────────────

RETRIEVAL (Busca de Documentos):
┌────────────────────────┬───────────────────────────────────────────────────────┐
│ Node Type              │ Descrição                                             │
├────────────────────────┼───────────────────────────────────────────────────────┤
│ lexical_retrieval      │ Busca léxica (BM25, TF-IDF)                          │
│ semantic_retrieval     │ Busca semântica (Vector DB)                          │
│ hybrid_retrieval       │ Combina léxico + semântico                           │
└────────────────────────┴───────────────────────────────────────────────────────┘

PRÉ-PROCESSAMENTO:
┌────────────────────────┬───────────────────────────────────────────────────────┐
│ Node Type              │ Descrição                                             │
├────────────────────────┼───────────────────────────────────────────────────────┤
│ query_expansion        │ Expande/reformula queries                            │
└────────────────────────┴───────────────────────────────────────────────────────┘

PÓS-PROCESSAMENTO:
┌────────────────────────┬───────────────────────────────────────────────────────┐
│ Node Type              │ Descrição                                             │
├────────────────────────┼───────────────────────────────────────────────────────┤
│ passage_augmenter      │ Aumenta contexto dos passages                        │
│ passage_reranker       │ Reordena resultados por relevância                   │
│ passage_filter         │ Filtra resultados por threshold                      │
│ passage_compressor     │ Comprime/sumariza passages                           │
└────────────────────────┴───────────────────────────────────────────────────────┘

GERAÇÃO:
┌────────────────────────┬───────────────────────────────────────────────────────┐
│ Node Type              │ Descrição                                             │
├────────────────────────┼───────────────────────────────────────────────────────┤
│ prompt_maker           │ Cria prompts para geração                            │
│ generator              │ Gera respostas com LLM                               │
└────────────────────────┴───────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 EXEMPLO COMPLETO - PIPELINE AVANÇADO:
───────────────────────────────────────

```yaml
vectordb:
  - name: chroma_local
    db_type: chroma
    client_type: ephemeral
    embedding_model: huggingface_all_mpnet_base_v2
    collection_name: local_collection

node_lines:
  # Pipeline 1: Retrieval
  - node_line_name: retrieve_node_line
    nodes:
      # Etapa 1: Busca Léxica
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision, retrieval_ndcg]
        top_k: 10
        modules:
          - module_type: bm25
            bm25_tokenizer: [porter_stemmer, space]
      
      # Etapa 2: Busca Semântica
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision, retrieval_ndcg]
        top_k: 10
        modules:
          - module_type: vectordb
            vectordb: chroma_local
      
      # Etapa 3: Busca Híbrida
      - node_type: hybrid_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision, retrieval_ndcg]
        top_k: 10
        modules:
          - module_type: hybrid_rrf
            weight_range: (4, 80)
          - module_type: hybrid_cc
            normalize_method: [mm, tmm, z]
            weight_range: (0.0, 1.0)
      
      # Etapa 4: Reranking
      - node_type: passage_reranker
        strategy:
          metrics: [retrieval_f1, retrieval_recall]
        top_k: 5
        modules:
          - module_type: pass_reranker
          - module_type: sentence_transformer_reranker
```
"""
    print(guia)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 3: COMANDOS CLI
# ═══════════════════════════════════════════════════════════════════════════════

def mostrar_comandos_cli():
    """
    Documenta todos os comandos CLI disponíveis no AutoRAG.
    """
    print("\n" + "=" * 80)
    print("🖥️  PARTE 3: COMANDOS CLI DO AUTORAG")
    print("=" * 80)
    
    cli_docs = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        TODOS OS COMANDOS AUTORAG CLI                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. autorag evaluate - Executar avaliação de pipeline RAG                    │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Executa a avaliação de um pipeline RAG com base na configuração YAML.
    Testa todas as combinações de módulos e salva os resultados.

  SINTAXE:
    autorag evaluate [OPTIONS]

  OPÇÕES:
    --config, -c PATH        Caminho para o arquivo YAML de configuração
    --qa_data_path PATH      Caminho para o arquivo QA (parquet)
    --corpus_data_path PATH  Caminho para o arquivo Corpus (parquet)
    --project_dir PATH       Diretório para salvar resultados (padrão: ./project)
    --skip_validation BOOL   Pular validação (padrão: False)

  EXEMPLOS:
    # Avaliação simples
    autorag evaluate \\
        --config tutorial/config_simples.yaml \\
        --qa_data_path tests/resources/qa_data_sample.parquet \\
        --corpus_data_path tests/resources/corpus_data_sample.parquet \\
        --project_dir tutorial/projeto_simples

    # Avaliação com modelos locais
    autorag evaluate \\
        --config tutorial/config_local.yaml \\
        --qa_data_path tests/resources/qa_data_sample.parquet \\
        --corpus_data_path tests/resources/corpus_data_sample.parquet \\
        --project_dir tutorial/projeto_local

    # Comparação de tokenizadores BM25
    autorag evaluate \\
        --config tutorial/config_comparacao_bm25.yaml \\
        --qa_data_path tests/resources/qa_data_sample.parquet \\
        --corpus_data_path tests/resources/corpus_data_sample.parquet \\
        --project_dir tutorial/projeto_bm25_compare

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. autorag validate - Validar configuração YAML                             │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Valida se um arquivo de configuração YAML está correto antes de executar.
    Verifica a estrutura, módulos e parâmetros.

  SINTAXE:
    autorag validate [OPTIONS]

  OPÇÕES:
    --config, -c PATH        Caminho para o arquivo YAML
    --qa_data_path PATH      Caminho para o arquivo QA (parquet)
    --corpus_data_path PATH  Caminho para o arquivo Corpus (parquet)

  EXEMPLO:
    autorag validate \\
        --config tutorial/config_local.yaml \\
        --qa_data_path tests/resources/qa_data_sample.parquet \\
        --corpus_data_path tests/resources/corpus_data_sample.parquet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. autorag dashboard - Dashboard interativo de resultados                   │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Inicia um dashboard web para visualizar e comparar resultados de avaliações.
    Mostra métricas, gráficos e permite análise interativa.

  SINTAXE:
    autorag dashboard [OPTIONS]

  OPÇÕES:
    --trial_dir PATH    Caminho para o diretório do trial (obrigatório)
    --port INT          Porta do servidor (padrão: 7690)

  EXEMPLOS:
    # Dashboard para projeto simples
    autorag dashboard --trial_dir tutorial/projeto_simples/0 --port 7690

    # Dashboard para projeto local
    autorag dashboard --trial_dir tutorial/projeto_local/0 --port 7691

  ACESSO:
    Após iniciar, acesse: http://localhost:7690

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. autorag run_api - Servidor REST API                                      │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Inicia um servidor REST API para fazer queries ao melhor pipeline encontrado.
    Permite integração com aplicações externas.

  SINTAXE:
    autorag run_api [OPTIONS]

  OPÇÕES:
    --trial_dir PATH     Caminho para o diretório do trial
    --config_path PATH   Caminho para arquivo YAML de config extraída
    --project_dir PATH   Caminho para o diretório do projeto
    --host STR           Endereço do host (padrão: 0.0.0.0)
    --port INT           Porta do servidor (padrão: 8000)
    --remote BOOL        Modo remoto (padrão: False)

  EXEMPLOS:
    # API a partir de trial existente
    autorag run_api --trial_dir tutorial/projeto_local/0 --port 8000

    # API a partir de config extraída
    autorag run_api --config_path best_config.yaml --project_dir ./projeto

  ENDPOINTS:
    POST /v1/run         Executar query
    GET  /health         Health check
    GET  /docs           Documentação Swagger

  USO DA API:
    # Via curl
    curl -X POST http://localhost:8000/v1/run \\
        -H "Content-Type: application/json" \\
        -d '{"query": "What is the meaning of life?"}'

    # Via Python
    import requests
    response = requests.post(
        "http://localhost:8000/v1/run",
        json={"query": "What is the meaning of life?"}
    )
    print(response.json())

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. autorag run_web - Interface Web Streamlit                                │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Inicia uma interface web interativa usando Streamlit.
    Permite fazer queries e ver resultados de forma visual.

  SINTAXE:
    autorag run_web [OPTIONS]

  OPÇÕES:
    --trial_path PATH    Caminho para o diretório do trial
    --yaml_path PATH     Caminho para arquivo YAML
    --project_dir PATH   Caminho para o diretório do projeto

  EXEMPLO:
    autorag run_web --trial_path tutorial/projeto_local/0

  NOTA:
    Requer Streamlit instalado: pip install streamlit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. autorag extract_best_config - Extrair melhor configuração                │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Extrai a melhor configuração encontrada em um trial para um arquivo YAML.
    Útil para produção ou para usar com run_api.

  SINTAXE:
    autorag extract_best_config [OPTIONS]

  OPÇÕES:
    --trial_path PATH    Caminho para o diretório do trial
    --output_path PATH   Caminho para salvar o YAML (deve terminar em .yaml)

  EXEMPLO:
    autorag extract_best_config \\
        --trial_path tutorial/projeto_local/0 \\
        --output_path tutorial/melhor_config.yaml

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. autorag restart_evaluate - Reiniciar avaliação interrompida              │
└─────────────────────────────────────────────────────────────────────────────┘

  DESCRIÇÃO:
    Reinicia uma avaliação que foi interrompida.
    Continua de onde parou sem perder progresso.

  SINTAXE:
    autorag restart_evaluate [OPTIONS]

  OPÇÕES:
    --trial_path PATH    Caminho para o diretório do trial incompleto

  EXEMPLO:
    autorag restart_evaluate --trial_path tutorial/projeto_local/0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│ RESUMO RÁPIDO DOS COMANDOS                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  autorag evaluate          → Executar avaliação de pipeline
  autorag validate          → Validar configuração YAML
  autorag dashboard         → Dashboard web de resultados
  autorag run_api           → Servidor REST API
  autorag run_web           → Interface web Streamlit
  autorag extract_best_config → Extrair melhor configuração
  autorag restart_evaluate  → Reiniciar avaliação interrompida
"""
    print(cli_docs)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 4: EXECUÇÃO DE AVALIAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

def verificar_dependencias(tipo="simples"):
    """
    Verifica se as dependências necessárias estão instaladas.
    
    Args:
        tipo: "simples" (só pandas), "local" (+ torch, sentence-transformers), "openai" (+ openai key)
    
    Returns:
        bool: True se todas as dependências estão disponíveis
    """
    print("\n🔍 Verificando dependências...")
    
    # Verificar pandas (sempre necessário)
    try:
        import pandas
        print(f"   ✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("   ❌ Pandas não instalado. Execute: pip install pandas")
        return False
    
    if tipo in ["local", "openai"]:
        try:
            import torch
            print(f"   ✅ PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"      GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("      GPU: Não disponível (usando CPU)")
        except ImportError:
            print("   ❌ PyTorch não instalado. Execute: pip install torch")
            return False
    
    if tipo == "local":
        try:
            import sentence_transformers
            print(f"   ✅ Sentence Transformers: {sentence_transformers.__version__}")
        except ImportError:
            print("   ❌ Sentence Transformers não instalado.")
            print("      Execute: pip install sentence-transformers llama-index-embeddings-huggingface")
            return False
    
    if tipo == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            print(f"   ✅ OPENAI_API_KEY: Configurada ({api_key[:8]}...)")
        else:
            print("   ❌ OPENAI_API_KEY não configurada.")
            print("      Execute: export OPENAI_API_KEY='sk-sua-chave'")
            return False
    
    return True


def executar_avaliacao(config_key, skip_deps_check=False):
    """
    Executa uma avaliação usando a configuração especificada.
    
    Args:
        config_key: Chave da configuração ("simples", "local", "bm25_compare", "openai")
        skip_deps_check: Pular verificação de dependências
    
    Returns:
        str: Caminho do diretório do projeto ou None se falhar
    """
    from autorag.evaluator import Evaluator
    
    config_path = CONFIG_FILES.get(config_key)
    project_dir = PROJECT_DIRS.get(config_key)
    
    if not config_path or not os.path.exists(config_path):
        print(f"❌ Configuração '{config_key}' não encontrada: {config_path}")
        return None
    
    # Determinar tipo de dependência
    deps_type = "simples"
    if config_key == "local":
        deps_type = "local"
    elif config_key == "openai":
        deps_type = "openai"
    
    # Verificar dependências
    if not skip_deps_check and not verificar_dependencias(deps_type):
        return None
    
    print(f"\n{'='*80}")
    print(f"🚀 EXECUTANDO AVALIAÇÃO: {config_key.upper()}")
    print("=" * 80)
    print(f"   📄 Config: {config_path}")
    print(f"   📁 QA Data: {QA_PATH}")
    print(f"   📁 Corpus: {CORPUS_PATH}")
    print(f"   📂 Output: {project_dir}")
    print(f"\n   ⏳ Iniciando avaliação...")
    
    start_time = datetime.now()
    
    evaluator = Evaluator(
        qa_data_path=QA_PATH,
        corpus_data_path=CORPUS_PATH,
        project_dir=project_dir
    )
    
    evaluator.start_trial(config_path, skip_validation=True)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   ✅ Avaliação concluída!")
    print(f"   ⏱️  Tempo: {duration}")
    print(f"   📂 Resultados: {project_dir}")
    
    return project_dir


def executar_avaliacao_simples():
    """Executa avaliação simples com apenas BM25."""
    return executar_avaliacao("simples")


def executar_avaliacao_local():
    """Executa avaliação com modelos locais (BM25 + VectorDB + Híbrido)."""
    return executar_avaliacao("local")


def executar_avaliacao_bm25_compare():
    """Executa comparação de tokenizadores BM25."""
    return executar_avaliacao("bm25_compare")


def executar_avaliacao_openai():
    """Executa avaliação com OpenAI embeddings."""
    return executar_avaliacao("openai")


def executar_todas_avaliacoes():
    """Executa todas as avaliações disponíveis."""
    print("\n" + "=" * 80)
    print("🚀 EXECUTANDO TODAS AS AVALIAÇÕES")
    print("=" * 80)
    
    results = {}
    
    # Sempre executar simples e bm25_compare (não precisam de deps especiais)
    for config_key in ["simples", "bm25_compare"]:
        try:
            project_dir = executar_avaliacao(config_key, skip_deps_check=True)
            results[config_key] = project_dir
        except Exception as e:
            print(f"❌ Erro em {config_key}: {e}")
            results[config_key] = None
    
    # Tentar local se deps disponíveis
    if verificar_dependencias("local"):
        try:
            project_dir = executar_avaliacao("local", skip_deps_check=True)
            results["local"] = project_dir
        except Exception as e:
            print(f"❌ Erro em local: {e}")
            results["local"] = None
    
    # Tentar openai se deps disponíveis
    if verificar_dependencias("openai"):
        try:
            project_dir = executar_avaliacao("openai", skip_deps_check=True)
            results["openai"] = project_dir
        except Exception as e:
            print(f"❌ Erro em openai: {e}")
            results["openai"] = None
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 5: ANÁLISE DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

def analisar_resultados(project_dir=None):
    """
    Analisa os resultados de uma avaliação do AutoRAG.
    
    Args:
        project_dir: Diretório do projeto. Se None, analisa todos os projetos do tutorial.
    """
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("📈 PARTE 5: ANÁLISE DOS RESULTADOS")
    print("=" * 80)
    
    if project_dir:
        project_dirs = [project_dir]
    else:
        # Encontrar todos os projetos do tutorial
        project_dirs = sorted(glob.glob(os.path.join(TUTORIAL_DIR, "projeto_*")))
    
    if not project_dirs:
        print("❌ Nenhum projeto encontrado para análise!")
        print("   Execute primeiro uma avaliação:")
        print("   $ python tutorial_autorag.py --run-simple")
        print("   $ python tutorial_autorag.py --run-local")
        return
    
    for proj_dir in project_dirs:
        proj_name = os.path.basename(proj_dir)
        
        # Encontrar o último trial
        trial_dirs = sorted(glob.glob(os.path.join(proj_dir, "[0-9]*")))
        if not trial_dirs:
            continue
        
        trial_dir = trial_dirs[-1]
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ 📁 PROJETO: {proj_name:<62} ║
║    Trial: {os.path.basename(trial_dir):<64} ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
        
        # Resumo geral
        summary_path = os.path.join(trial_dir, "summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            print("📊 RESUMO GERAL (Melhores Módulos):")
            print("─" * 70)
            cols = ['node_type', 'best_module_name', 'best_module_params']
            print(summary[cols].to_string(index=False))
            print()
        
        # Métricas detalhadas por node
        for node_line_dir in sorted(glob.glob(os.path.join(trial_dir, "*_node_line"))):
            node_line_name = os.path.basename(node_line_dir)
            
            for node_dir in sorted(glob.glob(os.path.join(node_line_dir, "*"))):
                if not os.path.isdir(node_dir):
                    continue
                
                node_name = os.path.basename(node_dir)
                node_summary_path = os.path.join(node_dir, "summary.csv")
                
                if os.path.exists(node_summary_path):
                    node_summary = pd.read_csv(node_summary_path)
                    
                    print(f"📌 {node_name}:")
                    print("─" * 70)
                    
                    # Encontrar colunas de métricas
                    metric_cols = [c for c in node_summary.columns 
                                   if c.startswith('retrieval_') or c.startswith('generation_')]
                    
                    if metric_cols:
                        # Mostrar todas as configurações testadas
                        cols_to_show = ['module_name', 'is_best'] + metric_cols[:5]
                        cols_to_show = [c for c in cols_to_show if c in node_summary.columns]
                        print(node_summary[cols_to_show].to_string(index=False))
                        
                        # Identificar o melhor
                        best = node_summary[node_summary['is_best'] == True]
                        if not best.empty:
                            print(f"\n   🏆 Melhor: {best['module_name'].iloc[0]}")
                            for col in metric_cols[:5]:
                                if col in best.columns:
                                    print(f"      • {col}: {best[col].iloc[0]:.4f}")
                    print()


def comparar_metodos():
    """
    Compara os diferentes métodos de retrieval entre os projetos.
    """
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("🏆 COMPARAÇÃO DE MÉTODOS DE RETRIEVAL")
    print("=" * 80)
    
    # Procurar por projeto_local ou projeto_local_py
    projeto_local = None
    for proj in ["projeto_local", "projeto_local_py"]:
        path = os.path.join(TUTORIAL_DIR, proj)
        if os.path.exists(path):
            projeto_local = path
            break
    
    if not projeto_local:
        print("❌ Nenhum projeto com comparação encontrado!")
        print("   Execute: python tutorial_autorag.py --run-local")
        return
    
    # Encontrar último trial
    trial_dirs = sorted(glob.glob(os.path.join(projeto_local, "[0-9]*")))
    if not trial_dirs:
        print("❌ Nenhum trial encontrado!")
        return
    
    trial_dir = trial_dirs[-1]
    
    results = []
    
    # Coletar métricas de cada método
    for node_type in ["lexical_retrieval", "semantic_retrieval", "hybrid_retrieval"]:
        summary_path = os.path.join(trial_dir, "retrieve_node_line", node_type, "summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            best = df[df['is_best'] == True]
            if not best.empty:
                best = best.iloc[0]
                results.append({
                    "Método": node_type.replace("_retrieval", "").title(),
                    "Módulo": best['module_name'],
                    "F1": best.get('retrieval_f1', 0),
                    "Recall": best.get('retrieval_recall', 0),
                    "Precision": best.get('retrieval_precision', 0),
                    "NDCG": best.get('retrieval_ndcg', 0),
                    "MRR": best.get('retrieval_mrr', 0),
                })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('F1', ascending=False)
        
        print(f"\n📊 Resultados do Trial: {trial_dir}")
        print("─" * 80)
        print(results_df.to_string(index=False))
        
        # Insights automáticos
        print("\n💡 INSIGHTS AUTOMÁTICOS:")
        print("─" * 40)
        
        best_f1 = results_df.iloc[0]
        print(f"   🏆 Melhor F1: {best_f1['Método']} ({best_f1['Módulo']}) = {best_f1['F1']:.4f}")
        
        best_ndcg = results_df.loc[results_df['NDCG'].idxmax()]
        print(f"   📈 Melhor NDCG: {best_ndcg['Método']} = {best_ndcg['NDCG']:.4f}")
        
        best_recall = results_df.loc[results_df['Recall'].idxmax()]
        print(f"   🎯 Melhor Recall: {best_recall['Método']} = {best_recall['Recall']:.4f}")
        
        # Recomendação
        print("\n📝 RECOMENDAÇÃO:")
        if best_f1['Método'] == 'Hybrid':
            print("   O método híbrido obteve os melhores resultados!")
            print("   Isso indica que combinar busca léxica e semântica é benéfico para seu dataset.")
        elif best_f1['Método'] == 'Lexical':
            print("   A busca léxica (BM25) obteve os melhores resultados!")
            print("   Seu dataset pode ter forte correspondência lexical entre queries e documentos.")
        else:
            print("   A busca semântica obteve os melhores resultados!")
            print("   Os embeddings capturam bem a semântica das queries.")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 6: MÉTRICAS E MÓDULOS
# ═══════════════════════════════════════════════════════════════════════════════

def listar_metricas():
    """Lista todas as métricas disponíveis no AutoRAG."""
    print("\n" + "=" * 80)
    print("📊 PARTE 6: MÉTRICAS DISPONÍVEIS")
    print("=" * 80)
    
    metricas = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MÉTRICAS DE RETRIEVAL                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Medem a qualidade da busca de documentos relevantes.

┌─────────────────────────┬──────────────────────────────────────────────────────┐
│ Métrica                 │ Descrição                                            │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ retrieval_f1            │ F1 Score - média harmônica de precision e recall     │
│                         │ Equilibra precisão e cobertura                       │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ retrieval_recall        │ Recall - proporção de docs relevantes encontrados    │
│                         │ Importante quando não queremos perder informação     │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ retrieval_precision     │ Precision - proporção de docs retornados relevantes  │
│                         │ Importante quando queremos resultados precisos       │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ retrieval_ndcg          │ Normalized Discounted Cumulative Gain                │
│                         │ Considera a posição dos resultados no ranking        │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ retrieval_mrr           │ Mean Reciprocal Rank                                 │
│                         │ Posição média do primeiro resultado relevante        │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ retrieval_map           │ Mean Average Precision                               │
│                         │ Média das precisões em cada posição relevante        │
└─────────────────────────┴──────────────────────────────────────────────────────┘

QUANDO USAR CADA MÉTRICA:
• F1: Quando você quer equilibrar precision e recall
• Recall: Quando é crítico não perder documentos relevantes (ex: pesquisa médica)
• Precision: Quando é crítico não mostrar documentos irrelevantes (ex: chatbot)
• NDCG: Quando a ordem dos resultados importa
• MRR: Quando você só se importa com o primeiro resultado relevante

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MÉTRICAS DE GERAÇÃO                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Medem a qualidade do texto gerado pelo LLM.

┌─────────────────────────┬──────────────────────────────────────────────────────┐
│ Métrica                 │ Descrição                                            │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ bleu                    │ BLEU Score - correspondência de n-gramas             │
│                         │ Mede sobreposição de palavras/frases                 │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ meteor                  │ METEOR - considera sinônimos e stemming              │
│                         │ Mais flexível que BLEU                               │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ rouge                   │ ROUGE - sobreposição de sequências                   │
│                         │ Bom para sumarização                                 │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ sem_score               │ Semantic Score - similaridade semântica              │
│                         │ Usa embeddings para comparar significado             │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ bert_score              │ BERTScore - similaridade contextual com BERT         │
│                         │ Captura nuances semânticas                           │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ g_eval                  │ G-Eval - avaliação por GPT                           │
│                         │ Usa LLM para avaliar qualidade                       │
└─────────────────────────┴──────────────────────────────────────────────────────┘

MÉTRICAS DE COMPRESSÃO:
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│ retrieval_token_f1              │ F1 a nível de token                          │
│ retrieval_token_recall          │ Recall a nível de token                      │
│ retrieval_token_precision       │ Precision a nível de token                   │
└─────────────────────────────────┴──────────────────────────────────────────────┘
"""
    print(metricas)


def listar_modulos():
    """Lista todos os módulos disponíveis no AutoRAG."""
    print("\n" + "=" * 80)
    print("🔧 MÓDULOS DISPONÍVEIS")
    print("=" * 80)
    
    modulos = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MÓDULOS DE RETRIEVAL                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

BUSCA LÉXICA (lexical_retrieval):
┌──────────────┬────────────────────────────────────────────────────────────────┐
│ Módulo       │ Descrição                                                      │
├──────────────┼────────────────────────────────────────────────────────────────┤
│ bm25         │ BM25 - algoritmo clássico de busca por palavras-chave          │
│              │ Parâmetros:                                                    │
│              │   • bm25_tokenizer: [porter_stemmer, space, gpt2, ko_kiwi]     │
└──────────────┴────────────────────────────────────────────────────────────────┘

BUSCA SEMÂNTICA (semantic_retrieval):
┌──────────────┬────────────────────────────────────────────────────────────────┐
│ Módulo       │ Descrição                                                      │
├──────────────┼────────────────────────────────────────────────────────────────┤
│ vectordb     │ Busca por similaridade de embeddings em Vector Database        │
│              │ Parâmetros:                                                    │
│              │   • vectordb: nome do vectordb configurado                     │
└──────────────┴────────────────────────────────────────────────────────────────┘

BUSCA HÍBRIDA (hybrid_retrieval):
┌──────────────┬────────────────────────────────────────────────────────────────┐
│ Módulo       │ Descrição                                                      │
├──────────────┼────────────────────────────────────────────────────────────────┤
│ hybrid_rrf   │ Reciprocal Rank Fusion - combina rankings por posição          │
│              │ Parâmetros:                                                    │
│              │   • weight_range: (min, max) para parâmetro k                  │
├──────────────┼────────────────────────────────────────────────────────────────┤
│ hybrid_cc    │ Convex Combination - combina scores com pesos                  │
│              │ Parâmetros:                                                    │
│              │   • normalize_method: [mm, tmm, z, dbsf]                       │
│              │   • weight_range: (min, max) para peso                         │
│              │   • test_weight_size: número de pesos a testar                 │
└──────────────┴────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MÓDULOS DE RERANKING                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────┬────────────────────────────────────────────────┐
│ Módulo                        │ Descrição                                      │
├───────────────────────────────┼────────────────────────────────────────────────┤
│ pass_reranker                 │ Não faz reranking (baseline)                   │
│ sentence_transformer_reranker │ Reranker com Sentence Transformers             │
│ cohere_reranker               │ Reranker da Cohere (API)                       │
│ jina_reranker                 │ Reranker da Jina (API)                         │
│ colbert_reranker              │ ColBERT reranker                               │
│ monot5                        │ MonoT5 reranker                                │
│ tart                          │ TART reranker                                  │
│ upr                           │ UPR reranker                                   │
│ rankgpt                       │ RankGPT (usa LLM)                              │
│ flag_embedding_reranker       │ Flag Embedding reranker                        │
│ flag_embedding_llm_reranker   │ Flag Embedding com LLM                         │
│ time_reranker                 │ Reranker baseado em data                       │
│ openvino_reranker             │ Reranker otimizado para Intel                  │
│ voyageai_reranker             │ Voyage AI reranker                             │
│ mixedbreadai_reranker         │ MixedBread AI reranker                         │
│ flashrank_reranker            │ FlashRank (rápido)                             │
└───────────────────────────────┴────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MÓDULOS DE QUERY EXPANSION                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────┬────────────────────────────────────────────────┐
│ Módulo                        │ Descrição                                      │
├───────────────────────────────┼────────────────────────────────────────────────┤
│ pass_query_expansion          │ Não expande (baseline)                         │
│ query_decompose               │ Decompõe query em sub-queries                  │
│ hyde                          │ HyDE - gera documento hipotético               │
│ multi_query_expansion         │ Gera múltiplas versões da query                │
└───────────────────────────────┴────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          VECTOR DATABASES                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌──────────────┬────────────┬──────────────────────────────────────────────────┐
│ DB           │ Tipo       │ Uso Recomendado                                  │
├──────────────┼────────────┼──────────────────────────────────────────────────┤
│ chroma       │ ephemeral  │ Testes rápidos, desenvolvimento                  │
│ chroma       │ persistent │ Desenvolvimento, projetos pequenos               │
│ milvus       │ server     │ Produção, escalabilidade                         │
│ pinecone     │ cloud      │ Produção serverless                              │
│ qdrant       │ server     │ Produção, alta performance                       │
│ weaviate     │ server     │ Produção, busca híbrida nativa                   │
│ couchbase    │ server     │ Produção, integração com dados existentes        │
└──────────────┴────────────┴──────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          EMBEDDING MODELS                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

LOCAL (não requer API):
┌────────────────────────────────────┬─────────┬─────────────────────────────────┐
│ Nome                               │ Dims    │ Notas                           │
├────────────────────────────────────┼─────────┼─────────────────────────────────┤
│ huggingface_all_mpnet_base_v2      │ 768     │ Bom equilíbrio qualidade/veloc. │
│ huggingface_baai_bge_small         │ 384     │ Menor e mais rápido             │
│ huggingface_bge_m3                 │ 1024    │ Multilingual                    │
│ mock                               │ 768     │ Para testes (aleatório)         │
└────────────────────────────────────┴─────────┴─────────────────────────────────┘

API (requer chave):
┌────────────────────────────────────┬─────────┬─────────────────────────────────┐
│ Nome                               │ Dims    │ Notas                           │
├────────────────────────────────────┼─────────┼─────────────────────────────────┤
│ openai_embed_3_small               │ 1536    │ Bom custo-benefício             │
│ openai_embed_3_large               │ 3072    │ Maior qualidade                 │
│ openai                             │ 1536    │ Ada embedding (legado)          │
└────────────────────────────────────┴─────────┴─────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          TOKENIZADORES BM25                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────┬────────────────────────────────────────────────────────────┐
│ Tokenizador     │ Descrição                                                  │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ porter_stemmer  │ Stemmer Porter - reduz palavras à raiz (inglês)            │
│ space           │ Tokenização simples por espaço                             │
│ gpt2            │ Tokenizador do GPT-2 (subword)                             │
│ ko_kiwi         │ Tokenizador coreano Kiwi                                   │
│ ko_okt          │ Tokenizador coreano OKT                                    │
│ ko_kkma         │ Tokenizador coreano KKMA                                   │
│ sudachipy       │ Tokenizador japonês                                        │
└─────────────────┴────────────────────────────────────────────────────────────┘
"""
    print(modulos)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 7: CRIAÇÃO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def mostrar_exemplo_criacao_dados():
    """Mostra como criar dados próprios para avaliação."""
    print("\n" + "=" * 80)
    print("📝 PARTE 7: CRIANDO SEUS PRÓPRIOS DADOS")
    print("=" * 80)
    
    exemplo = '''
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    GUIA PARA CRIAR DADOS DE AVALIAÇÃO                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

O AutoRAG precisa de dois arquivos Parquet: QA Dataset e Corpus Dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 EXEMPLO COMPLETO EM PYTHON:

```python
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 1: CRIAR CORPUS (seus documentos)
# ═══════════════════════════════════════════════════════════════════════════════

corpus_data = [
    {
        "doc_id": "doc_001",
        "contents": "Python é uma linguagem de programação de alto nível, "
                   "interpretada e de propósito geral. Foi criada por Guido "
                   "van Rossum e lançada em 1991. Python é conhecida por sua "
                   "sintaxe clara e legível.",
        "metadata": {"source": "wikipedia", "topic": "programming", "year": 2024}
    },
    {
        "doc_id": "doc_002",
        "contents": "Machine Learning é um subcampo da inteligência artificial "
                   "que permite que sistemas aprendam padrões a partir de dados "
                   "sem serem explicitamente programados. Existem três tipos "
                   "principais: supervisionado, não-supervisionado e por reforço.",
        "metadata": {"source": "textbook", "topic": "ai", "year": 2023}
    },
    {
        "doc_id": "doc_003",
        "contents": "RAG (Retrieval-Augmented Generation) é uma técnica que "
                   "combina busca de documentos relevantes com geração de "
                   "texto por modelos de linguagem. Foi introduzido em 2020 "
                   "por pesquisadores do Facebook AI.",
        "metadata": {"source": "paper", "topic": "nlp", "year": 2020}
    },
    {
        "doc_id": "doc_004",
        "contents": "Vector databases são sistemas de banco de dados otimizados "
                   "para armazenar e buscar embeddings de alta dimensionalidade. "
                   "Exemplos incluem Pinecone, Milvus, Chroma e Qdrant.",
        "metadata": {"source": "blog", "topic": "databases", "year": 2024}
    },
]

corpus_df = pd.DataFrame(corpus_data)
corpus_df.to_parquet("meu_corpus.parquet", index=False)
print(f"✅ Corpus criado: {len(corpus_df)} documentos")

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 2: CRIAR QA DATASET (perguntas e respostas de avaliação)
# ═══════════════════════════════════════════════════════════════════════════════

qa_data = [
    {
        "qid": "q_001",
        "query": "O que é Python e quem criou essa linguagem?",
        "retrieval_gt": [["doc_001"]],  # Lista de listas com IDs dos docs relevantes
        "generation_gt": ["Python é uma linguagem de programação criada por Guido van Rossum em 1991."]
    },
    {
        "qid": "q_002",
        "query": "Quais são os tipos de Machine Learning?",
        "retrieval_gt": [["doc_002"]],
        "generation_gt": ["Existem três tipos: supervisionado, não-supervisionado e por reforço."]
    },
    {
        "qid": "q_003",
        "query": "Como funciona RAG e quando foi criado?",
        "retrieval_gt": [["doc_003"]],
        "generation_gt": ["RAG combina busca de documentos com geração por LLMs, criado em 2020."]
    },
    {
        "qid": "q_004",
        "query": "Quais são exemplos de vector databases?",
        "retrieval_gt": [["doc_004"]],
        "generation_gt": ["Exemplos incluem Pinecone, Milvus, Chroma e Qdrant."]
    },
    {
        "qid": "q_005",
        "query": "O que são técnicas de NLP modernas?",
        "retrieval_gt": [["doc_003", "doc_002"]],  # Múltiplos docs relevantes
        "generation_gt": ["RAG e Machine Learning são técnicas modernas de NLP."]
    },
]

qa_df = pd.DataFrame(qa_data)
qa_df.to_parquet("meu_qa.parquet", index=False)
print(f"✅ QA criado: {len(qa_df)} perguntas")

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 3: VERIFICAR OS DADOS
# ═══════════════════════════════════════════════════════════════════════════════

print("\\n📊 Verificando dados criados:")
print(f"\\nCorpus:")
print(corpus_df[['doc_id', 'contents']].head())
print(f"\\nQA:")
print(qa_df[['qid', 'query', 'retrieval_gt']].head())

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 4: EXECUTAR AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

from autorag.evaluator import Evaluator

# Usando config local (sem API keys)
evaluator = Evaluator(
    qa_data_path="meu_qa.parquet",
    corpus_data_path="meu_corpus.parquet",
    project_dir="./meu_projeto"
)

# Executar avaliação com config do tutorial
evaluator.start_trial("tutorial/config_local.yaml", skip_validation=True)

print("\\n✅ Avaliação concluída!")
print("   Resultados em: ./meu_projeto/0/")
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 DICAS IMPORTANTES:

1. retrieval_gt deve ser uma LISTA DE LISTAS:
   • [["doc_001"]] → Um conjunto de docs relevantes
   • [["doc_001", "doc_002"]] → Múltiplos docs em um conjunto
   • [["doc_001"], ["doc_002"]] → Múltiplos conjuntos válidos

2. generation_gt é uma LISTA de respostas aceitáveis:
   • ["Resposta 1", "Resposta alternativa 2"]

3. metadata é OPCIONAL mas útil para filtros

4. Quanto mais dados de avaliação, melhor a qualidade das métricas

5. Certifique-se que todos os doc_id em retrieval_gt existem no corpus!
'''
    print(exemplo)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 8: EXEMPLO DE USO DA API
# ═══════════════════════════════════════════════════════════════════════════════

def mostrar_exemplo_api():
    """Mostra exemplos de uso da API do AutoRAG."""
    print("\n" + "=" * 80)
    print("🌐 PARTE 8: USANDO A API REST")
    print("=" * 80)
    
    exemplo = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          GUIA DA API REST DO AUTORAG                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📡 INICIAR O SERVIDOR

# Opção 1: A partir de um trial existente
autorag run_api --trial_dir tutorial/projeto_local/0 --port 8000

# Opção 2: A partir de config extraída
autorag extract_best_config --trial_path tutorial/projeto_local/0 --output_path best.yaml
autorag run_api --config_path best.yaml --project_dir tutorial/projeto_local --port 8000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔌 ENDPOINTS DISPONÍVEIS

┌────────────────────┬──────────┬─────────────────────────────────────────────────┐
│ Endpoint           │ Método   │ Descrição                                       │
├────────────────────┼──────────┼─────────────────────────────────────────────────┤
│ /v1/run            │ POST     │ Executa query no pipeline RAG                   │
│ /health            │ GET      │ Health check do servidor                        │
│ /docs              │ GET      │ Documentação Swagger UI                         │
│ /redoc             │ GET      │ Documentação ReDoc                              │
└────────────────────┴──────────┴─────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 EXEMPLOS DE USO

# 1. VIA CURL
curl -X POST http://localhost:8000/v1/run \\
    -H "Content-Type: application/json" \\
    -d '{"query": "What is the meaning of life?"}'

# 2. VIA PYTHON
import requests

# Query simples
response = requests.post(
    "http://localhost:8000/v1/run",
    json={"query": "What is Python?"}
)
result = response.json()
print(f"Resposta: {result}")

# Com timeout
response = requests.post(
    "http://localhost:8000/v1/run",
    json={"query": "Explain machine learning"},
    timeout=30
)

# 3. VIA HTTPX (assíncrono)
import httpx
import asyncio

async def query_api(question):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/run",
            json={"query": question}
        )
        return response.json()

# Executar
result = asyncio.run(query_api("What is RAG?"))
print(result)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 FORMATO DA RESPOSTA

{
    "result": "A resposta gerada pelo pipeline",
    "retrieved_passages": [
        {
            "doc_id": "doc_001",
            "content": "Conteúdo do documento...",
            "score": 0.95
        }
    ],
    "elapsed_time": 0.234
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖥️ DASHBOARD INTERATIVO

# Iniciar dashboard
autorag dashboard --trial_dir tutorial/projeto_local/0 --port 7690

# Acesse: http://localhost:7690

Funcionalidades:
• Visualização de métricas
• Comparação de módulos
• Gráficos interativos
• Exportação de resultados

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 INTERFACE WEB (STREAMLIT)

# Iniciar interface web
autorag run_web --trial_path tutorial/projeto_local/0

Funcionalidades:
• Interface visual para queries
• Visualização de resultados
• Histórico de consultas
"""
    print(exemplo)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Tutorial Completo do AutoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              EXEMPLOS DE USO                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  # Tutorial informativo (mostra tudo)
  python tutorial_autorag.py

  # Executar avaliações
  python tutorial_autorag.py --run-simple        # Apenas BM25
  python tutorial_autorag.py --run-local         # BM25 + VectorDB + Híbrido
  python tutorial_autorag.py --run-bm25-compare  # Comparação tokenizadores
  python tutorial_autorag.py --run-openai        # Com OpenAI (requer API key)
  python tutorial_autorag.py --run-all           # Todas as avaliações

  # Analisar resultados
  python tutorial_autorag.py --analyze           # Analisa todos os resultados
  python tutorial_autorag.py --compare           # Compara métodos

  # Documentação
  python tutorial_autorag.py --cli-help          # Todos os comandos CLI
  python tutorial_autorag.py --metrics           # Lista métricas
  python tutorial_autorag.py --modules           # Lista módulos
  python tutorial_autorag.py --configs           # Mostra configs YAML
  python tutorial_autorag.py --create-data       # Exemplo de criação de dados
  python tutorial_autorag.py --api               # Exemplo de uso da API

  # Combinar flags
  python tutorial_autorag.py --run-local --analyze --compare
        """
    )
    
    # Grupo de execução
    exec_group = parser.add_argument_group('Execução')
    exec_group.add_argument("--run-simple", action="store_true",
                           help="Executar avaliação simples (apenas BM25)")
    exec_group.add_argument("--run-local", action="store_true",
                           help="Executar com modelos locais (BM25 + VectorDB + Híbrido)")
    exec_group.add_argument("--run-bm25-compare", action="store_true",
                           help="Executar comparação de tokenizadores BM25")
    exec_group.add_argument("--run-openai", action="store_true",
                           help="Executar com OpenAI (requer OPENAI_API_KEY)")
    exec_group.add_argument("--run-all", action="store_true",
                           help="Executar todas as avaliações disponíveis")
    
    # Grupo de análise
    analysis_group = parser.add_argument_group('Análise')
    analysis_group.add_argument("--analyze", action="store_true",
                               help="Analisar resultados existentes")
    analysis_group.add_argument("--compare", action="store_true",
                               help="Comparar métodos de retrieval")
    
    # Grupo de documentação
    docs_group = parser.add_argument_group('Documentação')
    docs_group.add_argument("--cli-help", action="store_true",
                           help="Mostrar todos os comandos CLI")
    docs_group.add_argument("--metrics", action="store_true",
                           help="Listar métricas disponíveis")
    docs_group.add_argument("--modules", action="store_true",
                           help="Listar módulos disponíveis")
    docs_group.add_argument("--configs", action="store_true",
                           help="Mostrar conteúdo das configs YAML")
    docs_group.add_argument("--create-data", action="store_true",
                           help="Mostrar exemplo de criação de dados")
    docs_group.add_argument("--api", action="store_true",
                           help="Mostrar exemplo de uso da API")
    
    # Grupo de dados
    data_group = parser.add_argument_group('Dados')
    data_group.add_argument("--explore-data", action="store_true",
                           help="Explorar estrutura dos dados")
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                        🚀 TUTORIAL COMPLETO DO AUTORAG                        ║
║                                                                               ║
║   AutoRAG: Ferramenta AutoML para encontrar o melhor pipeline RAG             ║
║   Repositório: https://github.com/Marker-Inc-Korea/AutoRAG                    ║
║                                                                               ║
║   Configs disponíveis:                                                        ║
║   • tutorial/config_simples.yaml       (BM25 only)                            ║
║   • tutorial/config_local.yaml         (BM25 + VectorDB + Híbrido)            ║
║   • tutorial/config_comparacao_bm25.yaml (Tokenizers comparison)              ║
║   • tutorial/config_memoria_completo.yaml (OpenAI)                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar se algum argumento foi passado
    has_args = any([
        args.run_simple, args.run_local, args.run_bm25_compare, args.run_openai, args.run_all,
        args.analyze, args.compare,
        args.cli_help, args.metrics, args.modules, args.configs, args.create_data, args.api,
        args.explore_data
    ])
    
    # Executar ações específicas
    if args.run_simple:
        project_dir = executar_avaliacao_simples()
        if project_dir and args.analyze:
            analisar_resultados(project_dir)
    
    if args.run_local:
        project_dir = executar_avaliacao_local()
        if project_dir:
            if args.analyze:
                analisar_resultados(project_dir)
            if args.compare:
                comparar_metodos()
    
    if args.run_bm25_compare:
        project_dir = executar_avaliacao_bm25_compare()
        if project_dir and args.analyze:
            analisar_resultados(project_dir)
    
    if args.run_openai:
        project_dir = executar_avaliacao_openai()
        if project_dir and args.analyze:
            analisar_resultados(project_dir)
    
    if args.run_all:
        executar_todas_avaliacoes()
        if args.analyze:
            analisar_resultados()
        if args.compare:
            comparar_metodos()
    
    # Análise sem execução
    if args.analyze and not any([args.run_simple, args.run_local, args.run_bm25_compare, args.run_openai, args.run_all]):
        analisar_resultados()
    
    if args.compare and not any([args.run_local, args.run_all]):
        comparar_metodos()
    
    # Documentação
    if args.cli_help:
        mostrar_comandos_cli()
    
    if args.metrics:
        listar_metricas()
    
    if args.modules:
        listar_modulos()
    
    if args.configs:
        mostrar_conteudo_configs()
    
    if args.create_data:
        mostrar_exemplo_criacao_dados()
    
    if args.api:
        mostrar_exemplo_api()
    
    if args.explore_data:
        explorar_dados()
    
    # Modo padrão: tutorial completo
    if not has_args:
        explorar_dados()
        listar_configuracoes()
        mostrar_exemplo_yaml_completo()
        mostrar_comandos_cli()
        listar_metricas()
        listar_modulos()
        mostrar_exemplo_api()
        mostrar_exemplo_criacao_dados()
        
        print("\n" + "=" * 80)
        print("✅ TUTORIAL COMPLETO!")
        print("=" * 80)
        print("""
🎯 PRÓXIMOS PASSOS RECOMENDADOS:

1. TESTE RÁPIDO (apenas BM25, ~30s):
   $ python tutorial_autorag.py --run-simple

2. TESTE COMPLETO COM MODELOS LOCAIS (~3min):
   $ python tutorial_autorag.py --run-local --analyze --compare

3. COMPARAÇÃO DE TOKENIZADORES BM25 (~1min):
   $ python tutorial_autorag.py --run-bm25-compare

4. VISUALIZAR RESULTADOS NO DASHBOARD:
   $ autorag dashboard --trial_dir tutorial/projeto_local/0 --port 7690

5. INICIAR API SERVER:
   $ autorag run_api --trial_dir tutorial/projeto_local/0 --port 8000

6. VER AJUDA DE COMANDOS CLI:
   $ autorag --help

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 ARQUIVOS DE CONFIGURAÇÃO:

   tutorial/config_simples.yaml         → Apenas BM25
   tutorial/config_local.yaml           → BM25 + VectorDB + Híbrido (LOCAL)
   tutorial/config_comparacao_bm25.yaml → Comparação de tokenizadores
   tutorial/config_memoria_completo.yaml → Com OpenAI embeddings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 PARA MAIS INFORMAÇÕES:

   $ python tutorial_autorag.py --cli-help     # Comandos CLI
   $ python tutorial_autorag.py --metrics      # Métricas disponíveis
   $ python tutorial_autorag.py --modules      # Módulos disponíveis
   $ python tutorial_autorag.py --create-data  # Criar seus dados
   $ python tutorial_autorag.py --api          # Usar a API
""")


if __name__ == "__main__":
    main()
