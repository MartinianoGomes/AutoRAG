#!/usr/bin/env python3
"""
=============================================================================
🚀 TUTORIAL COMPLETO DO AUTORAG
=============================================================================

Este script demonstra como usar o AutoRAG para:
1. Explorar dados de entrada
2. Executar avaliações de pipeline RAG
3. Analisar resultados
4. Usar a API para inferência

Requisitos:
- AutoRAG instalado: pip install -e .
- Para recursos completos: pip install "AutoRAG[gpu]"
- Para usar OpenAI: export OPENAI_API_KEY=sua-chave
"""

import os
import sys
import pandas as pd

# Adicionar o diretório raiz ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# =============================================================================
# PARTE 1: ESTRUTURA DOS DADOS
# =============================================================================

def explorar_dados():
    """
    O AutoRAG requer dois arquivos parquet:
    
    1. QA Dataset (qa.parquet):
       - qid: ID único da pergunta
       - query: A pergunta
       - retrieval_gt: Lista de listas com IDs dos documentos relevantes
       - generation_gt: Lista de respostas esperadas (ground truth)
    
    2. Corpus Dataset (corpus.parquet):
       - doc_id: ID único do documento
       - contents: Conteúdo textual
       - metadata: Metadados opcionais (dict)
    """
    print("=" * 70)
    print("📊 ESTRUTURA DOS DADOS DO AUTORAG")
    print("=" * 70)
    
    # Carregar dados de exemplo
    qa_path = os.path.join(ROOT_DIR, "tests/resources/qa_data_sample.parquet")
    corpus_path = os.path.join(ROOT_DIR, "tests/resources/corpus_data_sample.parquet")
    
    qa_df = pd.read_parquet(qa_path)
    corpus_df = pd.read_parquet(corpus_path)
    
    print("\n📌 QA DATASET:")
    print(f"   Linhas: {len(qa_df)}")
    print(f"   Colunas: {list(qa_df.columns)}")
    print(f"\n   Exemplo de query: {qa_df['query'].iloc[0][:100]}...")
    print(f"   Exemplo de retrieval_gt: {qa_df['retrieval_gt'].iloc[0]}")
    
    print("\n📌 CORPUS DATASET:")
    print(f"   Linhas: {len(corpus_df)}")
    print(f"   Colunas: {list(corpus_df.columns)}")
    print(f"   Exemplo de doc_id: {corpus_df['doc_id'].iloc[0]}")
    print(f"   Exemplo de conteúdo: {corpus_df['contents'].iloc[0][:200]}...")
    
    return qa_df, corpus_df


# =============================================================================
# PARTE 2: CONFIGURAÇÃO YAML
# =============================================================================

def explicar_yaml():
    """
    O arquivo YAML define o pipeline de avaliação:
    
    vectordb:        # Configuração de vector databases
    node_lines:      # Linhas de nodes (etapas do pipeline)
      - nodes:       # Nodes dentro de cada linha
          - node_type: query_expansion / lexical_retrieval / etc
            strategy:
              metrics: [...]  # Métricas para avaliar
            modules:
              - module_type: nome_do_modulo
                param1: valor1
    """
    
    yaml_exemplo = """
# =============================================================================
# EXEMPLO DE CONFIGURAÇÃO YAML
# =============================================================================

# 1. CONFIGURAÇÃO SIMPLES (apenas BM25, sem API keys)
# -----------------------------------------------------------------------------
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 5
        modules:
          - module_type: bm25
            bm25_tokenizer: [porter_stemmer, space]

# 2. CONFIGURAÇÃO COM VECTOR DB (requer OpenAI API key)
# -----------------------------------------------------------------------------
vectordb:
  - name: chroma_openai
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_small
    collection_name: my_collection
    path: ${PROJECT_DIR}/resources/chroma

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

# 3. CONFIGURAÇÃO COM HÍBRIDO + RERANKER + GERAÇÃO
# -----------------------------------------------------------------------------
# (Veja sample_config/rag/full.yaml para exemplo completo)
"""
    print(yaml_exemplo)


# =============================================================================
# PARTE 3: EXECUÇÃO DA AVALIAÇÃO
# =============================================================================

def executar_avaliacao_simples():
    """
    Executa uma avaliação simples do pipeline RAG usando apenas BM25.
    Isso não requer API keys externas.
    """
    from autorag.evaluator import Evaluator
    
    print("\n" + "=" * 70)
    print("🚀 EXECUTANDO AVALIAÇÃO SIMPLES")
    print("=" * 70)
    
    # Caminhos dos dados
    qa_path = os.path.join(ROOT_DIR, "tests/resources/qa_data_sample.parquet")
    corpus_path = os.path.join(ROOT_DIR, "tests/resources/corpus_data_sample.parquet")
    config_path = os.path.join(ROOT_DIR, "tutorial/config_simples.yaml")
    project_dir = os.path.join(ROOT_DIR, "tutorial/projeto_python")
    
    # Criar avaliador
    evaluator = Evaluator(
        qa_data_path=qa_path,
        corpus_data_path=corpus_path,
        project_dir=project_dir
    )
    
    # Executar avaliação
    evaluator.start_trial(config_path, skip_validation=True)
    
    print("\n✅ Avaliação concluída!")
    print(f"   Resultados em: {project_dir}")
    
    return project_dir


# =============================================================================
# PARTE 4: ANÁLISE DOS RESULTADOS
# =============================================================================

def analisar_resultados(project_dir):
    """
    Analisa os resultados da avaliação.
    """
    import glob
    
    print("\n" + "=" * 70)
    print("📈 ANÁLISE DOS RESULTADOS")
    print("=" * 70)
    
    # Encontrar o diretório do trial mais recente
    trial_dirs = sorted(glob.glob(os.path.join(project_dir, "[0-9]*")))
    if not trial_dirs:
        print("❌ Nenhum trial encontrado!")
        return
    
    trial_dir = trial_dirs[-1]
    print(f"\n📁 Trial: {trial_dir}")
    
    # Carregar resumo
    summary_path = os.path.join(trial_dir, "summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        print("\n📊 Resumo do Trial:")
        print(summary.to_string())
    
    # Carregar métricas detalhadas de cada node
    for node_line_dir in glob.glob(os.path.join(trial_dir, "*_node_line")):
        node_line_name = os.path.basename(node_line_dir)
        print(f"\n📌 Node Line: {node_line_name}")
        
        for node_dir in glob.glob(os.path.join(node_line_dir, "*")):
            if not os.path.isdir(node_dir):
                continue
            
            node_name = os.path.basename(node_dir)
            summary_path = os.path.join(node_dir, "summary.csv")
            
            if os.path.exists(summary_path):
                node_summary = pd.read_csv(summary_path)
                print(f"\n   🔹 Node: {node_name}")
                print(f"   Melhor módulo: {node_summary.loc[node_summary['is_best'], 'module_name'].values[0]}")
                
                # Mostrar métricas
                metric_cols = [c for c in node_summary.columns 
                              if c.startswith('retrieval_') or c.startswith('generation_')]
                if metric_cols:
                    print(f"   Métricas:")
                    for _, row in node_summary.iterrows():
                        print(f"      {row['module_name']}:")
                        for col in metric_cols:
                            print(f"         {col}: {row[col]:.4f}")


# =============================================================================
# PARTE 5: USANDO A API PARA INFERÊNCIA
# =============================================================================

def exemplo_uso_api():
    """
    Exemplo de como usar a API do AutoRAG após a avaliação.
    
    Para iniciar o servidor API:
    $ autorag run_api --trial_dir tutorial/projeto_exemplo/0 --port 8000
    
    Para fazer uma query:
    $ curl -X POST http://localhost:8000/v1/run \
        -H "Content-Type: application/json" \
        -d '{"query": "What is the meaning of life?"}'
    """
    print("\n" + "=" * 70)
    print("🌐 USANDO A API")
    print("=" * 70)
    
    exemplo = """
# 1. Iniciar o servidor (em um terminal):
$ autorag run_api --trial_dir tutorial/projeto_exemplo/0 --port 8000

# 2. Fazer uma query (em outro terminal):
$ curl -X POST http://localhost:8000/v1/run \\
    -H "Content-Type: application/json" \\
    -d '{"query": "What is the meaning of life?"}'

# 3. Via Python:
import requests

response = requests.post(
    "http://localhost:8000/v1/run",
    json={"query": "What is the meaning of life?"}
)
print(response.json())
"""
    print(exemplo)


# =============================================================================
# PARTE 6: CRIANDO SEUS PRÓPRIOS DADOS
# =============================================================================

def criar_dados_exemplo():
    """
    Mostra como criar seus próprios dados para avaliação.
    """
    print("\n" + "=" * 70)
    print("📝 CRIANDO SEUS PRÓPRIOS DADOS")
    print("=" * 70)
    
    exemplo = """
import pandas as pd

# 1. CRIAR CORPUS (seus documentos)
# ----------------------------------
corpus_data = [
    {
        "doc_id": "doc_001",
        "contents": "Python é uma linguagem de programação de alto nível...",
        "metadata": {"source": "wikipedia", "topic": "programming"}
    },
    {
        "doc_id": "doc_002", 
        "contents": "Machine Learning é um subcampo da inteligência artificial...",
        "metadata": {"source": "wikipedia", "topic": "ai"}
    },
]
corpus_df = pd.DataFrame(corpus_data)
corpus_df.to_parquet("meu_corpus.parquet", index=False)

# 2. CRIAR QA DATASET (perguntas e respostas)
# -------------------------------------------
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
        "generation_gt": ["Machine Learning é um subcampo da IA"]
    },
]
qa_df = pd.DataFrame(qa_data)
qa_df.to_parquet("meu_qa.parquet", index=False)

# 3. EXECUTAR AVALIAÇÃO
# ---------------------
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="meu_qa.parquet",
    corpus_data_path="meu_corpus.parquet",
    project_dir="./meu_projeto"
)
evaluator.start_trial("config.yaml")
"""
    print(exemplo)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    🚀 TUTORIAL DO AUTORAG                             ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Explorar dados
    qa_df, corpus_df = explorar_dados()
    
    # 2. Explicar YAML
    print("\n" + "=" * 70)
    print("📄 CONFIGURAÇÃO YAML")
    print("=" * 70)
    explicar_yaml()
    
    # 3. Executar avaliação (descomente para executar)
    # project_dir = executar_avaliacao_simples()
    # analisar_resultados(project_dir)
    
    # 4. Mostrar exemplo de API
    exemplo_uso_api()
    
    # 5. Mostrar como criar dados
    criar_dados_exemplo()
    
    print("\n" + "=" * 70)
    print("✅ TUTORIAL COMPLETO!")
    print("=" * 70)
    print("""
Próximos passos:
1. Crie seus dados (corpus.parquet e qa.parquet)
2. Configure seu pipeline (config.yaml)
3. Execute a avaliação:
   $ autorag evaluate --config config.yaml --qa_data_path qa.parquet --corpus_data_path corpus.parquet --project_dir ./projeto
4. Visualize os resultados:
   $ autorag dashboard --trial_dir ./projeto/0 --port 7690
5. Faça deploy da API:
   $ autorag run_api --trial_dir ./projeto/0 --port 8000
""")
