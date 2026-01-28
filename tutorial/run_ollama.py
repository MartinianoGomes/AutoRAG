# Script para executar AutoRAG com Ollama
# Execute este script em vez de usar o comando autorag diretamente

import os
import sys

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')  # Volta para o diretório raiz do projeto

# Registra o Ollama como modelo suportado
import autorag
from llama_index.llms.ollama import Ollama

# Adiciona Ollama aos modelos suportados
autorag.generator_models['ollama'] = Ollama

print("✅ Ollama registrado como modelo de LLM")
print(f"Modelos disponíveis: {list(autorag.generator_models.keys())}")

# Executa a avaliação
from autorag.evaluator import Evaluator

config_path = "tutorial/config_local_ollama.yaml"
qa_data_path = "tests/resources/qa_data_sample.parquet"
corpus_data_path = "tests/resources/corpus_data_sample.parquet"
project_dir = "tutorial/projeto_ollama"

print(f"\n🚀 Iniciando avaliação com Ollama...")
print(f"   Config: {config_path}")
print(f"   QA Data: {qa_data_path}")
print(f"   Corpus: {corpus_data_path}")
print(f"   Projeto: {project_dir}")

evaluator = Evaluator(qa_data_path, corpus_data_path, project_dir=project_dir)
evaluator.start_trial(config_path)

print("\n✅ Avaliação concluída!")
