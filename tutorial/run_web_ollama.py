# Script para iniciar a interface Web do AutoRAG com suporte ao Ollama
# Execute este script em vez de usar streamlit diretamente

import os
import sys

# Adiciona o diretório ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Registra o Ollama como modelo suportado ANTES de importar o resto
import autorag
from llama_index.llms.ollama import Ollama

autorag.generator_models['ollama'] = Ollama
print("✅ Ollama registrado como modelo de LLM")

# Agora importa e executa a interface Web
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Define os argumentos para o Streamlit
    trial_path = sys.argv[1] if len(sys.argv) > 1 else "tutorial/projeto_ollama/0"
    port = sys.argv[2] if len(sys.argv) > 2 else "8505"
    
    sys.argv = [
        "streamlit", "run",
        "autorag/web.py",
        "--server.headless", "true",
        "--server.port", port,
        "--",
        "--trial_path", trial_path
    ]
    
    sys.exit(stcli.main())
