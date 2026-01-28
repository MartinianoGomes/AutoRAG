# Script para executar RAG completo com geração de texto
# Execute este script no PowerShell

# 1. Configure sua chave da API OpenAI aqui:
$env:OPENAI_API_KEY = "sua-chave-aqui"

# 2. Ativa o ambiente virtual
Write-Host "Ativando ambiente virtual..." -ForegroundColor Green
& "$PSScriptRoot/../.venv/Scripts/Activate.ps1"

# 3. Navega para o diretório do projeto
Set-Location "$PSScriptRoot/.."

# 4. Cria o projeto completo com geração
Write-Host "Executando avaliação do RAG completo..." -ForegroundColor Green
Write-Host "Isso pode levar alguns minutos e consumir créditos da API OpenAI." -ForegroundColor Yellow

autorag evaluate `
    --config tutorial/config_completo_memoria.yaml `
    --qa_data_path tests/resources/qa_data_sample.parquet `
    --corpus_data_path tests/resources/corpus_data_sample.parquet `
    --project_dir tutorial/projeto_completo

Write-Host "`n✅ Avaliação concluída!" -ForegroundColor Green
Write-Host "Agora execute o servidor web com:" -ForegroundColor Cyan
Write-Host "  streamlit run autorag/web.py --server.headless true -- --trial_path tutorial/projeto_completo/0" -ForegroundColor White
