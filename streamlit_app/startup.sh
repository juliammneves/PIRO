#!/bin/bash
# Startup script para Azure App Service.
# Azure passa a porta no $PORT (geralmente 8000).
# --server.address 0.0.0.0 e obrigatorio pra aceitar conexoes externas.

python -m streamlit run app.py \
    --server.port=${PORT:-8000} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
