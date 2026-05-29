# PIRO — App de Demonstração (Streamlit)

Aplicação web que carrega o modelo CNN treinado e classifica imagens como
contendo fogo ou não. Demonstração Funcional da
entrega ACV do projeto PIRO — FIAP Global Solution 2026.

## Estrutura

```
streamlit_app/
├── app.py                    # Aplicação Streamlit principal
├── modelo_final_piro.keras   # Pesos do modelo CNN2 (~3.7 MB)
├── requirements.txt          # Dependências Python
├── startup.sh                # Script de inicialização (Azure App Service)
├── .streamlit/
│   └── config.toml           # Tema e configurações do Streamlit
└── README.md                 # Este arquivo
```

## Rodando localmente

```bash
# 1. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate              # Linux/macOS
# venv\Scripts\activate                # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar
streamlit run app.py
```

A aplicação abre em `http://localhost:8501`.

## Sobre o modelo

| Item | Valor |
|------|-------|
| Arquitetura | CNN profunda (3 blocos Conv2D + BN + ReLU, GAP, Dense) |
| Parâmetros | 305.953 |
| Input | RGB 128×128 |
| Output | Sigmoide (probabilidade de fogo) |
| Acurácia (teste) | **98.55%** |
| F1-score | **0.986** |
| Recall (fogo) | **99.29%** |
| Precisão | 97.89% |
| Dataset | 1.832 imagens balanceadas (brsdincer/wildfire-detection-image-data) |

Modelo treinado do zero, sem uso de transfer learning ou pesos pré-treinados,
conforme exigido pelo edital da disciplina ACV.

## Limitações conhecidas

- **Domínio de treino restrito**: o modelo foi treinado em fotografias
  terrestres de incêndios florestais. Performance em imagens satelitais
  reais é limitada (gap de domínio identificado na análise de falsos negativos).
- **Fogo de baixa intensidade**: queimadas em estágio inicial, com muita
  fumaça e pouca chama visível, podem ser classificadas incorretamente
  como sem fogo (1 falso negativo em 140 imagens de teste).
- **Falsos positivos em paletas quentes**: paisagens com tonalidade laranja
  ou sépia dominante (falésias em canyon, El Capitan/Yosemite com luz dourada,
  florestas densas com iluminação atmosférica quente) podem ser confundidas
  com fogo (3 falsos positivos em 135 imagens de teste).

## Contexto do projeto PIRO

A **Plataforma Integrada de Resposta Orbital (PIRO)** é a solução
integrada da equipe para a Global Solution 2026 da FIAP, tema
*Indústria Espacial: O Código que Move o Universo*.

Esta aplicação é a camada de **Visão Computacional** que classifica
tiles candidatos a foco de fogo no pipeline:

```
NASA FIRMS / Sentinel-2 → Airflow (BDDI) → CNN (ACV) → Risco 24h (GAIE) → Alerta (RPA)
```

**ODS atendidos:** 13 (Ação Climática) · 9 (Inovação) · 11 (Cidades) · 15 (Vida Terrestre)

## Integrantes da equipe

| Nome completo | RM |
|---------------|-----|
| Júlia Marques | 98680 |
| Matheus Gusmão | 550826 |
| Guilherme Morais | 551981 |
