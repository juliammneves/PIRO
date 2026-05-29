"""
PIRO - Detector de Fogo
Aplicacao Streamlit para classificacao binaria fire/no-fire.
FIAP Global Solution 2026 - Applied Computer Vision.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ============ CONFIGURACAO ============
st.set_page_config(
    page_title="PIRO - Detector de Fogo",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / 'modelo_final_piro.keras'
IMG_SIZE = (128, 128)
THRESHOLD = 0.5  # probabilidade >= threshold => classe "fogo"

# ============ CSS ============
st.markdown("""
<style>
.result-box {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin: 1.2rem 0;
}
.fire-box {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 2px solid #dc2626;
}
.no-fire-box {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #16a34a;
}
.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.result-fire { color: #991b1b; }
.result-no-fire { color: #166534; }
.confidence-text {
    font-size: 1.1rem;
    color: #374151;
}
.disclaimer-small {
    font-size: 0.85rem;
    color: #6b7280;
    font-style: italic;
    text-align: center;
    margin-top: 0.5rem;
}
.sidebar-section {
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ============ HELPERS ============
@st.cache_resource(show_spinner="Carregando modelo PIRO...")
def carregar_modelo():
    """Carrega o modelo treinado. @st.cache_resource garante load uma so vez."""
    if not MODEL_PATH.exists():
        st.error(f"Modelo nao encontrado em {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def pre_processar(imagem: Image.Image) -> np.ndarray:
    """Mesmo pipeline do treino: RGB -> resize 128x128 -> /255 -> batch dim."""
    img = imagem.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def prever(modelo, imagem: Image.Image) -> float:
    """Retorna probabilidade [0,1] da imagem conter fogo."""
    arr = pre_processar(imagem)
    return float(modelo.predict(arr, verbose=0)[0][0])


# ============ HEADER ============
st.title("🔥 PIRO — Detector de Fogo")
st.markdown(
    "**Plataforma Integrada de Resposta Orbital** · "
    "Classificação de imagens por CNN treinada do zero"
)
st.divider()


# ============ SIDEBAR ============
with st.sidebar:
    st.header("📊 Sobre o modelo")
    st.markdown("""
    **Arquitetura:** CNN profunda treinada do zero
    
    - 3 blocos convolucionais (Conv2D + BatchNorm + ReLU)
    - GlobalAveragePooling2D no classificador
    - **305.953 parâmetros**
    - Input: 128×128 RGB
    """)
    
    st.markdown("**Performance no teste:**")
    cm1, cm2 = st.columns(2)
    cm1.metric("Acurácia", "98.5%")
    cm2.metric("F1-score", "0.986")
    cm3, cm4 = st.columns(2)
    cm3.metric("Recall (fogo)", "99.3%")
    cm4.metric("Precisão", "97.9%")
    
    st.divider()
    
    st.markdown("**Dataset de treino:**")
    st.caption(
        "1.832 imagens balanceadas (50.7% fire / 49.3% non-fire), "
        "resolução uniforme 250×250. "
        "Augmentation: rotação ±15°, flip horizontal, brightness 0.85-1.15, zoom 0.15."
    )
    
    st.divider()
    
    with st.expander("⚠️ Limitações conhecidas", expanded=False):
        st.markdown("""
        - Treinado em **fotos terrestres** — performance reduzida em
          imagens satelitais reais (gap de domínio)
        - Dificuldade com fogo de **baixa intensidade** ou estágio inicial
        - Pode confundir pôr-do-sol e iluminação cênica com fogo
          (paleta cromática similar)
        - Não testado em imagens noturnas ou infravermelhas
        """)
    
    st.divider()
    st.caption("FIAP · GS 2026 · 4º ESW Presencial")


# ============ MAIN ============
st.subheader("📤 Envie uma imagem para classificação")

uploaded = st.file_uploader(
    "Formatos aceitos: JPG, JPEG, PNG, WebP",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="A imagem será redimensionada para 128×128 antes da classificação.",
)

if uploaded is not None:
    try:
        imagem = Image.open(uploaded)
    except Exception as e:
        st.error(f"Não foi possível abrir a imagem: {e}")
        st.stop()
    
    # Mostrar original + pre-processada lado a lado
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Imagem original**")
        st.image(imagem, use_container_width=True)
        st.caption(f"{imagem.size[0]}×{imagem.size[1]} px · {imagem.mode}")
    
    with col2:
        st.markdown("**Pré-processada (128×128)**")
        thumb = imagem.convert('RGB').resize(IMG_SIZE)
        st.image(thumb, use_container_width=True)
        st.caption("É essa versão que o modelo analisa.")
    
    # Predicao
    try:
        modelo = carregar_modelo()
        with st.spinner("Analisando imagem..."):
            prob = prever(modelo, imagem)
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        st.stop()
    
    classe_fogo = prob >= THRESHOLD
    confianca = prob if classe_fogo else (1 - prob)
    
    # Caixa de resultado colorida
    if classe_fogo:
        st.markdown(f"""
        <div class="result-box fire-box">
            <div class="result-title result-fire">🔥 FOGO DETECTADO</div>
            <div class="confidence-text">Confiança: <b>{confianca*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box no-fire-box">
            <div class="result-title result-no-fire">✓ SEM FOGO</div>
            <div class="confidence-text">Confiança: <b>{confianca*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Probabilidades detalhadas
    st.markdown("**Distribuição de probabilidade:**")
    pcol1, pcol2 = st.columns(2)
    pcol1.metric("Probabilidade de fogo", f"{prob*100:.1f}%")
    pcol2.metric("Probabilidade de sem fogo", f"{(1-prob)*100:.1f}%")
    
    st.progress(prob, text=f"Score do modelo: {prob:.4f}")
    
    st.markdown(
        '<div class="disclaimer-small">'
        'Predição binária baseada em threshold de 0.5. '
        f'Probabilidade ≥ 0.5 → classe "fogo".'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # Interpretacao honesta
    with st.expander("💡 Como interpretar este resultado"):
        st.markdown("""
        Este modelo é uma **prova de conceito acadêmica** treinada em ~1.500 imagens
        de fogo florestal terrestre. Use com cautela:
        
        - **Não substitui validação humana** em decisões críticas
        - Imagens **fora do domínio de treino** (satelitais, infravermelho, fogo de
          fogão/vela) podem ter classificação imprevisível
        - O modelo aprendeu features visuais (chama laranja + paisagem natural) —
          paisagens com paleta cromática similar (pôr-do-sol, falésias em sépia,
          florestas densas com luz dourada) podem gerar **falso-positivo**
        - **Probabilidades entre 0.4 e 0.6** indicam baixa confiança do modelo —
          revisar manualmente
        """)

else:
    # Estado inicial - sem upload
    st.info("👆 Envie uma imagem acima para iniciar a classificação.")
    
    with st.expander("ℹ️ Sobre o projeto PIRO"):
        st.markdown("""
        O **PIRO (Plataforma Integrada de Resposta Orbital)** é um sistema
        que integra dados orbitais, machine learning e automação para resposta
        rápida a queimadas no Brasil.
        
        Esta interface é a camada de **Visão Computacional** do projeto,
        responsável por classificar imagens como contendo fogo ou não.
        
        **Como ela se conecta ao PIRO completo:**
        
        1. Dados de focos de calor entram via APIs NASA FIRMS / Sentinel-2
        2. Imagens satelitais associadas são classificadas por este modelo
        3. Risco de propagação em 24h é previsto por outro modelo (com SHAP)
        4. Alertas são disparados via bot RPA pra brigadistas e órgãos ambientais
        5. Tudo orquestrado em pipeline Airflow + dashboard no Azure
        
        **ODS:** 13 (Ação Climática) · 9 (Inovação) · 11 (Cidades) · 15 (Vida Terrestre)
        """)
    
    with st.expander("🧪 Sugestões para testar"):
        st.markdown("""
        Tente imagens de:
        - **Queimadas florestais reais** (deve detectar com alta confiança)
        - **Florestas verdes** sem fogo (deve classificar como sem fogo)
        - **Pôr-do-sol em paisagem natural** (caso interessante — pode gerar falso-positivo
          devido à paleta quente)
        - **Imagens satelitais reais** (provável falso-negativo — limitação de domínio)
        - **Fogo de fogão ou vela** (fora do domínio — resultado imprevisível)
        """)


# ============ FOOTER ============
st.divider()
st.caption(
    "PIRO · Global Solution 2026 · Engenharia de Software FIAP · "
    "Modelo treinado do zero com TensorFlow/Keras"
)
