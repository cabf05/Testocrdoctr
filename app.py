import os
import streamlit as st
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import logging
from pdf2image import convert_from_bytes
import io

# Configurações iniciais
st.set_page_config(
    page_title="DocTR OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização do estado da sessão
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'config' not in st.session_state:
    st.session_state.config = {
        "model_type": "accurate",
        "det_thresh": 0.5,
        "rec_thresh": 0.3,
        "enable_page_rotation": True
    }

@st.cache_resource
def load_doctr_model(model_type):
    """Carrega o modelo DocTR com cache"""
    logger.info(f"Carregando modelo {model_type}...")
    return ocr_predictor(
        det_arch='db_resnet50' if model_type == "accurate" else 'db_mobilenet_v3_large',
        reco_arch='crnn_vgg16_bn' if model_type == "accurate" else 'crnn_mobilenet_v3_small',
        pretrained=True
    )

def process_pdf(file_bytes):
    """Converte PDF para lista de imagens"""
    images = convert_from_bytes(
        file_bytes,
        dpi=300,
        poppler_path="/usr/bin"  # Caminho no Streamlit Cloud
    )
    return [np.array(img) for img in images]

def process_image(file_bytes):
    """Processa arquivo de imagem único"""
    image = Image.open(io.BytesIO(file_bytes))
    return [np.array(image)]

def extract_text(predictor, images):
    """Executa a extração de texto usando DocTR"""
    full_text = []
    for img in images:
        # Processar a imagem
        processed_doc = predictor([img])
        
        # Extrair texto organizado
        page_text = []
        for block in processed_doc.pages[0].blocks:
            for line in block.lines:
                words = [word.value for word in line.words]
                line_text = " ".join(words)
                page_text.append(line_text)
        
        full_text.append("\n".join(page_text))
    
    return "\n\n".join(full_text)

def main():
    st.title("📄 OCR com DocTR")
    st.markdown("Extraia texto de documentos PDF ou imagens usando IA")

    # Sidebar com configurações
    with st.sidebar:
        st.header("Configurações")
        st.session_state.config["model_type"] = st.selectbox(
            "Tipo de Modelo",
            ["accurate", "fast"],
            index=0,
            help="Escolha 'accurate' para melhor precisão ou 'fast' para velocidade"
        )
        
        st.session_state.config["det_thresh"] = st.slider(
            "Limiar de Detecção",
            0.1, 1.0, 0.5, 0.05,
            help="Limiar de confiança para detecção de áreas de texto"
        )
        
        st.session_state.config["rec_thresh"] = st.slider(
            "Limiar de Reconhecimento",
            0.1, 1.0, 0.3, 0.05,
            help="Limiar de confiança para reconhecimento de caracteres"
        )
        
        st.session_state.config["enable_page_rotation"] = st.checkbox(
            "Corrigir Orientação",
            value=True,
            help="Tentar corrigir automaticamente a orientação das páginas"
        )

    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Carregue seu documento (PDF ou imagem)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]
    )

    if uploaded_file is not None:
        if st.button("Processar Documento"):
            try:
                with st.spinner('Processando...'):
                    # Carregar modelo
                    predictor = load_doctr_model(st.session_state.config["model_type"])
                    
                    # Processar arquivo
                    file_bytes = uploaded_file.read()
                    
                    if uploaded_file.type == "application/pdf":
                        images = process_pdf(file_bytes)
                    else:
                        images = process_image(file_bytes)
                    
                    # Extrair texto
                    st.session_state.processed_text = extract_text(predictor, images)
                    
                st.success("Processamento concluído!")

            except Exception as e:
                st.error(f"Erro no processamento: {str(e)}")
                logger.error(traceback.format_exc())

    # Exibir resultados
    if st.session_state.processed_text:
        st.subheader("Texto Extraído")
        st.text_area("Resultado", st.session_state.processed_text, height=500)
        
        # Botão de download
        st.download_button(
            label="📥 Baixar Resultado",
            data=st.session_state.processed_text,
            file_name="texto_extraido.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
