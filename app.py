import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
import tempfile
from PIL import Image

# Configuração da página do Streamlit
st.set_page_config(page_title="OCR com DocTR", page_icon="📝", layout="wide")

# Função para processar a imagem com DocTR
def process_image(image):
    try:
        # Carrega o modelo OCR
        model = ocr_predictor(pretrained=True)
        
        # Converte a imagem para formato compatível
        doc = DocumentFile.from_images(image)
        
        # Realiza a predição do OCR
        result = model(doc)
        
        # Extrai o texto da saída do modelo
        extracted_text = "\n".join([word[0] for block in result.pages for line in block.blocks for word in line.words])
        
        return extracted_text if extracted_text else "Nenhum texto detectado."
    
    except Exception as e:
        return f"Erro ao processar a imagem: {str(e)}"

# Interface do Streamlit
st.title("📄 Extração de Texto com DocTR")
st.markdown("Faça o upload de uma imagem para extrair o texto usando **DocTR**.")

# Upload de arquivo
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png", "tiff", "bmp"])

if uploaded_file:
    # Exibe a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_column_width=True)

    # Processa a imagem
    with st.spinner("Processando..."):
        extracted_text = process_image(np.array(image))
    
    st.subheader("📄 Texto Extraído")
    st.text_area("Resultado", extracted_text, height=300)

    # Botão para baixar o texto extraído
    st.download_button(
        label="📥 Baixar Texto",
        data=extracted_text,
        file_name="texto_extraido.txt",
        mime="text/plain"
    )
