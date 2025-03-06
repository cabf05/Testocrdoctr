import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
import torch
from PIL import Image

# Verifica se há suporte para CUDA (GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega o modelo de OCR do DocTR
@st.cache_resource
def load_model():
    return ocr_predictor(pretrained=True).to(DEVICE)

model = load_model()

# Configuração da interface
st.title("OCR com DocTR")
st.write("Envie uma imagem ou um arquivo PDF para extração de texto.")

# Upload de arquivo
uploaded_file = st.file_uploader("Escolha um arquivo (PDF ou Imagem)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.type

    # Processa PDFs
    if file_type == "application/pdf":
        st.write("Processando PDF...")
        doc = DocumentFile.from_pdf(uploaded_file)

    # Processa imagens
    else:
        st.write("Processando Imagem...")
        doc = DocumentFile.from_images(uploaded_file)

    # Aplica OCR
    with st.spinner("Extraindo texto..."):
        result = model(doc)

    # Exibe texto extraído
    text = "\n".join([word for block in result.pages for word in block.lines])
    st.subheader("Texto Extraído:")
    st.text_area("Resultado:", text, height=300)

    # Exibe as imagens processadas
    st.subheader("Visualização do Documento:")
    for page in doc:
        img = Image.fromarray(np.array(page))
        st.image(img, caption="Página Processada", use_container_width=True)  # Corrigido o parâmetro
