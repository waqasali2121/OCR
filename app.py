import io
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="OCR Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI OCR Text Extractor")
st.caption("Upload an image or PDF and extract editable text with AI-powered OCR.")

# ---------- Helpers ----------
@st.cache_resource
def get_client(api_key: str):
    return InferenceClient(token=api_key)

def pdf_to_images(pdf_bytes: bytes, dpi: int = 180):
    """Render each PDF page as a PIL image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)

    doc.close()
    return images

def ocr_image(client, image: Image.Image, model: str):
    """
    Send an image to a Hugging Face OCR model.
    The returned object may vary slightly by model/provider, so the
    function handles the common text/image-to-text response shapes.
    """
    result = client.image_to_text(image=image, model=model)

    if isinstance(result, str):
        return result.strip()

    if hasattr(result, "text"):
        return (result.text or "").strip()

    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict):
                parts.append(str(item.get("generated_text", item.get("text", ""))))
            elif hasattr(item, "generated_text"):
                parts.append(str(item.generated_text))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(x for x in parts if x).strip()

    return str(result).strip()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    hf_key = st.text_input(
        "Hugging Face API token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Create a Hugging Face token with inference access and paste it here.",
    )

    model = st.text_input(
        "OCR model",
        value="microsoft/trocr-base-printed",
        help="You can change this to another Hugging Face image-to-text model.",
    )

    dpi = st.slider(
        "PDF rendering quality (DPI)",
        min_value=120,
        max_value=250,
        value=180,
        step=10,
    )

    st.divider()
    st.markdown(
        "**Supported input:** JPG, JPEG, PNG, WEBP, BMP, TIFF and PDF."
    )

# ---------- Upload ----------
uploaded = st.file_uploader(
    "Upload your document",
    type=["pdf", "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
)

if uploaded is None:
    st.info("⬆️ Upload a PDF or image to begin.")
    st.stop()

if not hf_key:
    st.warning("Please enter your Hugging Face API token in the sidebar.")
    st.stop()

if st.button("🚀 Extract Text", type="primary", use_container_width=True):
    client = get_client(hf_key)
    file_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()

    try:
        if suffix == ".pdf":
            with st.spinner("Rendering PDF pages..."):
                images = pdf_to_images(file_bytes, dpi=dpi)
        else:
            images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

        st.success(f"Loaded {len(images)} page(s)/image(s).")

        all_text = []

        progress = st.progress(0)
        status = st.empty()

        for index, image in enumerate(images, start=1):
            status.write(f"🔎 OCR processing page {index} of {len(images)}...")
            text = ocr_image(client, image, model)

            if not text:
                text = "[No text detected on this page.]"

            all_text.append(f"--- Page {index} ---\n{text}")
            progress.progress(index / len(images))

        final_text = "\n\n".join(all_text)

        st.session_state["ocr_text"] = final_text
        st.session_state["ocr_filename"] = uploaded.name

        status.success("✅ OCR completed.")

    except Exception as exc:
        st.error(
            "OCR failed. Check your Hugging Face token, model name, "
            f"network access, and model/provider availability.\n\nDetails: {exc}"
        )

# ---------- Results ----------
if "ocr_text" in st.session_state:
    st.divider()
    st.subheader("📝 Extracted Text")

    edited_text = st.text_area(
        "Review and edit the extracted text",
        value=st.session_state["ocr_text"],
        height=500,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Download TXT",
            data=edited_text.encode("utf-8"),
            file_name=f"{Path(st.session_state['ocr_filename']).stem}_ocr.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "📋 Download Markdown",
            data=edited_text.encode("utf-8"),
            file_name=f"{Path(st.session_state['ocr_filename']).stem}_ocr.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.caption(
        "Tip: For scanned PDFs, higher DPI can improve OCR accuracy but may "
        "increase processing time and API usage."
    )
