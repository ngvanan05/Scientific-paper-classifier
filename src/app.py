
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from predict import Predictor

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Text Classification System",
    page_icon="📊",
    layout="centered"
)

# --- Tiêu đề ---
st.title("📊 Text Classification System")
st.markdown("### Phân loại chủ đề văn bản khoa học sử dụng SVM & Sentence Embeddings")
st.markdown("---")

# --- Mô tả chi tiết ---
st.markdown("#### 📖 Giới thiệu")
st.markdown("""
Ứng dụng này sử dụng mô hình **SVM (Support Vector Machine)** kết hợp với **Sentence Embeddings** 
để phân loại văn bản khoa học theo lĩnh vực. Các lĩnh vực bao gồm:

* 🪐 **astro-ph** – Vật lý thiên văn
* ⚛️ **cond-mat** – Vật lý chất ngưng tụ
* 💻 **cs** – Khoa học máy tính
* 📐 **math** – Toán học
* 🔬 **physics** – Vật lý nói chung
""")
st.markdown("---")

# --- Tải mô hình ---
@st.cache_resource
def load_predictor():
    try:
        return Predictor()
    except FileNotFoundError as e:
        return str(e)

predictor = load_predictor()
if isinstance(predictor, str):
    st.error(predictor)
    st.stop()

# --- Hướng dẫn ---
st.markdown("#### 📝 Hướng dẫn sử dụng")
st.info("Nhập một hoặc nhiều đoạn văn bản khoa học (mỗi dòng một đoạn), hệ thống sẽ phân tích và dự đoán lĩnh vực tương ứng.")

# --- Nhập văn bản ---
st.markdown("#### ✍️ Nhập văn bản")
text_input = st.text_area(
    label="Text Input",
    height=200,
    placeholder="Example: This paper proposes a new deep learning approach for image classification...",
    label_visibility="collapsed"
)

# --- Nút dự đoán ---
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Phân tích văn bản", use_container_width=True, type="primary")

if predict_button:
    if not text_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản để tiếp tục")
    else:
        # Tách các dòng văn bản
        texts = [t.strip() for t in text_input.split("\n") if t.strip()]

        with st.spinner("⏳ Đang xử lý và phân tích..."):
            preds = predictor.predict(texts)

        st.success(f"✅ Hoàn tất! Đã phân tích {len(texts)} đoạn văn bản")
        
        st.markdown("---")
        st.markdown("#### 📋 Kết quả phân loại")

        # Hiển thị kết quả
        for i, (text, label) in enumerate(zip(texts, preds), 1):
            with st.container():
                st.markdown(f"**Đoạn văn {i}:**")
                st.markdown(f"> {text}")
                st.markdown(f"**Lĩnh vực:** :blue[{label}]")
                if i < len(texts):
                    st.divider()