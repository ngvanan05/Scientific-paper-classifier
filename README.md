# 📊 Hệ thống Phân loại Văn bản Khoa học arXiv

Hệ thống Machine Learning phân loại tự động các bài báo từ **arXiv** theo lĩnh vực nghiên cứu, kết hợp **Sentence Embeddings** (`intfloat/multilingual-e5-base`) và **SVM** để đạt hiệu quả cao trong việc hiểu và gán nhãn văn bản khoa học.

---

## 🎯 Mục tiêu

- Phân loại văn bản vào **5 lĩnh vực chính** của khoa học
- Đạt **độ chính xác trên 85%** trên tập kiểm thử
- Cung cấp **ứng dụng web trực quan** dễ sử dụng

---

## 📁 Cấu trúc Dự án

```
MIDTERM_PROJECT/
│
├── data/
│   ├── processed/              # Dữ liệu đã xử lý
│   │   ├── data_arxiv_preprocessed.jsonl
│   │   ├── processed_data.pkl
│   │   └── metadata.json
│   └── unprocessed/            # Dữ liệu gốc (tải thủ công)
│       └── arxiv-metadata-oai-snapshot.json
│
├── models/                     # Model đã huấn luyện
│   ├── svm_model.pkl
│   ├── training_metrics.json
│   └── vectorizer_config.json
│
├── notebooks/                  # Jupyter Notebooks phân tích
│   ├── 1.0-EDA.ipynb
│   ├── 2.0-Data_preprocessing.ipynb
│   └── 3.0-model-prototyping.ipynb
│
├── src/                        # Source code
│   ├── data_processing.py      # Xử lý dữ liệu
│   ├── train.py                # Huấn luyện model
│   ├── predict.py              # Dự đoán
│   └── app.py                  # Streamlit web app
│
├── reports/
│   ├── demo.png
│   └── wordclouds/
│
├── README.md
└── requirements.txt
```

---

## 🖼️ Demo Ứng dụng

![Demo App](reports/demo.png)

---

## 🏷️ Các Lĩnh vực Phân loại

| Nhãn | Mô tả | Ví dụ |
|------|-------|-------|
| `astro-ph` | Vật lý thiên văn | Nghiên cứu về sao, thiên hà, vũ trụ |
| `cond-mat` | Vật lý chất ngưng tụ | Siêu dẫn, từ tính, vật liệu nano |
| `cs` | Khoa học máy tính | AI, thuật toán, lập trình |
| `math` | Toán học | Đại số, giải tích, hình học |
| `physics` | Vật lý nói chung | Cơ học, nhiệt động lực học, quang học |

---

## �️ Nguồn Dữ liệu

Dữ liệu sử dụng trong dự án là **arXiv Dataset** — bộ dữ liệu metadata của hơn 2 triệu bài báo khoa học từ nền tảng [arXiv.org](https://arxiv.org).

- **Nguồn**: [HuggingFace — UniverseTBD/arxiv-abstracts-large](https://huggingface.co/datasets/UniverseTBD/arxiv-abstracts-large)
- **Nhà cung cấp**: UniverseTBD / Cornell University
- **Định dạng**: JSON (mỗi dòng là một bài báo)
- **Các trường sử dụng**: `title`, `abstract`, `categories`
- **Số lượng mẫu huấn luyện**: ~3,500 bài (700 mỗi lĩnh vực)
- **Tải dữ liệu**: Tự động qua thư viện `datasets` — **không cần tải thủ công**

### Phân bố dữ liệu gốc (top 5 categories được chọn)

| Category | Số bài | Tỷ lệ |
|----------|--------|--------|
| math | 461,568 | 20.14% |
| cs | 431,766 | 18.84% |
| cond-mat | 297,127 | 12.96% |
| astro-ph | 285,798 | 12.47% |
| physics | 163,944 | 7.15% |

> Dataset gốc có 38 categories, dự án chọn 5 categories lớn nhất và lấy mẫu cân bằng 700 bài/category.

---

## 🔬 Phân tích & Thực nghiệm (Notebooks)

### Notebook 1 — EDA (`1.0-EDA.ipynb`)

Khám phá dataset gốc với 2,292,057 bài báo:
- Kiểm tra missing values: **0 abstract thiếu, 0 category thiếu**
- Phân tích phân bố 38 categories
- Tạo Word Clouds cho từng lĩnh vực

### Word Clouds theo lĩnh vực

| astro-ph | cond-mat | cs |
|:---:|:---:|:---:|
| ![astro-ph](reports/wordclouds/wordcloud_astro-ph.png) | ![cond-mat](reports/wordclouds/wordcloud_cond-mat.png) | ![cs](reports/wordclouds/wordcloud_cs.png) |

| math | physics |
|:---:|:---:|
| ![math](reports/wordclouds/wordcloud_math.png) | ![physics](reports/wordclouds/wordcloud_physics.png) |

### Notebook 2 — Data Preprocessing (`2.0-Data_preprocessing.ipynb`)

Pipeline tiền xử lý toàn bộ 2,292,057 bài:
- Xóa ký tự đặc biệt, số, khoảng trắng thừa
- Chuyển về chữ thường
- Trích xuất category chính từ chuỗi categories (vd: `"cs.LG math.ST"` → `"cs"`)
- Gộp `title + abstract` thành trường `text`
- Kết quả: file `data_arxiv_preprocessed.jsonl` với 2 trường `text` và `label`

### Notebook 3 — Model Prototyping (`3.0-model-prototyping.ipynb`)

Thực nghiệm so sánh **2 phương pháp vector hóa** × **4 model** = **8 tổ hợp**.

#### Phương pháp vector hóa

| Phương pháp | Mô tả | Số chiều |
|-------------|-------|----------|
| **Bag-of-Words (Count)** | Binary CountVectorizer, max 5000 features, ngram (1,2) | 5,000 |
| **Sentence Embeddings** | `intfloat/multilingual-e5-base`, L2 normalized | 768 |

#### Kết quả so sánh 8 models (200 mẫu test)

| Model | Vector hóa | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|----------|-----------|--------|----------|
| **SVM RBF** | **Embeddings** | **0.840** | **0.849** | **0.840** | **0.838** |
| SVM Linear | Embeddings | 0.825 | 0.830 | 0.825 | 0.823 |
| Logistic Regression | Embeddings | 0.790 | 0.780 | 0.790 | 0.783 |
| KNN | Embeddings | 0.775 | 0.781 | 0.775 | 0.763 |
| Logistic Regression | Count | 0.755 | 0.768 | 0.755 | 0.758 |
| SVM RBF | Count | 0.715 | 0.728 | 0.715 | 0.710 |
| SVM Linear | Count | 0.705 | 0.718 | 0.705 | 0.708 |
| KNN | Count | 0.695 | 0.693 | 0.695 | 0.677 |

**Nhận xét:**
- Sentence Embeddings vượt trội hoàn toàn so với Bag-of-Words ở mọi model (~8-10% accuracy)
- Các model dùng Count Vectorizer đều bị **overfitting nặng** (gap F1 lên đến 0.29), trong khi Embeddings có gap nhỏ hơn nhiều (~0.08)
- SVM RBF + Embeddings là tổ hợp tốt nhất

#### Hyperparameter Tuning (GridSearchCV, 5-fold CV)

Tuning SVM RBF với Embeddings:

```
Param grid: C=[0.1, 1.0, 10.0], gamma=['scale', 0.01, 0.1], class_weight=[None, 'balanced']
Best params: C=10.0, gamma=0.1, class_weight=None
Best CV F1-score: 0.8414
```

| Metric | Trước Tuning | Sau Tuning |
|--------|-------------|-----------|
| Accuracy | 0.840 | **0.845** |
| Precision | 0.849 | **0.854** |
| Recall | 0.840 | **0.845** |
| F1-Score | 0.838 | **0.843** |

---

## � Kết quả Mô hình Cuối cùng

Model được train lại trên toàn bộ ~3,500 mẫu (700/category) với hyperparameters tốt nhất:

| Metric | Giá trị |
|--------|---------|
| Accuracy | **87.57%** |
| F1-Score (weighted) | **87.53%** |

### Kiến trúc Pipeline

```
Title + Abstract
      ↓
Sentence Embedding (intfloat/multilingual-e5-base, 768 chiều, L2 normalized)
      ↓
SVM Classifier (kernel=RBF, C=10.0, gamma=0.1)
      ↓
Label (astro-ph / cond-mat / cs / math / physics)
```

---

## 🛠️ Công nghệ Sử dụng

| Nhóm | Thư viện |
|------|---------|
| **Embedding** | `sentence-transformers` — `intfloat/multilingual-e5-base` |
| **Classifier** | `scikit-learn` — SVC (RBF kernel) |
| **Data** | `datasets` (HuggingFace), `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn`, `wordcloud` |
| **App** | `streamlit` |

---

## 📦 Cài đặt & Sử dụng

### Yêu cầu

- Python 3.8+
- Kết nối internet (lần đầu chạy để tải embedding model và dataset)

### 1. Clone và cài đặt

```bash
git clone <repo-url>
cd MIDTERM_PROJECT
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Hướng 1: Dùng Model Có Sẵn (Nhanh nhất)

```bash
cd src
streamlit run app.py
```

Truy cập `http://localhost:8501`, nhập văn bản và nhấn **🔍 Phân tích văn bản**.

---

### Hướng 2: Train Lại Từ Đầu

> Xóa `models/svm_model.pkl`, `data/processed/processed_data.pkl` và `data/processed/metadata.json` trước khi chạy.

**Bước 1 — Xử lý dữ liệu** (tự động tải ~3.8GB từ HuggingFace lần đầu):

```bash
cd src
python data_processing.py
```

**Bước 2 — Huấn luyện model:**

```bash
python train.py
```

**Bước 3 — Kiểm tra model:**

```bash
python predict.py
```

**Bước 4 — Chạy ứng dụng:**

```bash
streamlit run app.py
```
