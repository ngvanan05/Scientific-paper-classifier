# Hệ thống Phân loại Bài báo Khoa học

**[🇬🇧 English version](README.md)** | 🇻🇳 Tiếng Việt

> Phân loại tự động bài báo khoa học từ arXiv theo lĩnh vực nghiên cứu sử dụng Sentence Embeddings và Support Vector Machine.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Accuracy](https://img.shields.io/badge/Độ_chính_xác-87.57%25-2ea44f?style=flat)
![F1 Score](https://img.shields.io/badge/F1--Score-87.53%25-2ea44f?style=flat)
![License](https://img.shields.io/badge/Giấy_phép-MIT-blue?style=flat)

---

## Tóm tắt

Sự bùng nổ của các ấn phẩm khoa học trong những năm gần đây đặt ra nhu cầu cấp thiết về các hệ thống tổ chức tài liệu tự động. Đề tài này trình bày một pipeline phân loại bài báo khoa học từ kho lưu trữ arXiv vào năm lĩnh vực nghiên cứu chính, theo hướng tiếp cận nhẹ nhàng nhưng hiệu quả cao. Phương pháp sử dụng **sentence embeddings đa ngôn ngữ** (`intfloat/multilingual-e5-base`) làm biểu diễn đặc trưng, kết hợp với bộ phân lớp **Support Vector Machine (SVM)** nhân RBF. Kết quả thực nghiệm cho thấy biểu diễn ngữ nghĩa dày đặc vượt trội đáng kể so với phương pháp biểu diễn thưa truyền thống (Bag-of-Words), đạt **độ chính xác 87.57%** và **F1-score (weighted) 87.53%** trên tập kiểm thử cân bằng gồm 3.500 bài báo — vượt qua mục tiêu đề ra ban đầu là 85%.

---

## 1. Đặt vấn đề & Động lực

Tốc độ tăng trưởng của tài liệu khoa học đang ở mức chưa từng có. Riêng arXiv đã lưu trữ hơn **2,2 triệu preprint** trải dài trên hàng chục lĩnh vực nghiên cứu. Việc gán nhãn và phân loại thủ công các bài báo này tốn nhiều nhân lực và khó mở rộng quy mô. Hệ thống phân loại văn bản khoa học tự động có thể:

- **Tăng tốc khám phá tài liệu** cho các nhà nghiên cứu khi tiếp cận lĩnh vực mới
- **Làm nền tảng cho các tác vụ hạ nguồn** như hệ thống gợi ý bài báo, mạng trích dẫn
- **Đặt benchmark đánh giá** khả năng hiểu ngôn ngữ tự nhiên trong văn bản kỹ thuật chuyên sâu

Thách thức cốt lõi nằm ở chỗ ngôn ngữ khoa học cô đọng, chứa nhiều thuật ngữ chuyên ngành và thường mang tính liên ngành, khiến ngay cả các phương pháp NLP truyền thống cũng gặp khó khăn. Đề tài khảo sát mức độ hiệu quả của **bộ mã hóa câu đa ngôn ngữ tiền huấn luyện** so với phương pháp trích xuất đặc trưng cổ điển (Bag-of-Words) trên nhiều bộ phân lớp khác nhau, cung cấp cả kết quả thực nghiệm lẫn khả năng tái tạo đầy đủ.

---

## 2. Dữ liệu

### 2.1 Nguồn dữ liệu

| Trường | Chi tiết |
|--------|----------|
| **Bộ dữ liệu** | arXiv Abstracts (snapshot metadata arXiv) |
| **Nhà cung cấp** | UniverseTBD / Cornell University qua HuggingFace |
| **Liên kết** | [UniverseTBD/arxiv-abstracts-large](https://huggingface.co/datasets/UniverseTBD/arxiv-abstracts-large) |
| **Tổng số bài** | 2.292.057 |
| **Định dạng** | JSONL — mỗi dòng là một bài báo |
| **Các trường sử dụng** | `title`, `abstract`, `categories` |

### 2.2 Chiến lược chọn nhãn & lấy mẫu

Bộ dữ liệu gốc bao gồm **38 categories**. Để đảm bảo cân bằng nhãn và tập trung vào các lĩnh vực lớn nhất, chiến lược **lấy mẫu phân tầng** được áp dụng: **700 bài/category** được rút ngẫu nhiên, tạo thành tập huấn luyện cuối cùng gồm **3.500 mẫu**.

| Category | Mô tả | Kích thước gốc | Tỷ lệ |
|----------|-------|---------------|-------|
| `math` | Toán học | 461.568 | 20,14% |
| `cs` | Khoa học máy tính | 431.766 | 18,84% |
| `cond-mat` | Vật lý chất ngưng tụ | 297.127 | 12,96% |
| `astro-ph` | Vật lý thiên văn | 285.798 | 12,47% |
| `physics` | Vật lý tổng quát | 163.944 | 7,15% |

> **Lưu ý:** Bài báo có nhãn đa lĩnh vực (ví dụ: `"cs.LG math.ST"`) được gán về *category chính* (token đầu tiên).

### 2.3 Tiền xử lý dữ liệu

Pipeline tiền xử lý được áp dụng toàn bộ trên 2.292.057 bài trước khi lấy mẫu:
- Xóa ký tự đặc biệt, chữ số và khoảng trắng thừa
- Chuyển toàn bộ về chữ thường
- Trích xuất category chính từ chuỗi nhãn phức hợp
- Ghép đặc trưng: `text = title + " " + abstract`
- Kết quả lưu tại: `data_arxiv_preprocessed.jsonl` với hai trường `{text, label}`

---

## 3. Phương pháp

### 3.1 Trích xuất đặc trưng

Hai chiến lược trích xuất đặc trưng khác nhau được đánh giá trong thực nghiệm:

| Phương pháp | Mô tả | Số chiều |
|-------------|-------|----------|
| **Bag-of-Words (BoW)** | Binary `CountVectorizer`, tối đa 5.000 features, n-gram (1, 2) | 5.000 |
| **Sentence Embeddings** | `intfloat/multilingual-e5-base`, chuẩn hóa L2 | 768 |

**Tại sao chọn `intfloat/multilingual-e5-base`?**
Model này được ưu tiên so với các lựa chọn thay thế (ví dụ: `all-MiniLM-L6-v2`, `paraphrase-mpnet-base-v2`) vì:
1. **Hỗ trợ đa ngôn ngữ** — tóm tắt bài báo trên arXiv đôi khi chứa thuật ngữ hoặc tên tác giả phi tiếng Anh
2. **Biểu diễn ngữ nghĩa dày đặc** — E5 được huấn luyện có giám sát tường minh cho bài toán khớp văn bản, tạo ra không gian đặc trưng có ý nghĩa hình học rõ ràng hơn cho phân lớp
3. **Kích thước cân bằng** — 768 chiều cung cấp tín hiệu ngữ nghĩa mạnh trong khi vẫn phù hợp với phương pháp kernel SVM

**Tại sao chọn SVM thay vì fine-tune BERT?**
Với tập huấn luyện tương đối nhỏ (~3.500 mẫu), fine-tune toàn bộ bộ phân lớp dựa trên transformer có nguy cơ overfitting cao và đòi hỏi tài nguyên tính toán lớn hơn nhiều. SVM với hàm kernel phù hợp với bài toán có không gian đặc trưng chiều cao và quy mô dữ liệu vừa, đồng thời có đảm bảo lý thuyết vững chắc (bộ phân lớp maximum-margin). Kết hợp embedding tiền huấn luyện cố định + SVM là baseline tính toán hiệu quả và được kiểm chứng rộng rãi.

### 3.2 Pipeline phân lớp (Mô hình cuối cùng)

```
Đầu vào: Tiêu đề + Tóm tắt (văn bản thô)
        │
        ▼
Bộ mã hóa câu: intfloat/multilingual-e5-base
        │   → Vector 768 chiều, chuẩn hóa L2
        ▼
Bộ phân lớp SVM: kernel=RBF, C=10.0, gamma=0.1
        │
        ▼
Nhãn đầu ra: {astro-ph | cond-mat | cs | math | physics}
```

---

## 4. Thực nghiệm

### 4.1 So sánh 8 cấu hình

Tất cả các model được đánh giá trên tập kiểm thử giữ lại gồm **200 mẫu** (40 mẫu/category), sử dụng chiến lược chia phân tầng.

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
- Sentence Embeddings vượt trội Bag-of-Words ở **mọi** bộ phân lớp với biên độ **~8–10% accuracy**
- Các model dùng BoW bị **overfitting nặng** (gap F1 train–test lên đến 0.29), trong khi Embeddings có gap nhỏ hơn nhiều (~0.08), phản ánh chất lượng biểu diễn tốt hơn
- **SVM RBF + Embeddings** được chọn là cấu hình tối ưu

### 4.2 Tối ưu hóa siêu tham số

Grid search với **cross-validation 5-fold** được áp dụng cho cấu hình tốt nhất (SVM RBF + Embeddings):

```
Không gian tìm kiếm:
  C           : [0.1, 1.0, 10.0]
  gamma       : ['scale', 0.01, 0.1]
  class_weight: [None, 'balanced']

Tham số tốt nhất : C=10.0, gamma=0.1, class_weight=None
F1-score CV tốt nhất: 0.8414
```

| Metric | Trước Tuning | Sau Tuning | Δ |
|--------|:-----------:|:---------:|:---:|
| Accuracy | 0.840 | **0.845** | +0,5% |
| Precision | 0.849 | **0.854** | +0,5% |
| Recall | 0.840 | **0.845** | +0,5% |
| F1-Score | 0.838 | **0.843** | +0,5% |

### 4.3 Hiệu năng mô hình cuối cùng

Mô hình cuối được huấn luyện lại trên **toàn bộ 3.500 mẫu** với siêu tham số tốt nhất tìm được ở trên.

#### Chỉ số tổng thể

| Metric | Giá trị |
|--------|---------|
| Accuracy | **87,57%** |
| F1-Score (weighted) | **87,53%** |

#### Báo cáo phân lớp theo từng nhãn

| Category | Precision | Recall | F1-Score | Support |
|----------|:---------:|:------:|:--------:|:-------:|
| `astro-ph` | 0.93 | 0.94 | 0.93 | 140 |
| `cond-mat` | 0.85 | 0.83 | 0.84 | 140 |
| `cs` | 0.91 | 0.90 | 0.90 | 140 |
| `math` | 0.88 | 0.89 | 0.88 | 140 |
| `physics` | 0.81 | 0.82 | 0.81 | 140 |
| **weighted avg** | **0.88** | **0.88** | **0.88** | **700** |

> `astro-ph` đạt F1 cao nhất (0.93) nhờ vốn từ vựng chuyên ngành rất đặc thù (ví dụ: *redshift*, *quasar*, *stellar*). `physics` (tổng quát) là nhãn khó phân lớp nhất do chồng lấn ngữ nghĩa với `astro-ph` và `cond-mat`.

---

## 5. Demo Ứng dụng

Ứng dụng web **Streamlit** được cung cấp để thực hiện suy luận tương tác. Người dùng nhập tiêu đề và tóm tắt của bài báo để nhận kết quả phân loại tức thì.

![Demo App](reports/demo.png)

---

## 6. Cấu trúc Dự án

```
scientific-paper-classifier/
│
├── data/
│   ├── processed/
│   │   ├── data_arxiv_preprocessed.jsonl   # Corpus đã tiền xử lý toàn bộ
│   │   ├── processed_data.pkl              # Dữ liệu huấn luyện đã lấy mẫu & embedding
│   │   └── metadata.json                   # Metadata và thống kê quá trình chạy
│   └── unprocessed/
│       └── arxiv-metadata-oai-snapshot.json  # Dữ liệu gốc (tải thủ công)
│
├── models/
│   ├── svm_model.pkl                       # Bộ phân lớp SVM đã huấn luyện
│   ├── training_metrics.json               # Kết quả đánh giá
│   └── vectorizer_config.json              # Cấu hình embedding
│
├── notebooks/
│   ├── 1.0-EDA.ipynb                       # Phân tích khám phá dữ liệu
│   ├── 2.0-Data_preprocessing.ipynb        # Pipeline tiền xử lý toàn bộ
│   └── 3.0-model-prototyping.ipynb         # Thực nghiệm so sánh & tối ưu
│
├── src/
│   ├── data_processing.py                  # Tải & tiền xử lý dữ liệu
│   ├── train.py                            # Huấn luyện mô hình
│   ├── predict.py                          # Script suy luận
│   └── app.py                              # Ứng dụng web Streamlit
│
├── reports/
│   ├── demo.png
│   └── wordclouds/
│
├── requirements.txt
└── README.md
```

---

## 7. Word Clouds theo Lĩnh vực

Các hình ảnh trực quan được tạo trong quá trình EDA trên toàn bộ corpus 2,2 triệu bài báo, cung cấp bằng chứng định tính về từ vựng đặc trưng của từng lĩnh vực.

| astro-ph | cond-mat | cs |
|:---:|:---:|:---:|
| ![astro-ph](reports/wordclouds/wordcloud_astro-ph.png) | ![cond-mat](reports/wordclouds/wordcloud_cond-mat.png) | ![cs](reports/wordclouds/wordcloud_cs.png) |

| math | physics |
|:---:|:---:|
| ![math](reports/wordclouds/wordcloud_math.png) | ![physics](reports/wordclouds/wordcloud_physics.png) |

---

## 8. Giới hạn & Hướng phát triển

### Giới hạn hiện tại

- **Phạm vi nhãn hẹp**: Chỉ bao phủ 5 trong 38 categories của arXiv; các bài báo liên ngành bị ép về một nhãn duy nhất
- **Tập dữ liệu nhỏ**: 700 mẫu/category có thể chưa đại diện đầy đủ cho các chủ đề con hiếm trong mỗi lĩnh vực
- **Chưa đánh giá đa ngôn ngữ**: Mặc dù sử dụng bộ mã hóa đa ngôn ngữ, toàn bộ thực nghiệm chỉ dùng tóm tắt tiếng Anh
- **Embedding cố định**: Bộ mã hóa bị đóng băng; fine-tuning theo tác vụ (ví dụ: contrastive learning trên các cặp bài báo arXiv) có thể cải thiện chất lượng biểu diễn
- **Không có ước lượng độ không chắc chắn**: Mô hình chỉ đưa ra dự đoán cứng mà không có hiệu chỉnh độ tin cậy

### Hướng phát triển tiềm năng

- Fine-tune bộ mã hóa câu end-to-end bằng contrastive loss trên các cặp bài báo arXiv
- Mở rộng phân loại lên toàn bộ 38 categories theo kiến trúc phân cấp (hierarchical classification)
- Khám phá **SetFit** (few-shot fine-tuning) cho các lĩnh vực con ít dữ liệu
- Tích hợp với arXiv API để phân loại bài báo theo thời gian thực

---

## 9. Công nghệ Sử dụng

| Thành phần | Thư viện / Công cụ |
|------------|-------------------|
| **Bộ mã hóa câu** | `sentence-transformers` — `intfloat/multilingual-e5-base` |
| **Bộ phân lớp** | `scikit-learn` — `SVC` (nhân RBF) |
| **Tải dữ liệu** | `datasets` (HuggingFace) |
| **Xử lý dữ liệu** | `pandas`, `numpy` |
| **Trực quan hóa** | `matplotlib`, `seaborn`, `wordcloud` |
| **Ứng dụng web** | `streamlit` |

---

## 10. Cài đặt & Sử dụng

### Yêu cầu

- Python 3.8+
- Kết nối internet (lần chạy đầu tiên để tải embedding model và dataset)

### Cài đặt môi trường

```bash
git clone https://gitlab.com/vanan-portfolio/scientific-paper-classifier.git
cd scientific-paper-classifier
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Hướng A — Dùng mô hình có sẵn (Khuyến nghị)

```bash
cd src
streamlit run app.py
```

Truy cập `http://localhost:8501`, nhập tiêu đề và tóm tắt bài báo, sau đó nhấn **🔍 Phân tích văn bản**.

### Hướng B — Huấn luyện lại từ đầu

> Xóa `models/svm_model.pkl`, `data/processed/processed_data.pkl` và `data/processed/metadata.json` trước khi chạy.

**Bước 1 — Xử lý dữ liệu** (tự động tải ~3,8 GB từ HuggingFace lần đầu):
```bash
cd src
python data_processing.py
```

**Bước 2 — Huấn luyện mô hình:**
```bash
python train.py
```

**Bước 3 — Đánh giá mô hình:**
```bash
python predict.py
```

**Bước 4 — Chạy ứng dụng:**
```bash
streamlit run app.py
```

---

## 11. Tài liệu tham khảo

1. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. https://arxiv.org/abs/1908.10084

2. Wang, L., Yang, N., Huang, X., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv preprint. https://arxiv.org/abs/2212.03533

3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297.

4. Clement, C. B., Bierbaum, M., O'Keeffe, K. P., & Alemi, A. A. (2019). *On the Use of arXiv as a Dataset*. arXiv preprint. https://arxiv.org/abs/1905.00075

5. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020 (System Demonstrations). https://arxiv.org/abs/1910.03771

---

## Giấy phép

Dự án được phân phối theo giấy phép MIT. Xem chi tiết tại [LICENSE](LICENSE).
