
import json
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class Predictor:
    """Lớp xử lý việc load model, vectorizer và thực hiện dự đoán"""
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir = base_dir / "models"

        model_path = self.model_dir / "svm_model.pkl"
        config_path = self.model_dir / "vectorizer_config.json"

        if not model_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy model tại: {model_path}\nHãy chạy train.py trước.")
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy vectorizer config tại: {config_path}\nHãy chạy train.py trước.")

        print("Loading trained SVM model...")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print("Loading vectorizer config...")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load label_map từ config (được lưu lúc train).
        # Fallback về mặc định nếu dùng model cũ chưa có trường này.
        raw_map = config.get("id_to_label", {
            "0": "astro-ph", "1": "cond-mat",
            "2": "cs", "3": "math", "4": "physics"
        })
        self.label_map = {int(k): v for k, v in raw_map.items()}

        print(f"Loading embedding model: {config['model_name']}")
        self.vectorizer = SentenceTransformer(config["model_name"])
        self.normalize = config.get("normalize", True)
        print("✅ All models loaded successfully!")

    def encode_text(self, texts):
        """Chuyển văn bản sang vector embeddings.
        Dùng prefix 'query:' theo đúng chuẩn của multilingual-e5 cho inference.
        """
        # Chuẩn hóa: xóa newline như lúc preprocessing, rồi thêm prefix
        formatted = [f"query: {t.strip().replace(chr(10), ' ')}" for t in texts]
        emb = self.vectorizer.encode(
            formatted,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(formatted) > 1  # chỉ hiện progress bar khi batch > 1
        )
        return emb

    def predict(self, texts, return_label_names=True):
        """Trả về nhãn dự đoán (có thể là chỉ số hoặc tên nhãn)"""
        if isinstance(texts, str):
            texts = [texts]

        # Lọc bỏ các text rỗng hoặc None, giữ index để map kết quả về
        valid_texts = []
        valid_indices = []
        for i, t in enumerate(texts):
            if t and isinstance(t, str) and t.strip():
                valid_texts.append(t)
                valid_indices.append(i)

        if not valid_texts:
            return []

        emb = self.encode_text(valid_texts)
        preds = self.model.predict(emb)

        if return_label_names:
            return [self.label_map[int(p)] for p in preds]
        return preds.tolist()


def main():
    """Chạy thử dự đoán"""
    predictor = Predictor()

    samples = [
        "This paper proposes a new deep learning approach for image classification."
    ]

    preds = predictor.predict(samples)
    print("testing prediction...")
    for text, label in zip(samples, preds):
        print(f"\n Text: {text}\n Predicted label: {label}")


if __name__ == "__main__":
    main()
