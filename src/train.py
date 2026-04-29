
import pickle
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


class EmbeddingVectorizer:
    """Tạo sentence embeddings từ văn bản"""
    def __init__(self, model_name='intfloat/multilingual-e5-base', normalize=True):
        print(f"🔹 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        print("✅ Embedding model loaded.")

    def transform(self,
                  texts: List[str],
                  mode: str = 'passage',
                  show_progress: bool = True,
                  batch_size: int = 64) -> np.ndarray:
        """
        Chuyển đổi texts thành embeddings với hiển thị tiến trình
        """
        if mode in ['passage', 'query']:
            formatted_texts = [f"{mode}: {text.strip()}" for text in texts]
        else:
            formatted_texts = texts
        
        print(f"\n📦 Encoding {len(formatted_texts)} texts (batch size={batch_size})...")
        embeddings = []
        
        # Hiển thị tiến trình bằng tqdm
        for i in tqdm(range(0, len(formatted_texts), batch_size), desc="🔄 Generating embeddings", ncols=100):
            batch = formatted_texts[i:i+batch_size]
            batch_emb = self.model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False  # tắt thanh mặc định của SentenceTransformer
            )
            embeddings.append(batch_emb)
        
        embeddings = np.vstack(embeddings)
        print(f"✅ Embeddings created! Shape: {embeddings.shape}")
        return embeddings


class ModelTrainer:
    """Quản lý quá trình training mô hình"""
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        self.data_file = base_dir / "data" / "processed" / "processed_data.pkl"
        self.model_dir = base_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.vectorizer = None
        self.classifier = None

    def load_data(self):
        if not self.data_file.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file: {self.data_file}\nHãy chạy data_processing.py trước.")
        with open(self.data_file, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Data loaded ({len(data['X_train'])} train, {len(data['X_test'])} test).")
        return data

    def create_embeddings(self, X_train, X_test):
        self.vectorizer = EmbeddingVectorizer()
        X_train_emb = self.vectorizer.transform(X_train)
        X_test_emb = self.vectorizer.transform(X_test)
        print(f"✅ Embeddings created: {X_train_emb.shape[1]} dimensions.")
        return X_train_emb, X_test_emb

    def train(self, X_train_emb, y_train):
        self.classifier = SVC(kernel='rbf', C=10.0, gamma=0.1, class_weight=None, random_state=42)
        self.classifier.fit(X_train_emb, y_train)
        print("✅ Model trained successfully.")

    def evaluate(self, X_test_emb, y_test):
        y_pred = self.classifier.predict(X_test_emb)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"🎯 Test Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        return {"accuracy": acc, "f1": f1}

    def save_model(self, metrics, data):
        model_path = self.model_dir / "svm_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        vectorizer_config = {
            "model_name": "intfloat/multilingual-e5-base",
            "normalize": self.vectorizer.normalize,
            "id_to_label": {str(k): v for k, v in data['id_to_label'].items()}
        }
        with open(self.model_dir / "vectorizer_config.json", "w") as f:
            json.dump(vectorizer_config, f, indent=2)

        with open(self.model_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Model + metrics saved to: {self.model_dir}")


def main():
    print("\n🚀 TRAINING PIPELINE START\n")

    trainer = ModelTrainer()
    data = trainer.load_data()

    X_train_emb, X_test_emb = trainer.create_embeddings(data['X_train'], data['X_test'])
    trainer.train(X_train_emb, data['y_train'])
    metrics = trainer.evaluate(X_test_emb, data['y_test'])
    trainer.save_model(metrics, data)

    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
