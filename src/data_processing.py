
from typing import Dict, List, Tuple
import json
import pickle
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Dataset HuggingFace — không cần tải file thủ công
HF_DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"

# Các nhãn mục tiêu (prefix của categories arXiv)
TARGET_CATEGORIES = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']


def get_primary_category(categories_str: str) -> str:
    """
    Lấy category chính từ chuỗi categories arXiv.
    Ví dụ: "cs.LG math.ST" -> "cs"
             "astro-ph.GA" -> "astro-ph"
             "cond-mat.supr-con" -> "cond-mat"
    """
    first = categories_str.strip().split()[0]  # lấy category đầu tiên
    # Xử lý dạng "astro-ph.XX" -> "astro-ph"
    if first.startswith("astro-ph"):
        return "astro-ph"
    if first.startswith("cond-mat"):
        return "cond-mat"
    # Các category dạng "cs.XX", "math.XX", "physics.XX"
    prefix = first.split(".")[0]
    return prefix


class DataProcessor:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_to_id = None
        self.id_to_label = None

    def load_raw_data(self) -> dict:
        """Load dataset trực tiếp từ HuggingFace (tự động tải về lần đầu)."""
        print(f"🔹 Đang tải dataset từ HuggingFace: {HF_DATASET_NAME}")
        print("   (Lần đầu chạy sẽ mất vài phút để tải ~3.8GB)")
        ds = load_dataset(HF_DATASET_NAME)
        print(f"✅ Đã tải dữ liệu: {len(ds['train']):,} mẫu")
        return ds

    def balanced_sampling(self, ds: dict, categories: List[str], samples_per_category: int = 700) -> List[dict]:
        """Lấy mẫu cân bằng từ dataset, lọc và chuẩn hóa nhãn."""
        print(f"🔹 Đang lấy mẫu cân bằng ({samples_per_category} mẫu/lĩnh vực)...")
        samples, counts = [], {cat: 0 for cat in categories}

        for row in ds['train']:
            # Dataset gốc có cột 'categories' (chuỗi) và 'abstract', 'title'
            raw_cat = row.get('categories', '')
            if not raw_cat:
                continue

            label = get_primary_category(raw_cat)
            if label not in categories or counts[label] >= samples_per_category:
                continue

            title = (row.get('title') or '').strip().replace('\n', ' ')
            abstract = (row.get('abstract') or '').strip().replace('\n', ' ')
            text = f"{title} {abstract}".strip()
            if not text:
                continue

            samples.append({'text': text, 'label': label})
            counts[label] += 1

            if all(v >= samples_per_category for v in counts.values()):
                break

        print(f"✅ Lấy mẫu xong: {len(samples)} mẫu — phân bố: {counts}")
        return samples

    def create_label_mapping(self, samples: List[dict]) -> Tuple[Dict, Dict]:
        labels = sorted({s['label'] for s in samples})
        self.label_to_id = {l: i for i, l in enumerate(labels)}
        self.id_to_label = {i: l for i, l in enumerate(labels)}
        print(f"✅ Đã mã hóa {len(labels)} nhãn: {self.label_to_id}")
        return self.label_to_id, self.id_to_label

    def split_data(self, samples: List[dict], test_size: float = 0.2, random_state: int = 42):
        X = [s['text'] for s in samples]
        y = [self.label_to_id[s['label']] for s in samples]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✅ Đã chia dữ liệu: {len(X_train)} train / {len(X_test)} test")
        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'label_to_id': self.label_to_id, 'id_to_label': self.id_to_label
        }
        with open(self.output_dir / "processed_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                'num_train': len(X_train),
                'num_test': len(X_test),
                'num_classes': len(self.label_to_id)
            }, f, indent=2, ensure_ascii=False)
        print("💾 Dữ liệu đã được lưu.")

    def load_processed_data(self):
        file = self.output_dir / "processed_data.pkl"
        if not file.exists():
            raise FileNotFoundError("❌ Không tìm thấy file dữ liệu đã xử lý. Hãy chạy data_processing.py trước.")
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.label_to_id = data['label_to_id']
        self.id_to_label = data['id_to_label']
        print("✅ Đã load dữ liệu đã xử lý.")
        return data


def main():
    processor = DataProcessor()
    ds = processor.load_raw_data()

    samples = processor.balanced_sampling(
        ds,
        categories=TARGET_CATEGORIES,
        samples_per_category=700  # Thay số mẫu mỗi category ở đây
    )
    processor.create_label_mapping(samples)
    X_train, X_test, y_train, y_test = processor.split_data(samples)
    processor.save_processed_data(X_train, X_test, y_train, y_test)
    print("🎯 Hoàn tất xử lý dữ liệu.")


if __name__ == "__main__":
    main()
