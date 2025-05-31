# Dự án Phát hiện Tin tức Giả tiếng Việt bằng Late Fusion Architecture

## Tổng quan dự án
Dự án này sử dụng kỹ thuật **Late Fusion** để phân tích và phát hiện tin tức giả tiếng Việt thông qua việc kết hợp thông tin từ nhiều trường dữ liệu khác nhau: `title`, `summary`, `content`, `domain`.

## Cấu trúc thư mục
```
FakeNewsDetection/
├── config.py                  # Cấu hình toàn dự án
├── data_loader.py             # Load và xử lý dữ liệu 
├── embeddings.py              # Tạo embeddings cho từng trường
├── models.py                  # Định nghĩa các mô hình fusion
├── dataset.py                 # Dataset và DataLoader
├── training.py                # Utilities cho training
├── main_real_data.py          # Script thực thi
├── requirements_updated.txt   # Dependencies 
└── README.md                  # File Readme
```

### **Bước 1: Cài đặt dependencies**
```bash
pip install -r requirements_updated.txt
```

### **Bước 2: Chuẩn bị dữ liệu**
Đảm bảo bạn có 2 file CSV với format:
```
domain,title,publish_date,summary,content_html,label
```

### **Bước 3: Cập nhật đường dẫn**
Sửa file `main_real_data.py` dòng 370-371:
```python
REAL_FILE_PATH = "đường/dẫn/tới/file_tin_thật.csv"  
FAKE_FILE_PATH = "đường/dẫn/tới/file_tin_giả.csv"  
```

### **Bước 4: Chạy thí nghiệm**
```bash
python main_real_data.py
```

## Chi tiết từng file

### 1. `requirements_updated.txt` - Dependencies
**Mục đích:** Danh sách thư viện Python cần thiết (đã dọn sạch duplicate)

**Cách sử dụng:**
```bash
pip install -r requirements_updated.txt
```

### 2. `config.py` - File cấu hình
**Mục đích:** Tập trung tất cả các tham số cấu hình

**Chức năng chính:**
- **DEVICE:** Cấu hình GPU/CPU
- **MODEL_CONFIG:** Tham số mô hình (hidden_dim, learning_rate, epochs...)
- **TEXT_CONFIG:** Tham số xử lý văn bản (max_features, ngram_range...)
- **TRUSTED_DOMAINS:** Danh sách domain tin cậy
- **DATA_SPLIT:** Tỷ lệ chia train/test/validation

### 3. `data_loader.py` - 
**Mục đích:** Xử lý dữ liệu

**Chức năng chính:**
- **Load 2 file CSV riêng biệt:** real vs fake news
- **Xử lý format đúng:** `domain,title,publish_date,summary,content_html,label`
- **Làm sạch HTML:** Parse `content_html` thành text thuần
- **Tiền xử lý tiếng Việt:** Tokenization, normalization
- **Domain processing:** Credibility scoring và one-hot encoding
- **Data validation:** Kiểm tra missing values, format

**Với dữ liệu của bạn:**
```python
# File real news: label tự động = 0
# File fake news: label tự động = 1
# content_html được tự động parse thành content text
```

### 4. `embeddings.py` - Tạo vector đặc trưng
**Mục đích:** Chuyển đổi văn bản thành vector số

**Các class chính:**
- **`TFIDFEmbedder`:** TF-IDF vectors (nhanh)
- **`BERTEmbedder`:** BERT embeddings (chậm nhưng tốt hơn)

**Lưu ý:** Có thể dùng simulated BERT để test nhanh

### 5. `models.py` - Mô hình Late Fusion
**Mục đích:** Implement 3 kiến trúc fusion khác nhau

#### `LateFusionModel` - Mô hình chính
**3 fusion strategies:**
- **Concat Fusion:** Nối đơn giản các vector
- **Attention Fusion:** Học trọng số cho từng trường  
- **Gated Fusion:** Dùng gate để kiểm soát thông tin

### 6. `dataset.py` - PyTorch Dataset
**Mục đích:** Tạo PyTorch Dataset và DataLoader với stratified splitting

### 7. `training.py` - Training utilities
**Mục đích:** Functions hỗ trợ training và evaluation

### 8. `main_real_data.py` - Script CHÍNH

**Workflow:**
1. Load dữ liệu từ 2 file CSV
2. Preprocess và validate
3. Tạo embeddings (TF-IDF hoặc BERT)
4. Train 3 mô hình fusion
5. So sánh và tìm mô hình tốt nhất
6. Tạo visualizations
7. Lưu kết quả và báo cáo

## Kết quả mong đợi

### **Console Output:**
```
================================================================================
VIETNAMESE FAKE NEWS DETECTION - LATE FUSION ARCHITECTURE
================================================================================

🏆 BEST MODEL: Attention Fusion
📊 Best Accuracy: 0.8750
📈 Best AUC-ROC: 0.9320
🎯 Best F1-Score: 0.8690
```

### **Files được tạo:**
- `experiment_results/` folder với:
  - `attention_fusion_model.pth`: Mô hình tốt nhất
  - `BEST_MODEL_attention_fusion.pth`: Backup mô hình tốt nhất
  - `fusion_results_comparison.png`: Biểu đồ so sánh
  - `experiment_report.txt`: Báo cáo chi tiết

## Tùy chỉnh tham số

### **Trong `config.py`:**
```python
# Tăng epochs
MODEL_CONFIG['num_epochs'] = 25

# Điều chỉnh learning rate  
MODEL_CONFIG['learning_rate'] = 0.0005

# Sử dụng BERT thay vì TF-IDF
# Trong main_real_data.py: use_bert=True
```

### **Thêm domain tin cậy:**
```python
TRUSTED_DOMAINS = [
    'vnexpress.net', 'tuoitre.vn', 'vietnamnet.vn', 
    'dantri.com.vn', 'thanhnien.vn', 'zing.vn',
    'your-domain.vn'  
]
```

## Troubleshooting

### **Lỗi thường gặp:**

1. **FileNotFoundError:** Kiểm tra đường dẫn file trong `main_real_data.py`
2. **CUDA out of memory:** Giảm `batch_size` trong `config.py`
3. **Missing columns:** Kiểm tra format CSV có đúng không
4. **UnicodeDecodeError:** File CSV phải encoding UTF-8

### **Kiểm tra dữ liệu:**
```python
# Test data loader
python data_loader.py

# Check file format
import pandas as pd
df = pd.read_csv("your_file.csv")
print(df.columns.tolist())
print(df.head())
```

## Mở rộng

### **Thêm fusion method mới:**
```python
# Trong models.py
class CustomFusionModel(LateFusionModel):
    def _apply_fusion(self, title_emb, summary_emb, content_emb, domain_emb):
        # Implement custom fusion logic
        pass
```

### **Hyperparameter tuning:**
```python
# Sử dụng Optuna
import optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # Training và return accuracy
```

---

**Nếu gặp vấn đề:**
1. Đọc `USAGE_GUIDE.md` để biết hướng dẫn chi tiết
2. Kiểm tra format dữ liệu CSV
3. Test từng module riêng: `python data_loader.py`
4. Kiểm tra GPU memory: `nvidia-smi`

