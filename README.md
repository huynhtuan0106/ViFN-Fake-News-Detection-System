# 🇻🇳 Vietnamese Fake News Detection System

## 🚀 BERT/PhoBERT Fine-tuning cho Phát hiện Tin giả Tiếng Việt

Hệ thống phát hiện tin giả tiếng Việt sử dụng **BERT/PhoBERT fine-tuning** với **multimodal fusion**, **domain regularization** và **SMOTETomek data balancing**.

---

## 🎯 **TỔNG QUAN HỆ THỐNG**

✅ **Đã thực hiện:**
- ✅ Fine-tune PhoBERT/BERT end-to-end cho tiếng Việt
- ✅ Multimodal fusion (title + summary + content + domain features)
- ✅ Domain regularization để tránh domain overfitting  
- ✅ 3 fusion strategies: Concat, Attention, Gated
- ✅ So sánh PhoBERT vs Multilingual BERT
- ✅ SMOTETomek data balancing (65:35 ratio)
- ✅ Mixed precision training (AMP) để tiết kiệm GPU memory
- ✅ Comprehensive evaluation và visualization

---

## 🏗️ **KIẾN TRÚC HỆ THỐNG**

```
📊 Data Loading & SMOTETomek Balancing
       ↓
🤖 PhoBERT Tokenization (Title, Summary, Content)
       ↓
🔥 Fine-tuned BERT Embedders (Separate for each modality)
       ↓
🌐 Domain Features Processing + Regularization
       ↓
🔀 Multimodal Fusion (Attention/Concat/Gated)
       ↓
🎯 Classification Head (2 classes: Real/Fake)
       ↓
📈 Results & Comprehensive Evaluation
```

---

## 📁 **CẤU TRÚC FILES VÀ VAI TRÒ**

### **🎯 Core Training Files**

| File | Vai trò | Mô tả |
|------|---------|-------|
| `main.py` | 🚪 **Main entry point** | Điểm khởi chạy chính, chọn experiment type |
| `bert_training.py` | 🏋️ **Single model training** | Training logic cho 1 model với cấu hình specific |
| `main_bert_experiment.py` | 🔬 **Model comparison** | So sánh multiple BERT models/fusion strategies |
| `bert_fine_tuner.py` | 🤖 **Model architectures** | Định nghĩa multimodal BERT fusion models |
| `bert_dataset.py` | 📊 **BERT dataset handling** | DataLoader và collate functions cho BERT |

### **📚 Data Processing Files**

| File | Vai trò | Mô tả |
|------|---------|-------|
| `data_loader.py` | 📥 **Data loading & preprocessing** | Load CSV, text processing, domain features |
| `data_balancer.py` | ⚖️ **SMOTETomek balancing** | Cân bằng dữ liệu 65:35 ratio |

### **⚙️ Configuration & Utilities**

| File | Vai trò | Mô tả |
|------|---------|-------|
| `config.py` | ⚙️ **System configuration** | GPU settings, BERT config, training parameters |
| `results_saver.py` | 💾 **Results management** | Lưu metrics, plots, comprehensive reports |

### **📋 Documentation**

| File | Vai trò | Mô tả |
|------|---------|-------|
| `README.md` | 📖 **Documentation** | Hướng dẫn sử dụng và cấu trúc hệ thống |
| `requirements_updated.txt` | 📦 **Dependencies** | Python packages cần thiết |

---

## 🚀 **CÁCH SỬ DỤNG**

### **Bước 1: Setup Environment**

```bash
# Tạo virtual environment
python -m venv fake_news_env
source fake_news_env/bin/activate  # Linux/Mac
# fake_news_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements_updated.txt
```

### **Bước 2: Cấu hình GPU (Quan trọng!)**

```bash
# Kiểm tra GPU và đề xuất cấu hình tối ưu
python gpu_optimization_guide.py
```

**Các cấu hình được đề xuất:**
- **4GB GPU**: batch_size=2, freeze 6 layers
- **6GB GPU**: batch_size=4, freeze 3 layers  
- **8GB GPU**: batch_size=8, no freezing (recommended)
- **12GB+ GPU**: batch_size=16+, full performance

### **Bước 3: Cập nhật đường dẫn dữ liệu**

Chỉnh sửa `main.py` dòng 125-126:

```python
REAL_FILE_PATH = "path/to/your/real_articles.csv"
FAKE_FILE_PATH = "path/to/your/fake_articles.csv"
```

**Format dữ liệu cần thiết:**
- `title_processed`: Tiêu đề đã xử lý
- `summary_processed`: Tóm tắt đã xử lý  
- `content_processed`: Nội dung đã xử lý
- `label`: 0 (Real) / 1 (Fake)
- `domain`: Tên miền (optional)

### **Bước 4: Chạy Training**

```bash
fake_news_env\Scripts\Activate.ps1
python main.py
```

**Chọn experiment type:**
- `1`: **Single Model Training** - Train 1 model với cấu hình tối ưu
- `2`: **Model Comparison** - So sánh 5 configurations khác nhau

---

## 🧪 **CÁC THỰC NGHIỆM**

### **1. Single Model Training**

Cấu hình mặc định (khuyến nghị):

```python
{
    'bert_model': 'vinai/phobert-base',
    'fusion_type': 'attention',
    'use_domain': True,
    'num_epochs': 3,
    'batch_size': 8,  # Tùy thuộc GPU
    'learning_rate': 2e-5,
    'balance_strategy': 'smotetomek'
}
```

### **2. Model Comparison**

So sánh 5 configurations tự động:

1. **PhoBERT_Concat**: Concat fusion
2. **PhoBERT_Attention**: Attention fusion + domain regularization ⭐
3. **PhoBERT_Gated**: Gated fusion
4. **MultiBERT_Attention**: Multilingual BERT comparison
5. **PhoBERT_Attention_NoDomain**: Test domain impact

---

## 🎯 **FUSION STRATEGIES**

### **1. Attention Fusion** ⭐ (Tốt nhất)
```python
# Learnable attention weights với domain regularization
weights = F.softmax(attention_weights, dim=0)
# Giới hạn domain weight ≤ 15% để tránh overfitting
if domain_weight > 0.15:
    penalty = (domain_weight - 0.15) ** 2
```

### **2. Concat Fusion**
```python
# Đơn giản nối các embeddings
fused = torch.cat([title_emb, summary_emb, content_emb, domain_emb], dim=1)
```

### **3. Gated Fusion**
```python
# Gating mechanism để kiểm soát information flow
gate_weights = sigmoid(gate_network(concat_embeddings))
fused = concat_embeddings * gate_weights
```

---

## 🎮 **GPU OPTIMIZATION & MEMORY MANAGEMENT**

### **⚡ Memory Optimization Features:**

1. **🛠️ Enhanced Memory Management**:
   - ✅ Automatic GPU cache clearing before evaluation
   - ✅ CPU fallback for model loading when GPU memory insufficient
   - ✅ Gradient checkpointing for memory efficiency
   - ✅ Expandable CUDA memory segments
   - ✅ Periodic memory cleanup during training

2. **🔧 Configuration Optimizations**:
   - ✅ Reduced max_length from 128 → 96 tokens
   - ✅ Increased gradient accumulation steps: 8 → 16
   - ✅ Mixed precision training (AMP)
   - ✅ Smart batch size recommendations by GPU memory

### **Key Factors Affecting GPU Memory:**

1. **Batch Size** (Quan trọng nhất):
   - Linear scaling: batch_size x2 → memory x2
   - Khuyến nghị: bắt đầu với 1, tăng dần theo GPU memory

2. **Sequence Length**:
   - Quadratic scaling: length x2 → memory x4
   - 96 tokens: cân bằng tốt cho 4GB GPU

3. **🆕 Memory Error Handling**:
   ```python
   # Automatic fallback to CPU if GPU memory insufficient
   try:
       checkpoint = torch.load(model_path, map_location='cpu')
       model.load_state_dict(checkpoint['model_state_dict'])
       model = model.to(DEVICE)
   except torch.cuda.OutOfMemoryError:
       # Fallback to CPU evaluation
       model = model.cpu()
   ```

### **🧪 Test Memory Optimizations:**

```bash
# Test các cải tiến memory
python test_memory_optimizations.py
```

### **⚙️ Environment Variables for Memory:**

```bash
# Thiết lập environment variable để tối ưu memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### **📊 GPU Memory Monitoring:**

Hệ thống tự động monitor và báo cáo:
- GPU memory usage trước/sau mỗi epoch
- Memory cleanup status
- Fallback warnings khi cần thiết

3. **Model Count**:
   - 3 BERT models (title, summary, content)
   - Shared weights để giảm memory

4. **Optimization Techniques**:
   - **Mixed Precision (AMP)**: -40% memory, +20% speed
   - **Gradient Checkpointing**: -30% memory, -20% speed
   - **Freeze Layers**: -50% memory, +30% speed

---

## 📊 **KẾT QUỢ MONG ĐỢI**

### **Performance Benchmarks:**
- **Accuracy**: 85-92%
- **F1-Score (Fake News)**: 83-90%
- **PhoBERT** > **Multilingual BERT** cho tiếng Việt
- **Attention Fusion** thường tốt nhất

### **Training Time:**
- **8GB GPU**: ~2-3 hours (batch_size=8)
- **12GB GPU**: ~1-2 hours (batch_size=16)
- **CPU only**: ~8-12 hours (không khuyến nghị)

### **Output Files:**
```
results_single_YYYYMMDD_HHMMSS/
├── bert_attention_best.pt              # Best model weights
├── bert_attention_comprehensive_results.png  # Evaluation plots
├── bert_attention_detailed_metrics.json      # Detailed metrics
├── bert_attention_summary_report.txt         # Human-readable report
├── training_history.png                      # Training curves
└── experiment_config.json                    # Experiment settings
```

---

## ⚡ **PERFORMANCE OPTIMIZATIONS**

### **Memory Optimizations:**
```python
# Trong config.py
GPU_CONFIG = {
    'mixed_precision': True,      # Enable AMP
    'pin_memory': True,           # Faster data transfer
    'prefetch_factor': 2,         # Background data loading
    'persistent_workers': True    # Reuse worker processes
}
```

### **Speed Optimizations:**
```python
# BERT caching
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Allow faster algorithms
```

---

## 🚨 **TROUBLESHOOTING**

### **CUDA Out of Memory:**
```bash
# 1. Giảm batch size
batch_size = 4  # instead of 8

# 2. Enable gradient checkpointing
gradient_checkpointing = True

# 3. Freeze BERT layers
freeze_bert_layers = 6

# 4. Reduce sequence length
max_length = 128  # instead of 256
```

### **Slow Training:**
```bash
# 1. Increase batch size (if memory allows)
batch_size = 16

# 2. Use mixed precision
mixed_precision = True

# 3. Increase num_workers
num_workers = 4
```

### **Poor Performance:**
```bash
# 1. Check data balancing
python -c "from data_balancer import *; check_balance_distribution()"

# 2. Verify domain regularization
domain_penalty_weight = 0.5

# 3. Try different fusion types
fusion_type = 'attention'  # usually best
```

---

## 🔧 **CUSTOM CONFIGURATIONS**

### **For Different Hardware:**

**Low Memory (≤6GB):**
```python
config = {
    'batch_size': 4,
    'max_length': 128,
    'mixed_precision': True,
    'freeze_bert_layers': 6,
    'gradient_checkpointing': True
}
```

**High Memory (≥16GB):**
```python
config = {
    'batch_size': 32,
    'max_length': 512,
    'mixed_precision': False,
    'freeze_bert_layers': 0,
    'gradient_checkpointing': False
}
```

### **For Different Datasets:**

**Small Dataset (<5K samples):**
```python
config = {
    'num_epochs': 5,
    'balance_strategy': 'weighted',
    'learning_rate': 1e-5  # Lower LR
}
```

**Large Dataset (>50K samples):**
```python
config = {
    'num_epochs': 2,
    'balance_strategy': 'smotetomek',
    'learning_rate': 3e-5  # Higher LR
}
```

---

## 📈 **MONITORING & DEBUGGING**

### **GPU Memory Monitoring:**
```python
# Trong training loop
if batch_idx % 100 == 0:
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

### **Training Progress:**
- Loss curves được lưu tự động
- Validation metrics mỗi epoch
- Best model checkpoint tự động

### **Performance Analysis:**
```bash
# Xem detailed results
python -c "from results_saver import *; analyze_results('results_dir')"
```

---

## 📞 **SUPPORT & FAQ**

### **Common Issues:**

**Q: GPU không được detect?**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: Training bị killed?**
```bash
# Reduce batch size hoặc enable swapping
batch_size = 2
```

**Q: Accuracy thấp?**
```bash
# Check data quality và balance ratio
python data_balancer.py
```

### **Best Practices:**
1. 🎯 Luôn bắt đầu với cấu hình conservative
2. 💾 Monitor GPU memory usage
3. 📊 Validate trên multiple runs
4. 🔄 Save checkpoints thường xuyên
5. 📈 Track experiments với tensorboard/wandb

---

## 🎯 **CONCLUSION**

### **Recommended Workflow:**
1. **Setup**: Check GPU → Install dependencies
2. **Config**: Run `gpu_optimization_guide.py` 
3. **Data**: Prepare CSV với format đúng
4. **Training**: Start với single model (option 1)
5. **Optimize**: Tăng batch size gradually
6. **Compare**: Chạy model comparison (option 2)
7. **Deploy**: Use best model cho inference

### **Key Success Factors:**
- ✅ **PhoBERT** tối ưu cho tiếng Việt
- ✅ **Attention fusion** với domain regularization
- ✅ **SMOTETomek balancing** 65:35 ratio
- ✅ **Mixed precision training** để tiết kiệm memory
- ✅ **Proper GPU configuration** theo hardware

---

**📝 Note**: Codebase này được thiết kế đặc biệt cho Vietnamese fake news detection với BERT fine-tuning. Mọi phương pháp TF-IDF cũ đã được loại bỏ hoàn toàn.

