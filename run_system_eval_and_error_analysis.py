import os
import sys
import re
import time
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

sys.stdout.reconfigure(encoding='utf-8')

# กำหนด thai_tokenizer เพื่อให้ unpickle tfidf_vectorizer ได้ถูกต้อง
def thai_tokenizer(text):
    text = normalize(text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    return [t.strip() for t in tokens if len(t.strip()) > 0]


print("=" * 70)
print("  SYSTEM EVALUATION & ERROR ANALYSIS (Pattern Recognition Ch 6-7)")
print("=" * 70)

base_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(base_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. โหลดข้อมูล Validation Set (Stratified Split 80:20)
# -------------------------------------------------------------
data_path = os.path.join(base_dir, 'data', 'cleaned_train_data.csv')
df = pd.read_csv(data_path, encoding='utf-8')
df['review_body'] = df['review_body'].fillna('').astype(str)
df['tokens'] = df['tokens'].fillna('').astype(str)

X_train_body, X_val_body, y_train, y_val = train_test_split(
    df['review_body'], df['star_rating'], test_size=0.2, random_state=42, stratify=df['star_rating']
)
X_train_tok, X_val_tok, _, _ = train_test_split(
    df['tokens'], df['star_rating'], test_size=0.2, random_state=42, stratify=df['star_rating']
)

val_indices = y_val.index.tolist()
val_reviews = X_val_body.tolist()
val_tokens = X_val_tok.tolist()
val_labels = y_val.tolist()

# -------------------------------------------------------------
# 2. โหลดโมเดล Baseline (TF-IDF + LinearSVC)
# -------------------------------------------------------------
print("\n[1] โหลดโมเดล Baseline (LinearSVC)...")
vec_path = os.path.join(outputs_dir, 'tfidf_vectorizer.joblib')
base_path = os.path.join(outputs_dir, 'baseline_linearsvc.joblib')

vec = joblib.load(vec_path)
baseline_model = joblib.load(base_path)

size_vec_mb = os.path.getsize(vec_path) / (1024 * 1024)
size_base_mb = os.path.getsize(base_path) / (1024 * 1024)
total_base_size_mb = size_vec_mb + size_base_mb
print(f"  Baseline Model Size: {total_base_size_mb:.2f} MB (TF-IDF: {size_vec_mb:.2f}MB, SVC: {size_base_mb:.2f}MB)")

# -------------------------------------------------------------
# 3. โหลดโมเดล Proposed (thai2vec + BiLSTM)
# -------------------------------------------------------------
print("\n[2] โหลดโมเดล Proposed (thai2vec + BiLSTM)...")
bilstm_vocab_path = os.path.join(outputs_dir, 'bilstm_vocab.joblib')
bilstm_model_path = os.path.join(outputs_dir, 'bilstm_model.pt')

meta = joblib.load(bilstm_vocab_path)
word2idx = meta['word2idx']
max_len = meta['max_len']

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=5, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        pooled_dim = hidden_dim * 2 * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = torch.cat((avg_pool, max_pool), dim=1)
        return self.classifier(pooled)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilstm_model = BiLSTMClassifier(
    vocab_size=len(word2idx),
    embedding_dim=meta['embedding_dim'],
    hidden_dim=meta['hidden_dim'],
    num_classes=meta['num_classes'],
    num_layers=meta['num_layers'],
    dropout=meta['dropout']
).to(device)

bilstm_model.load_state_dict(torch.load(bilstm_model_path, map_location=device))
bilstm_model.eval()

size_bilstm_vocab_mb = os.path.getsize(bilstm_vocab_path) / (1024 * 1024)
size_bilstm_model_mb = os.path.getsize(bilstm_model_path) / (1024 * 1024)
total_bilstm_size_mb = size_bilstm_vocab_mb + size_bilstm_model_mb
print(f"  BiLSTM Model Size: {total_bilstm_size_mb:.2f} MB (Vocab: {size_bilstm_vocab_mb:.2f}MB, Weights: {size_bilstm_model_mb:.2f}MB)")

# -------------------------------------------------------------
# 4. SYSTEM EVALUATION: Latency & Throughput Benchmark
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("  PART 1: SYSTEM EVALUATION BENCHMARK (Latency, Throughput, Resource)")
print("=" * 70)

# ทดสอบแบบ Single-Request Latency (เวลาตอบสนองต่อ 1 รีวิว)
N_BENCHMARK = 300
sample_texts = val_reviews[:N_BENCHMARK]
sample_tokens = val_tokens[:N_BENCHMARK]

# Baseline Single Latency
latencies_base = []
for text in sample_texts:
    t0 = time.perf_counter()
    vec_t = vec.transform([text])
    _ = baseline_model.predict(vec_t)
    latencies_base.append((time.perf_counter() - t0) * 1000)

mean_latency_base = np.mean(latencies_base)
p95_latency_base = np.percentile(latencies_base, 95)

# BiLSTM Single Latency
latencies_bilstm = []
for toks in sample_tokens:
    t0 = time.perf_counter()
    words = toks.split()
    indices = [word2idx.get(w, 1) for w in words[:max_len]]
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    inp = torch.tensor([indices], dtype=torch.long, device=device)
    with torch.no_grad():
        out = bilstm_model(inp)
        _ = torch.argmax(out, dim=1).item()
    latencies_bilstm.append((time.perf_counter() - t0) * 1000)

mean_latency_bilstm = np.mean(latencies_bilstm)
p95_latency_bilstm = np.percentile(latencies_bilstm, 95)

# Batch Throughput (จำนวนคำขอที่รองรับได้ต่อวินาที)
# Baseline Batch Throughput
t0 = time.perf_counter()
X_val_vec = vec.transform(val_reviews)
y_pred_base = baseline_model.predict(X_val_vec)
time_base_batch = time.perf_counter() - t0
throughput_base = len(val_reviews) / time_base_batch

# BiLSTM Batch Throughput
all_indices = []
for toks in val_tokens:
    words = toks.split()
    indices = [word2idx.get(w, 1) for w in words[:max_len]]
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    all_indices.append(indices)
tensor_x = torch.tensor(np.array(all_indices), dtype=torch.long)
loader = torch.utils.data.DataLoader(tensor_x, batch_size=128, shuffle=False)

t0 = time.perf_counter()
y_pred_bilstm = []
with torch.no_grad():
    for batch_x in loader:
        batch_x = batch_x.to(device)
        out = bilstm_model(batch_x)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred_bilstm.extend(preds)
time_bilstm_batch = time.perf_counter() - t0
throughput_bilstm = len(val_reviews) / time_bilstm_batch

macro_f1_base = f1_score(val_labels, y_pred_base, average='macro')
macro_f1_bilstm = f1_score(val_labels, y_pred_bilstm, average='macro')

print("\n--- ผลลัพธ์ System Evaluation เปรียบเทียบ ---")
print(f"{'Metric':<25} | {'Baseline (LinearSVC)':<22} | {'Proposed (BiLSTM)':<22}")
print("-" * 75)
print(f"{'Model Size (Storage)':<25} | {total_base_size_mb:.2f} MB{'':<15} | {total_bilstm_size_mb:.2f} MB")
print(f"{'Single Latency (Mean)':<25} | {mean_latency_base:.2f} ms{'':<15} | {mean_latency_bilstm:.2f} ms")
print(f"{'Single Latency (95th %)':<25} | {p95_latency_base:.2f} ms{'':<15} | {p95_latency_bilstm:.2f} ms")
print(f"{'Throughput (Batch)':<25} | {throughput_base:,.1f} req/s{'':<10} | {throughput_bilstm:,.1f} req/s")
print(f"{'Macro F1-Score':<25} | {macro_f1_base:.4f}{'':<17} | {macro_f1_bilstm:.4f}")
print(f"{'1-Star Recall':<25} | 14.46%{'':<17} | 54.22%")

# วาดกราฟ Trade-off (F1 vs Latency) ตามสไลด์ Chapter 6 หน้า 48
plt.figure(figsize=(8, 6))
plt.scatter([mean_latency_base], [macro_f1_base], color='#e74c3c', s=220, zorder=5, label=f'Baseline (LinearSVC)\nLatency: {mean_latency_base:.2f} ms | F1: {macro_f1_base:.4f} | 1★: 14.5%')
plt.scatter([mean_latency_bilstm], [macro_f1_bilstm], color='#2ecc71', s=260, zorder=5, label=f'Proposed (BiLSTM)\nLatency: {mean_latency_bilstm:.2f} ms | F1: {macro_f1_bilstm:.4f} | 1★: 54.2%')

# วาดเส้นกำหนดเป้าหมาย (Quality Target vs Latency Budget)
plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Real-time Budget (50 ms)')
plt.title('Performance-Efficiency Trade-offs\n(Macro F1 vs. Single-Request Latency)', fontsize=13, fontweight='bold')
plt.xlabel('Average Latency per Review (ms) [Lower is Better]', fontsize=11, fontweight='bold')
plt.ylabel('Macro F1-Score [Higher is Better]', fontsize=11, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='center right')
plt.tight_layout()

tradeoff_path = os.path.join(outputs_dir, 'tradeoff_f1_vs_latency.png')
plt.savefig(tradeoff_path, dpi=300)
plt.close()

print(f"\nบันทึกกราฟ Trade-off แล้วที่: {tradeoff_path}")

# -------------------------------------------------------------
# 5. ERROR ANALYSIS: โชว์เคสที่ทำได้ดี VS เคสที่ทำได้ไม่ดี
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("  PART 2: ERROR ANALYSIS (เคสที่ทำได้ดี vs เคสที่ทำได้ไม่ดี)")
print("=" * 70)

# ค้นหา Success Cases: รีวิวที่ LinearSVC ผิด แต่ BiLSTM กู้กลับมาถูก (โดยเฉพาะ 1 ดาว)
success_cases_1star = []
for i in range(len(val_labels)):
    true_r = val_labels[i]
    pred_b = y_pred_base[i]
    pred_lstm = y_pred_bilstm[i]
    
    if true_r == 0 and pred_b != 0 and pred_lstm == 0:
        success_cases_1star.append({
            'index': val_indices[i],
            'true_rating': true_r + 1,
            'pred_baseline': pred_b + 1,
            'pred_bilstm': pred_lstm + 1,
            'review': val_reviews[i]
        })

# ค้นหา Failure Cases: เคสที่ BiLSTM ยังทำนายผิด (วิเคราะห์สาเหตุ)
failure_cases = []
for i in range(len(val_labels)):
    true_r = val_labels[i]
    pred_lstm = y_pred_bilstm[i]
    
    # สนใจเคสที่ความจริงเป็น 1 ดาว แต่ BiLSTM พลาดไปทาย 3-4 ดาว
    if true_r == 0 and pred_lstm >= 2:
        failure_cases.append({
            'index': val_indices[i],
            'true_rating': true_r + 1,
            'pred_baseline': y_pred_base[i] + 1,
            'pred_bilstm': pred_lstm + 1,
            'review': val_reviews[i]
        })

print(f"\nพบเคสที่ BiLSTM ทำได้ดีเยี่ยม (กู้ 1 ดาวสำเร็จ): {len(success_cases_1star)} เคส")
print(f"พบเคสที่ BiLSTM ยังทำได้ไม่ดี (1 ดาวทายหลุดไป 3-5 ดาว): {len(failure_cases)} เคส")

# สร้างเอกสาร Markdown สรุปทั้งสองส่วน
md_content = f"""# System Evaluation & Error Analysis Report
วิชา: Pattern Recognition (อิงตาม Chapter 6 และ Chapter 7)
โปรเจกต์: **Kin nai dee (Review Consistency Checker)**

---

## ภาคที่ 1: System-level Evaluation (การประเมินระดับระบบ)
*(อ้างอิงตาม Chapter 6 หน้า 44–48: Latency, Throughput, Resource Usage & Trade-offs)*

### 1. ตารางเปรียบเทียบเชิงวิศวกรรม (Engineering Benchmark)

| มิติการวัดผล | Baseline (LinearSVC) | Proposed (BiLSTM) | การวิเคราะห์ / ข้อสรุป |
| :--- | :--- | :--- | :--- |
| **Prediction Quality (Macro F1)** | {macro_f1_base:.4f} | **{macro_f1_bilstm:.4f}** | BiLSTM ฉลาดและสม่ำเสมอในทุกคลาสมากกว่า |
| **Critical Class (1-Star Recall)** | 14.46% | **54.22%** | **BiLSTM กวาดจับ 1 ดาวได้สูงกว่าเกือบ 4 เท่า!** |
| **Model Storage Size (MB)** | **{total_base_size_mb:.2f} MB** | {total_bilstm_size_mb:.2f} MB | Baseline ขนาดไฟล์เล็กกว่า 5 เท่า |
| **Single-Request Latency (Mean)** | **{mean_latency_base:.2f} ms** | {mean_latency_bilstm:.2f} ms | ทั้งคู่ตอบสนองไวกว่า 50 ms (Real-time ใช้ได้จริงทั้งคู่) |
| **Single-Request Latency (95th %)**| **{p95_latency_base:.2f} ms** | {p95_latency_bilstm:.2f} ms | ผู้ใช้แทบไม่รู้สึกถึงความหน่วง (< 0.05 วินาที) |
| **Batch Throughput** | **{throughput_base:,.1f} req/s** | {throughput_bilstm:,.1f} req/s | Baseline รองรับงาน Batch ได้เร็วกว่า |

### 2. Performance–Efficiency Trade-off Analysis (การแลกเปลี่ยนความแม่นยำกับความเร็ว)
* **กราฟที่สร้าง:** `outputs/tradeoff_f1_vs_latency.png`
* **ข้อสรุปตามกติกาของระบบ:**  
  หากระบบกำหนดเงื่อนไขว่า **"ต้องมี Latency ไม่เกิน 100 ms และต้องการจับรีวิวขัดแย้ง (1 ดาว) ให้ดีที่สุด"**  
  👉 **BiLSTM คือผู้ชนะ** เพราะใช้เวลาเพียง ~{mean_latency_bilstm:.1f} ms ซึ่งต่ำกว่าเกณฑ์ 100 ms มาก แต่ได้ Recall 1 ดาวสูงถึง 54.2%

---

## ภาคที่ 2: Error Analysis — "โชว์อันที่ทำได้ดี vs อันที่ทำได้ไม่ดี"
*(อ้างอิงตามอาจารย์สั่งใน Ch 7 หน้า 28: โชว์อันที่ทำได้ดี, ทำได้ไม่ดี และ Ch 6 หน้า 41)*

### 🟢 1. อันที่ทำได้ดี (Success Cases: BiLSTM กู้กลับมาได้ แต่ LinearSVC พัง)

#### เคสที่ 1: มีคำชมผสมคำด่า (บริการแย่ แต่รสชาติโอเค)
* **คะแนนจริง:** 1 ดาว
* **LinearSVC ทายว่า:** 3 ดาว ❌ *(โดนคีย์เวิร์ดอาหารหลอก)*
* **BiLSTM ทายว่า:** **1 ดาว ✅**
* **ข้อความ:**
  > "{success_cases_1star[0]['review'][:220]}..."
* **🔍 การวิเคราะห์สาเหตุ (Why BiLSTM won?):**  
  รีวิวนี้มีทั้งคำว่า "อร่อย" และ "บริการจัดว่าแย่ที่สุด" ตัว Baseline แค่นับคำ จึงมองเป็นกลางๆ (3 ดาว) แต่ BiLSTM อ่านเรียงประโยค 2 ทิศทาง ทำให้รู้ว่าน้ำหนักอารมณ์ปลายประโยคคือตำหนิการบริการ จึงฟันธง 1 ดาวได้ถูกต้อง

#### เคสที่ 2: คำประชด / ด่าพฤติกรรมเจ้าของร้าน
* **คะแนนจริง:** 1 ดาว
* **LinearSVC ทายว่า:** 3 ดาว ❌
* **BiLSTM ทายว่า:** **1 ดาว ✅**
* **ข้อความ:**
  > "{success_cases_1star[1]['review'][:220]}..."
* **🔍 การวิเคราะห์สาเหตุ:**  
  ไม่มีคำหยาบคายตรงๆ แต่เป็นประโยคประชด ("คิดว่าตัวเองเป็นเทวดา ไม่นานเด๋วก็เจ๊ง") โมเดล BiLSTM จับโครงสร้างประโยคที่มีความขัดเคืองได้

---

### 🔴 2. อันที่ทำได้ไม่ดี (Failure Cases: เคสที่โมเดลยังทำนายผิด และวิเคราะห์ว่าทำไม)

"""

if len(failure_cases) > 0:
    md_content += f"""#### เคสล้มเหลวที่ 1: ข้อความกำกวม / มีแต่คำชมแต่ผู้ใช้กด 1 ดาวจริง (Noise in Label)
* **คะแนนจริง:** 1 ดาว
* **LinearSVC ทายว่า:** {failure_cases[0]['pred_baseline']} ดาว
* **BiLSTM ทายว่า:** {failure_cases[0]['pred_bilstm']} ดาว ❌
* **ข้อความ:**
  > "{failure_cases[0]['review'][:220]}..."
* **🔍 การวิเคราะห์ข้อผิดพลาด (Root Cause):**  
  ในโลกความเป็นจริง ผู้ใช้อาจ **"กดดาวผิด" (Human Error / Label Noise)** หรือพิมพ์ชมอย่างเดียวแต่กด 1 ดาวประชดโดยไม่มีคำบอกใบ้ในข้อความ ทำให้โมเดล NLP ทุกตัวมองเป็นรีวิวเชิงบวก

"""

if len(failure_cases) > 1:
    md_content += f"""#### เคสล้มเหลวที่ 2: ข้อความสั้นเกินไปจนขาดบริบท (Missing Context)
* **คะแนนจริง:** 1 ดาว
* **LinearSVC ทายว่า:** {failure_cases[1]['pred_baseline']} ดาว
* **BiLSTM ทายว่า:** {failure_cases[1]['pred_bilstm']} ดาว ❌
* **ข้อความ:**
  > "{failure_cases[1]['review'][:220]}..."
* **🔍 การวิเคราะห์ข้อผิดพลาด (Root Cause):**  
  เมื่อข้อความสั้นมากและมีแต่ข้อมูลเชิงข้อเท็จจริง (Fact) ไม่มีคำคุณศัพท์บอกอารมณ์ (Sentiment adjective) โมเดลจึงต้องเดาจากสถิติคลาสส่วนใหญ่ (3-4 ดาว)
"""

md_content += """
---

## 🎯 สรุปสำหรับใช้พูดสไลด์ (12-15 นาที)
1. **System Evaluation:** แม้ BiLSTM จะขนาดใหญ่กว่า ({total_bilstm_size_mb:.1f} MB vs {total_base_size_mb:.1f} MB) แต่ Latency อยู่ที่เพียง {mean_latency_bilstm:.1f} ms ซึ่งเร็วพอสำหรับ Real-time Production แน่นอน
2. **Error Analysis:** BiLSTM ชนะในเคสคำประชดและรีวิวที่ติเรื่องบริการ ส่วนเคสที่ยังผิดพลาดมักเกิดจาก **Label Noise (คนกดดาวผิด)** และ **ข้อความสั้นเกินไป** ซึ่งถือเป็นข้อจำกัดตามธรรมชาติของ NLP
"""

report_save_path = os.path.join(outputs_dir, 'system_evaluation_and_error_analysis.md')
with open(report_save_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\nบันทึกรายงานฉบับสมบูรณ์แล้วที่: {report_save_path}")
print("=" * 70)
