import os
import sys
import re
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

sys.stdout.reconfigure(encoding='utf-8')

# กำหนดฟังก์ชัน thai_tokenizer เพื่อให้ unpickle vectorizer ได้ถูกต้อง
def thai_tokenizer(text):
    text = normalize(text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    return [t.strip() for t in tokens if len(t.strip()) > 0]

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'cleaned_train_data.csv')
if not os.path.exists(data_path) and os.path.exists(data_path + '.zip'):
    import zipfile
    print('Extracting cleaned_train_data.csv.zip ...')
    with zipfile.ZipFile(data_path + '.zip', 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(data_path))
df = pd.read_csv(data_path, encoding='utf-8')
df['review_body'] = df['review_body'].fillna('').astype(str)
df['tokens'] = df['tokens'].fillna('').astype(str)

# Train/Val Split เดียวกัน 100%
X_train_body, X_val_body, y_train, y_val = train_test_split(
    df['review_body'], df['star_rating'], test_size=0.2, random_state=42, stratify=df['star_rating']
)
X_train_tok, X_val_tok, _, _ = train_test_split(
    df['tokens'], df['star_rating'], test_size=0.2, random_state=42, stratify=df['star_rating']
)

print(f"=== Loaded Validation Set: {len(y_val):,} samples ===")

# ----------------------------------------------------
# 1. ทำนายด้วย Baseline (LinearSVC)
# ----------------------------------------------------
print("1. Loading Baseline Model (LinearSVC)...")
vec = joblib.load(os.path.join(base_dir, 'outputs', 'tfidf_vectorizer.joblib'))
baseline_model = joblib.load(os.path.join(base_dir, 'outputs', 'baseline_linearsvc.joblib'))

X_val_vec = vec.transform(X_val_body)
y_pred_base = baseline_model.predict(X_val_vec)

# ----------------------------------------------------
# 2. ทำนายด้วย BiLSTM
# ----------------------------------------------------
print("2. Loading Proposed Model (thai2vec + BiLSTM)...")
meta = joblib.load(os.path.join(base_dir, 'outputs', 'bilstm_vocab.joblib'))
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

bilstm_model.load_state_dict(torch.load(os.path.join(base_dir, 'outputs', 'bilstm_model.pt'), map_location=device))
bilstm_model.eval()

# สร้าง Tensor ของ Val
data_indices = []
for tokens_str in X_val_tok:
    words = tokens_str.split()
    indices = [word2idx.get(w, 1) for w in words[:max_len]]
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    data_indices.append(indices)

tensor_x = torch.tensor(np.array(data_indices), dtype=torch.long)
loader = torch.utils.data.DataLoader(tensor_x, batch_size=128, shuffle=False)

y_pred_bilstm = []
with torch.no_grad():
    for batch in loader:
        out = bilstm_model(batch.to(device))
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred_bilstm.extend(preds)
y_pred_bilstm = np.array(y_pred_bilstm)

# ----------------------------------------------------
# 3. เจาะลึก Confusion Matrix
# ----------------------------------------------------
cm_base = confusion_matrix(y_val, y_pred_base)
cm_bilstm = confusion_matrix(y_val, y_pred_bilstm)

print("\n" + "="*70)
print("=== 1. เจาะลึกคลาส '1 ดาว' (Class 0) ทั้งหมด 83 รีวิวในชุด Validation ===")
print("="*70)
print(f"รีวิวที่เป็น 1 ดาวจริงมีทั้งหมด: {cm_base[0].sum()} รีวิว")
print("\n[Baseline - LinearSVC ทาย 1 ดาวว่าอย่างไร?]")
for star, count in enumerate(cm_base[0]):
    print(f"  ทายว่าเป็น {star+1} ดาว: {count:2d} รีวิว ({count/83*100:5.1f}%)")

print("\n[BiLSTM + thai2vec ทาย 1 ดาวว่าอย่างไร?]")
for star, count in enumerate(cm_bilstm[0]):
    print(f"  ทายว่าเป็น {star+1} ดาว: {count:2d} รีวิว ({count/83*100:5.1f}%)")

print("\n" + "="*70)
print("=== 2. เจาะลึกคลาส '5 ดาว' (Class 4) ทั้งหมด 1,359 รีวิวในชุด Validation ===")
print("="*70)
print("\n[Baseline - LinearSVC ทาย 5 ดาวว่าอย่างไร?]")
for star, count in enumerate(cm_base[4]):
    print(f"  ทายว่าเป็น {star+1} ดาว: {count:4d} รีวิว ({count/1359*100:5.1f}%)")

print("\n[BiLSTM + thai2vec ทาย 5 ดาวว่าอย่างไร?]")
for star, count in enumerate(cm_bilstm[4]):
    print(f"  ทายว่าเป็น {star+1} ดาว: {count:4d} รีวิว ({count/1359*100:5.1f}%)")

# ----------------------------------------------------
# 4. ตารางเปรียบเทียบ Precision / Recall / F1 ทุกคลาส
# ----------------------------------------------------
rep_base = classification_report(y_val, y_pred_base, output_dict=True, digits=4)
rep_bi = classification_report(y_val, y_pred_bilstm, output_dict=True, digits=4)

print("\n" + "="*70)
print("=== 3. ตารางเปรียบเทียบ Precision, Recall, F1 ทุกคลาส (Side-by-Side) ===")
print("="*70)
header = f"{'Class (Rating)':<18} | {'LinearSVC Precision':<20} {'BiLSTM Precision':<18} | {'LinearSVC Recall':<17} {'BiLSTM Recall':<15} | {'LinearSVC F1':<13} {'BiLSTM F1':<10}"
print(header)
print("-" * len(header))

for c in range(5):
    name = f"{c+1} Star (Class {c})"
    c_str = str(c)
    p_base = rep_base[c_str]['precision'] * 100
    p_bi = rep_bi[c_str]['precision'] * 100
    r_base = rep_base[c_str]['recall'] * 100
    r_bi = rep_bi[c_str]['recall'] * 100
    f_base = rep_base[c_str]['f1-score']
    f_bi = rep_bi[c_str]['f1-score']
    print(f"{name:<18} | {p_base:6.2f}%              {p_bi:6.2f}%            | {r_base:6.2f}%           {r_bi:6.2f}%         | {f_base:6.4f}        {f_bi:6.4f}")

print("-" * len(header))
print(f"{'Macro Average':<18} | {rep_base['macro avg']['precision']*100:6.2f}%              {rep_bi['macro avg']['precision']*100:6.2f}%            | {rep_base['macro avg']['recall']*100:6.2f}%           {rep_bi['macro avg']['recall']*100:6.2f}%         | {rep_base['macro avg']['f1-score']:6.4f}        {rep_bi['macro avg']['f1-score']:6.4f}")
print(f"{'Accuracy':<18} |                                                    | {accuracy_score(y_val, y_pred_base)*100:6.2f}%           {accuracy_score(y_val, y_pred_bilstm)*100:6.2f}%         |")

# ----------------------------------------------------
# 5. ตัวอย่างข้อความรีวิวจริงที่ Baseline ทายผิด แต่ BiLSTM ทายถูก!
# ----------------------------------------------------
print("\n" + "="*70)
print("=== 4. ตัวอย่างรีวิว 1 ดาวจริง ที่ LinearSVC ทายผิด (หลุดไป 3-4 ดาว) แต่ BiLSTM ทายถูก ===")
print("="*70)

val_df = pd.DataFrame({
    'text': X_val_body.values,
    'actual': y_val.values,
    'pred_base': y_pred_base,
    'pred_bilstm': y_pred_bilstm
})

# เคสที่ Actual=0 (1 ดาว), LinearSVC ทาย >= 2 (3-5 ดาว), แต่ BiLSTM ทาย 0 (1 ดาว)
diff_cases = val_df[(val_df['actual'] == 0) & (val_df['pred_base'] >= 2) & (val_df['pred_bilstm'] == 0)]

print(f"จำนวนรีวิว 1 ดาวที่ LinearSVC ทายผิดเป็น 3-5 ดาว แต่ BiLSTM กู้กลับมาทายถูกได้: {len(diff_cases)} เคส!\n")

for i, (_, row) in enumerate(diff_cases.head(4).iterrows(), 1):
    print(f"--- [ตัวอย่างที่ {i}] ---")
    print(f"คะแนนจริง: 1 ดาว")
    print(f"LinearSVC ทายว่า: {row['pred_base'] + 1} ดาว (ผิดพลาด)")
    print(f"BiLSTM    ทายว่า: {row['pred_bilstm'] + 1} ดาว (ถูกต้องแม่นยำ!)")
    print(f"ข้อความรีวิว: {row['text'][:250]}...")
    print()

# ----------------------------------------------------
# 6. สร้างกราฟรูปภาพเปรียบเทียบ Confusion Matrix 2 โมเดลเคียงข้างกัน
# ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

labels = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title(f"Baseline (LinearSVC)\nMacro F1: {rep_base['macro avg']['f1-score']:.4f} | 1-Star Recall: {rep_base['0']['recall']*100:.1f}%")
axes[0].set_xlabel("Predicted Rating")
axes[0].set_ylabel("True Rating")

sns.heatmap(cm_bilstm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title(f"Proposed Model (thai2vec + BiLSTM)\nMacro F1: {rep_bi['macro avg']['f1-score']:.4f} | 1-Star Recall: {rep_bi['0']['recall']*100:.1f}%")
axes[1].set_xlabel("Predicted Rating")
axes[1].set_ylabel("True Rating")

plt.tight_layout()
out_fig = os.path.join(base_dir, 'outputs', 'model_comparison_detailed.png')
plt.savefig(out_fig, dpi=300)
print(f"Comparison Chart saved to: {out_fig}")
