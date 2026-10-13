import os
import sys
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from pythainlp.word_vector import WordVector

# กำหนด encoding ให้รองรับภาษาไทย
sys.stdout.reconfigure(encoding='utf-8')

# กำหนดสุ่ม Seed เพื่อความสามารถในการทำซ้ำ (Reproducibility)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== Running on Device: {device} ===")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. โหลดข้อมูล (Load Dataset)
# ============================================================
print("\n=== 1. Loading Cleaned Dataset ===")
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'cleaned_train_data.csv')
if not os.path.exists(data_path) and os.path.exists(data_path + '.zip'):
    import zipfile
    print('Extracting cleaned_train_data.csv.zip ...')
    with zipfile.ZipFile(data_path + '.zip', 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(data_path))
df = pd.read_csv(data_path, encoding='utf-8')

# เติมค่าว่างใน tokens
df['tokens'] = df['tokens'].fillna('').astype(str)
df = df[df['tokens'].str.strip() != '']
print(f"Total reviews: {len(df):,}")

# แบ่ง Train / Validation 80:20 (ใช้ random_state=42 เพื่อให้ชุดเดียวกับ Baseline)
X_train_tokens, X_val_tokens, y_train, y_val = train_test_split(
    df['tokens'],
    df['star_rating'],
    test_size=0.2,
    random_state=42,
    stratify=df['star_rating']
)
print(f"Train samples: {len(X_train_tokens):,} | Val samples: {len(X_val_tokens):,}")

# ============================================================
# 2. เตรียม thai2vec Pre-trained Word Embeddings
# ============================================================
print("\n=== 2. Loading thai2vec Word Embeddings ===")
wv_loader = WordVector(model_name="thai2fit_wv")
thai2vec = wv_loader.get_model()
embedding_dim = wv_loader.WV_DIM
print(f"thai2vec Vocabulary size: {len(thai2vec.key_to_index):,} | Dimension: {embedding_dim}")

# สร้าง Token to Index Mapping
# 0: <PAD>, 1: <UNK>
PAD_IDX = 0
UNK_IDX = 1
word2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
idx2word = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}

# เวกเตอร์ของ Embedding Matrix
embedding_weights = [
    np.zeros(embedding_dim, dtype=np.float32),              # <PAD>
    np.random.normal(0, 0.1, embedding_dim).astype(np.float32) # <UNK>
]

for word, idx in thai2vec.key_to_index.items():
    cur_idx = len(word2idx)
    word2idx[word] = cur_idx
    idx2word[cur_idx] = word
    embedding_weights.append(thai2vec[word].astype(np.float32))

embedding_matrix = np.array(embedding_weights)
print(f"Total Model Vocab Size: {len(word2idx):,} words")

# ============================================================
# 3. เตรียม PyTorch Dataset & DataLoader
# ============================================================
MAX_LEN = 150  # ครอบคลุมประมาณ 80% ของรีวิว ความยาวกำลังดีและรันเร็วบน GPU

class ReviewDataset(Dataset):
    def __init__(self, token_series, labels=None, max_len=MAX_LEN):
        self.max_len = max_len
        self.labels = labels.values if labels is not None else None
        
        # แปลงข้อความ tokens เป็น list of indices ล่วงหน้า
        self.data_indices = []
        for tokens_str in token_series:
            words = tokens_str.split()
            indices = [word2idx.get(w, UNK_IDX) for w in words[:self.max_len]]
            if len(indices) < self.max_len:
                indices = indices + [PAD_IDX] * (self.max_len - len(indices))
            self.data_indices.append(indices)
        self.data_indices = np.array(self.data_indices, dtype=np.int64)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.data_indices[idx], dtype=torch.long)
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

print("\n=== 3. Creating PyTorch DataLoaders ===")
train_dataset = ReviewDataset(X_train_tokens, y_train, max_len=MAX_LEN)
val_dataset = ReviewDataset(X_val_tokens, y_val, max_len=MAX_LEN)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
print(f"Batch size: {BATCH_SIZE} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ============================================================
# 4. ออกแบบสถาปัตยกรรมโมเดล BiLSTM
# ============================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=5, 
                 pretrained_weights=None, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
            # ให้ Fine-tune Embedding ได้เล็กน้อย
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dual Pooling: Max-pooling + Average-pooling (hidden_dim * 2 * 2)
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
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Global Avg Pooling & Max Pooling across sequence length
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        pooled = torch.cat((avg_pool, max_pool), dim=1)
        logits = self.classifier(pooled)
        return logits

# ============================================================
# 5. กำหนด Class Weights และ Loss Function
# ============================================================
# คำนวณค่าน้ำหนักเพื่อแก้ปัญหา Class Imbalance (ดาว 1, 2 มีน้อยมาก)
class_counts = np.bincount(y_train, minlength=5)
print("\nClass distribution in Training set:")
for c, cnt in enumerate(class_counts):
    print(f"  Class {c} ({c+1} Stars): {cnt:,} samples ({cnt/len(y_train)*100:.2f}%)")

# Smoothed Inverse Class Weighting
smoothed_weights = (len(y_train) / (5.0 * class_counts)) ** 0.5
smoothed_weights = smoothed_weights / smoothed_weights.mean()
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)
print(f"Calculated Class Weights: {[round(w, 2) for w in smoothed_weights]}")

model = BiLSTMClassifier(
    vocab_size=len(word2idx),
    embedding_dim=embedding_dim,
    hidden_dim=128,
    num_classes=5,
    pretrained_weights=embedding_matrix,
    num_layers=2,
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

# ============================================================
# 6. ฟังก์ชันการฝึกสอนและการทดสอบ (Training Loop)
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, acc, macro_f1

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, acc, macro_f1, np.array(all_preds), np.array(all_targets)

# ============================================================
# 7. เริ่มต้นการฝึกสอน (Train BiLSTM)
# ============================================================
EPOCHS = 6
best_macro_f1 = 0.0
best_preds = None
best_targets = None
best_model_path = os.path.join(base_dir, 'outputs', 'bilstm_model.pt')

print("\n=== 7. Training BiLSTM Model ===")
start_total = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_f1, val_preds, val_targets = evaluate(model, val_loader, criterion)
    scheduler.step(val_f1)
    
    elapsed = time.time() - epoch_start
    print(f"Epoch [{epoch}/{EPOCHS}] ({elapsed:.1f}s) | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% Val Macro-F1: {val_f1:.4f}")
    
    if val_f1 > best_macro_f1:
        best_macro_f1 = val_f1
        best_preds = val_preds
        best_targets = val_targets
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> Saved new best model to {best_model_path} (Macro F1: {best_macro_f1:.4f})")

print(f"\nAll Epochs completed in {time.time() - start_total:.1f} seconds")
print(f"Best Validation Macro F1-score: {best_macro_f1:.4f}")

# ============================================================
# 8. ประเมินผลอย่างละเอียดและสร้าง Confusion Matrix
# ============================================================
print("\n=== 8. Detailed Evaluation on Validation Set ===")
target_names = ['1 Star (Class 0)', '2 Stars (Class 1)', '3 Stars (Class 2)', '4 Stars (Class 3)', '5 Stars (Class 4)']
print(classification_report(best_targets, best_preds, target_names=target_names, digits=4))

# สร้างและบันทึกรูปภาพ Confusion Matrix
cm = confusion_matrix(best_targets, best_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Proposed Model (thai2vec + BiLSTM) Confusion Matrix\nMacro F1: {best_macro_f1:.4f}')
plt.xlabel('Predicted Rating')
plt.ylabel('True Rating')
plt.tight_layout()

cm_path = os.path.join(base_dir, 'outputs', 'bilstm_confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
print(f"Confusion Matrix saved to: {cm_path}")

# บันทึก Vocab & Metadata สำหรับนำไปใช้ใน Web App
vocab_metadata = {
    'word2idx': word2idx,
    'max_len': MAX_LEN,
    'embedding_dim': embedding_dim,
    'hidden_dim': 128,
    'num_classes': 5,
    'num_layers': 2,
    'dropout': 0.3
}
meta_path = os.path.join(base_dir, 'outputs', 'bilstm_vocab.joblib')
joblib.dump(vocab_metadata, meta_path)
print(f"Metadata and Vocab saved to: {meta_path}")

# ============================================================
# 9. ทดสอบระบบตรวจจับความขัดแย้ง (Conflict Detection Test)
# ============================================================
print("\n=== 9. Sample Conflict Detection Test ===")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

test_reviews = [
    ('ร้านนี้บรรยากาศดี บริการยอดเยี่ยม อาหารอร่อยถูกปากมาก แนะนำเลยครับ ประทับใจสุดๆ', 0),
    ('รอนานเป็นชั่วโมง พนักงานบริการแย่มาก อาหารก็จืดชืด ไม่สมราคาเลย เสียความรู้สึก', 4),
    ('รสชาติพอใช้ได้ ราคากลางๆ ไม่ได้แย่แต่ก็ไม่ได้ว้าว', 2)
]

for text, actual_class in test_reviews:
    # Preprocess text
    t = normalize(text)
    t = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', t)
    tokens = [w.strip() for w in word_tokenize(t, engine='newmm', keep_whitespace=False) if len(w.strip()) > 0]
    
    indices = [word2idx.get(w, UNK_IDX) for w in tokens[:MAX_LEN]]
    if len(indices) < MAX_LEN:
        indices = indices + [PAD_IDX] * (MAX_LEN - len(indices))
        
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = torch.argmax(logits, dim=1).item()
        
    diff = abs(actual_class - pred_class)
    conflict_status = 'Normal' if diff <= 1 else ('Mild Conflict' if diff == 2 else 'Severe Conflict')
    print('Review:', text)
    print(f'  Actual Star: {actual_class + 1} | Predicted: {pred_class + 1} | Status: {conflict_status}\n')

print("=== BiLSTM Training & Evaluation Pipeline Completed Successfully! ===")
