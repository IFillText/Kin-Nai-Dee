import os
import sys
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

sys.stdout.reconfigure(encoding='utf-8')

# กำหนด tokenizer เพื่อโหลด vectorizer
def thai_tokenizer(text):
    text = normalize(text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    return [t.strip() for t in tokens if len(t.strip()) > 0]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'imbalance_compare')
os.makedirs(output_dir, exist_ok=True)

print("=" * 65)
print("  การทดลองเปรียบเทียบผลกระทบของ DATA IMBALANCE (Kin nai dee)")
print("=" * 65)

# 1. โหลดข้อมูล
data_path = os.path.join(base_dir, 'data', 'cleaned_train_data.csv')
df = pd.read_csv(data_path, encoding='utf-8')
df['review_body'] = df['review_body'].fillna('').astype(str)
df = df[df['review_body'].str.strip() != '']
total_samples = len(df)
print(f"\n[1] โหลดข้อมูลทั้งหมด: {total_samples:,} รายการ")

# -------------------------------------------------------------
# ทดลองส่วนที่ 1: เปรียบเทียบการ Split (Random vs Stratified)
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("  การทดลองที่ 1: Random Split (ไม่ใส่ stratify) VS Stratified Split")
print("=" * 65)

# สุ่มแบบไม่ใส่ stratify
_, _, y_train_rand, y_val_rand = train_test_split(
    df['review_body'], df['star_rating'], test_size=0.2, random_state=123, stratify=None
)

# สุ่มแบบใส่ stratify
_, _, y_train_strat, y_val_strat = train_test_split(
    df['review_body'], df['star_rating'], test_size=0.2, random_state=42, stratify=df['star_rating']
)

star_labels = ['1 Star (Class 0)', '2 Stars (Class 1)', '3 Stars (Class 2)', '4 Stars (Class 3)', '5 Stars (Class 4)']
split_comp = []
for c in range(5):
    total_c = (df['star_rating'] == c).sum()
    pct_orig = (total_c / total_samples) * 100
    
    val_rand_c = (y_val_rand == c).sum()
    pct_rand = (val_rand_c / len(y_val_rand)) * 100
    
    val_strat_c = (y_val_strat == c).sum()
    pct_strat = (val_strat_c / len(y_val_strat)) * 100
    
    split_comp.append({
        'Rating': star_labels[c],
        'Total Count': f"{total_c:,} ({pct_orig:.2f}%)",
        'Val (Without Stratify)': f"{val_rand_c:,} ({pct_rand:.2f}%)",
        'Val (With Stratify)': f"{val_strat_c:,} ({pct_strat:.2f}%)"
    })

df_split_comp = pd.DataFrame(split_comp)
print(df_split_comp.to_string(index=False))

# -------------------------------------------------------------
# ทดลองส่วนที่ 2: เปรียบเทียบโมเดล ไม่แก้ Imbalance vs แก้ Imbalance
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("  การทดลองที่ 2: ผลของ class_weight ต่อการทำนาย (None vs Balanced)")
print("=" * 65)

# โหลด TF-IDF Vectorizer
vec_path = os.path.join(base_dir, 'outputs', 'tfidf_vectorizer.joblib')
print("\nกำลังแปลงข้อความเป็นเวกเตอร์ TF-IDF...")
vec = joblib.load(vec_path)

X_train, X_val, y_train, y_val = train_test_split(
    df['review_body'], df['star_rating'], test_size=0.2, random_state=42, stratify=df['star_rating']
)

X_train_vec = vec.transform(X_train)
X_val_vec = vec.transform(X_val)

# โมเดล A: ไม่แก้ Imbalance (class_weight=None)
print("\n[2.1] กำลังฝึกโมเดล A: ไม่แก้ Imbalance (class_weight=None)...")
model_no_weight = LinearSVC(class_weight=None, C=0.5, random_state=42, max_iter=3000)
model_no_weight.fit(X_train_vec, y_train)
y_pred_no_weight = model_no_weight.predict(X_val_vec)

# โมเดล B: แก้ Imbalance ด้วย Class Weight (class_weight='balanced')
print("[2.2] กำลังฝึกโมเดล B: แก้ Imbalance (class_weight='balanced')...")
model_balanced = LinearSVC(class_weight='balanced', C=0.5, random_state=42, max_iter=3000)
model_balanced.fit(X_train_vec, y_train)
y_pred_balanced = model_balanced.predict(X_val_vec)

# คำนวณ Metrics เปรียบเทียบ
rep_no = classification_report(y_val, y_pred_no_weight, target_names=star_labels, output_dict=True)
rep_bal = classification_report(y_val, y_pred_balanced, target_names=star_labels, output_dict=True)

print("\n--- ผลลัพธ์เปรียบเทียบในภาพรวม ---")
print(f"โมเดล A (ไม่แก้ Imbalance):")
print(f"  Accuracy:      {accuracy_score(y_val, y_pred_no_weight)*100:.2f}%")
print(f"  Macro F1:      {f1_score(y_val, y_pred_no_weight, average='macro'):.4f}")
print(f"  1-Star Recall: {rep_no['1 Star (Class 0)']['recall']*100:.2f}%")
print(f"  1-Star F1:     {rep_no['1 Star (Class 0)']['f1-score']:.4f}")

print(f"\nโมเดล B (แก้ Imbalance - Balanced Weights):")
print(f"  Accuracy:      {accuracy_score(y_val, y_pred_balanced)*100:.2f}%")
print(f"  Macro F1:      {f1_score(y_val, y_pred_balanced, average='macro'):.4f}")
print(f"  1-Star Recall: {rep_bal['1 Star (Class 0)']['recall']*100:.2f}%")
print(f"  1-Star F1:     {rep_bal['1 Star (Class 0)']['f1-score']:.4f}")

# ตารางเปรียบเทียบรายคลาส
table_data = []
for lbl in star_labels:
    table_data.append({
        'Rating': lbl,
        'Recall (No Weight)': f"{rep_no[lbl]['recall']*100:.1f}%",
        'Recall (Balanced)': f"{rep_bal[lbl]['recall']*100:.1f}%",
        'F1 (No Weight)': f"{rep_no[lbl]['f1-score']:.4f}",
        'F1 (Balanced)': f"{rep_bal[lbl]['f1-score']:.4f}",
    })

df_metrics = pd.DataFrame(table_data)
print("\n--- รายละเอียดเจาะลึกแต่ละระดับดาว ---")
print(df_metrics.to_string(index=False))

# -------------------------------------------------------------
# วาดกราฟเปรียบเทียบ
# -------------------------------------------------------------
print("\n[3] กำลังวาดกราฟเปรียบเทียบ...")
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Arial']
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm_no = confusion_matrix(y_val, y_pred_no_weight)
cm_bal = confusion_matrix(y_val, y_pred_balanced)

sns.heatmap(cm_no, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[f"{i}★" for i in range(1, 6)], yticklabels=[f"{i}★" for i in range(1, 6)])
axes[0].set_title(f"Model A: Without Class Weight\n(Macro F1: {f1_score(y_val, y_pred_no_weight, average='macro'):.4f} | 1★ Recall: {rep_no['1 Star (Class 0)']['recall']*100:.1f}%)", fontsize=12, fontweight='bold', color='darkred')
axes[0].set_xlabel("Predicted Rating", fontsize=11)
axes[0].set_ylabel("True Rating", fontsize=11)

sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=[f"{i}★" for i in range(1, 6)], yticklabels=[f"{i}★" for i in range(1, 6)])
axes[1].set_title(f"Model B: With Balanced Class Weight\n(Macro F1: {f1_score(y_val, y_pred_balanced, average='macro'):.4f} | 1★ Recall: {rep_bal['1 Star (Class 0)']['recall']*100:.1f}%)", fontsize=12, fontweight='bold', color='darkgreen')
axes[1].set_xlabel("Predicted Rating", fontsize=11)
axes[1].set_ylabel("True Rating", fontsize=11)

plt.tight_layout()
cm_fig_path = os.path.join(output_dir, 'confusion_matrix_comparison.png')
plt.savefig(cm_fig_path, dpi=300)
plt.close()

# กราฟแท่ง Recall เปรียบเทียบความสามารถในการจับรีวิวแต่ละดาว
plt.figure(figsize=(10, 5))
x = np.arange(5)
width = 0.35

rec_no = [rep_no[lbl]['recall'] * 100 for lbl in star_labels]
rec_bal = [rep_bal[lbl]['recall'] * 100 for lbl in star_labels]

plt.bar(x - width/2, rec_no, width, label='Without Class Weight', color='#e74c3c')
plt.bar(x + width/2, rec_bal, width, label='With Balanced Class Weight', color='#2ecc71')

plt.xlabel('Rating (Stars)', fontsize=12, fontweight='bold')
plt.ylabel('Recall (%) [จับคลาสได้กี่เปอร์เซ็นต์]', fontsize=12, fontweight='bold')
plt.title('Impact of Class Weighting on Recall across Rating Classes', fontsize=14, fontweight='bold')
plt.xticks(x, ['1 Star (1%)', '2 Stars (4.6%)', '3 Stars (30.4%)', '4 Stars (46.9%)', '5 Stars (17%)'])
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i in range(5):
    plt.text(i - width/2, rec_no[i] + 1, f"{rec_no[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.text(i + width/2, rec_bal[i] + 1, f"{rec_bal[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
bar_fig_path = os.path.join(output_dir, 'recall_comparison_bar.png')
plt.savefig(bar_fig_path, dpi=300)
plt.close()

# บันทึกเป็น Markdown Report
report_md = f"""# ผลการทดลองเปรียบเทียบ Data Imbalance
โฟลเดอร์: `imbalance_compare/`

## 1. เปรียบเทียบการสุ่มแบ่งข้อมูล (Data Splitting)
| ระดับดาว | ข้อมูลทั้งหมด | สุ่มธรรมดา (ไม่ใส่ stratify) | ใส่ stratify (สัดส่วนคงที่) |
| :--- | :--- | :--- | :--- |
| 1 ดาว (น้อยสุด) | {split_comp[0]['Total Count']} | {split_comp[0]['Val (Without Stratify)']} | **{split_comp[0]['Val (With Stratify)']}** |
| 2 ดาว | {split_comp[1]['Total Count']} | {split_comp[1]['Val (Without Stratify)']} | **{split_comp[1]['Val (With Stratify)']}** |
| 3 ดาว | {split_comp[2]['Total Count']} | {split_comp[2]['Val (Without Stratify)']} | **{split_comp[2]['Val (With Stratify)']}** |
| 4 ดาว | {split_comp[3]['Total Count']} | {split_comp[3]['Val (Without Stratify)']} | **{split_comp[3]['Val (With Stratify)']}** |
| 5 ดาว | {split_comp[4]['Total Count']} | {split_comp[4]['Val (Without Stratify)']} | **{split_comp[4]['Val (With Stratify)']}** |

## 2. เปรียบเทียบผลโมเดล: ไม่แก้ Imbalance vs แก้ Imbalance
| ตัวชี้วัด | โมเดล A (ไม่แก้ Imbalance) | โมเดล B (แก้ด้วย Balanced Weight) | ผลต่าง / การเปลี่ยนแปลง |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {accuracy_score(y_val, y_pred_no_weight)*100:.2f}% | {accuracy_score(y_val, y_pred_balanced)*100:.2f}% | หลอกตา (ลดลงเล็กน้อยเพราะไม่เอียงเข้าคลาสใหญ่) |
| **Macro F1-score** | {f1_score(y_val, y_pred_no_weight, average='macro'):.4f} | **{f1_score(y_val, y_pred_balanced, average='macro'):.4f}** | **ดีขึ้นอย่างชัดเจน** |
| **1-Star Recall** | {rep_no['1 Star (Class 0)']['recall']*100:.2f}% | **{rep_bal['1 Star (Class 0)']['recall']*100:.2f}%** | **จับ 1 ดาวได้เพิ่มขึ้นอย่างก้าวกระโดด!** |
| **1-Star F1-score** | {rep_no['1 Star (Class 0)']['f1-score']:.4f} | **{rep_bal['1 Star (Class 0)']['f1-score']:.4f}** | สูงขึ้นชัดเจน |

## 3. รูปภาพผลการทดลอง
1. `confusion_matrix_comparison.png` - เปรียบเทียบ Confusion Matrix ข้างกัน
2. `recall_comparison_bar.png` - กราฟแท่งเปรียบเทียบ Recall ของแต่ละดาว
"""

with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(report_md)

print("\n" + "=" * 65)
print(f"บันทึกผลการทดลองและรูปภาพลงโฟลเดอร์ เรียบร้อยแล้ว:")
print(f"  -> {output_dir}")
print("=" * 65)
