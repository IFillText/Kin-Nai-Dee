import sys, os, re, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

sys.stdout.reconfigure(encoding='utf-8')

print('=== 1. Loading Dataset ===')
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'cleaned_train_data.csv')
if not os.path.exists(data_path) and os.path.exists(data_path + '.zip'):
    import zipfile
    print('Extracting cleaned_train_data.csv.zip ...')
    with zipfile.ZipFile(data_path + '.zip', 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(data_path))
df = pd.read_csv(data_path, encoding='utf-8')

df['review_body'] = df['review_body'].fillna('').astype(str)
df = df[df['review_body'].str.strip() != '']
print('Total reviews:', len(df))

print('\n=== 2. Preprocessing & Tokenization Definition ===')
def thai_tokenizer(text):
    text = normalize(text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    return [t.strip() for t in tokens if len(t.strip()) > 0]

X_train, X_val, y_train, y_val = train_test_split(
    df['review_body'],
    df['star_rating'],
    test_size=0.2,
    random_state=42,
    stratify=df['star_rating']
)

print('Train samples:', len(X_train), '| Val samples:', len(X_val))

print('\n=== 3. TF-IDF Vectorization (Unigram + Bigram) ===')
start_time = time.time()
vectorizer = TfidfVectorizer(
    tokenizer=thai_tokenizer,
    ngram_range=(1, 2),
    max_features=25000,
    min_df=3,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print('TF-IDF completed in', round(time.time() - start_time, 2), 'seconds')
print('Feature matrix shape:', X_train_vec.shape)

print('\n=== 4. Training Baseline Model (LinearSVC) ===')
start_time = time.time()
baseline_model = LinearSVC(
    class_weight='balanced',
    C=0.5,
    random_state=42,
    max_iter=3000
)
baseline_model.fit(X_train_vec, y_train)
print('Training completed in', round(time.time() - start_time, 2), 'seconds')

print('\n=== 5. Evaluation on Validation Set ===')
y_pred = baseline_model.predict(X_val_vec)

acc = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')
weighted_f1 = f1_score(y_val, y_pred, average='weighted')

print('Accuracy:', round(acc * 100, 2), '%')
print('Macro F1-score:', round(macro_f1, 4))
print('Weighted F1-score:', round(weighted_f1, 4))

target_names = ['1 Star (Class 0)', '2 Stars (Class 1)', '3 Stars (Class 2)', '4 Stars (Class 3)', '5 Stars (Class 4)']
print('\nDetailed Classification Report:')
print(classification_report(y_val, y_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Baseline Model (LinearSVC) Confusion Matrix')
plt.xlabel('Predicted Rating')
plt.ylabel('True Rating')
plt.tight_layout()

os.makedirs(r'D:\kin nai dee\outputs', exist_ok=True)
cm_path = r'D:\kin nai dee\outputs\baseline_confusion_matrix.png'
plt.savefig(cm_path, dpi=300)
print('\nConfusion matrix saved to:', cm_path)

joblib.dump(baseline_model, r'D:\kin nai dee\outputs\baseline_linearsvc.joblib')
joblib.dump(vectorizer, r'D:\kin nai dee\outputs\tfidf_vectorizer.joblib')
print('Model and Vectorizer saved to D:\\kin nai dee\\outputs\\')

print('\n=== 6. Sample Conflict Detection Test ===')
test_reviews = [
    ('ร้านนี้บรรยากาศดี บริการยอดเยี่ยม อาหารอร่อยถูกปากมาก แนะนำเลยครับ ประทับใจสุดๆ', 0),
    ('รอนานเป็นชั่วโมง พนักงานบริการแย่มาก อาหารก็จืดชืด ไม่สมราคาเลย เสียความรู้สึก', 4),
    ('รสชาติพอใช้ได้ ราคากลางๆ ไม่ได้แย่แต่ก็ไม่ได้ว้าว', 2)
]

for text, actual_class in test_reviews:
    vec = vectorizer.transform([text])
    pred_class = baseline_model.predict(vec)[0]
    diff = abs(actual_class - pred_class)
    conflict_status = 'Normal' if diff <= 1 else ('Mild Conflict' if diff == 2 else 'Severe Conflict')
    print('Review:', text)
    print(f'  Actual Star: {actual_class + 1} | Predicted: {pred_class + 1} | Status: {conflict_status}\n')