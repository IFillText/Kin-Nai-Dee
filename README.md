# 🍽️ Kin Nai Dee (กินไหนดี)
> **Thai Food Review Rating Consistency Inspector & Latency Benchmark**  
> โครงงานวิชา Pattern Recognition / Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 ภาพรวมโครงงาน (Overview)
**Kin Nai Dee (กินไหนดี)** คือระบบตรวจสอบความสอดคล้องระหว่าง **"เนื้อหาข้อความรีวิวภาษาไทย"** กับ **"คะแนนดาวที่ผู้ใช้ให้ (1–5 ดาว)"** บนแพลตฟอร์มรีวิวร้านอาหาร (เช่น Wongnai, Google Maps) เพื่อตรวจจับและแจ้งเตือนข้อผิดพลาด เช่น:
* ลูกค้าพิมพ์บ่นบริการแย่มาก แต่อาจเผลอกดผิดให้ 5 ดาว
* ลูกค้าเขียนชมอาหารทุกอย่าง แต่แกล้งประชดกด 1 ดาว

ระบบมีทั้ง **Dual-Model Inference Engine** เปรียบเทียบระหว่าง Classical ML และ Deep Learning พร้อมหน้าจอ **Interactive Web Application** ที่แสดงผลการทำนาย, แถบวัดความหน่วง (Latency Meter), การแจกแจงความน่าจะเป็น (Probability Distribution), และตัวชี้วัดความขัดแย้งแบบ Real-time

---

## 🏆 ผลการทดลองสำคัญ (Key Benchmark Results)

| Model | Accuracy | Macro F1 | 1-Star Recall ⭐ | Latency (Mean) | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LinearSVC + TF-IDF (Baseline)** | 56.78% | 0.4176 | 14.46% | **4.39 ms** | **2.18 MB** |
| **BiLSTM + thai2vec (Proposed)** | 52.41% | **0.4185** | **54.22%** | 10.18 ms | 63.25 MB |

> 🎯 **Best Result ➔ BiLSTM:** สามารถกู้คืนความสามารถในการตรวจจับรีวิว 1 ดาว (ที่มีข้อมูลเพียง ~1% ในชุดข้อมูล) พุ่งขึ้นเกือบ 4 เท่า (+39.76%) โดยยังคงความเร็วตอบสนองระดับ 10 ms บน CPU พร้อมใช้งานจริง

---

## 🚀 วิธีการติดตั้งและรันใช้งาน (Quick Start)

### 1. โคลน Repository
```bash
git clone https://github.com/IFillText/Kin-Nai-Dee.git
cd Kin-Nai-Dee
```

### 2. สร้าง Virtual Environment และติดตั้ง Dependencies
```bash
# สร้าง venv
python -m venv venv

# เปิดใช้งาน venv (Windows)
venv\Scripts\activate

# ติดตั้งแพ็กเกจที่จำเป็น
pip install -r requirements.txt
```

### 3. รัน Web Application
สามารถเลือกได้ 2 วิธี:
* **วิธีที่ 1 (สะดวกสุด):** ดับเบิ้ลคลิกไฟล์ `web_app/run_app.bat`
* **วิธีที่ 2 (ผ่าน Terminal):**
  ```bash
  python web_app/app.py
  ```
เปิดเบราว์เซอร์ไปที่: **`http://127.0.0.1:5000`**

---

## 🛠️ สคริปต์การทดลองต่าง ๆ (Experiment Scripts)

* **ตรวจสอบและทำความสะอาดข้อมูล:**
  ```bash
  python clean_data.py
  ```
* **ฝึกสอน Baseline Model (LinearSVC):**
  ```bash
  python train_baseline.py
  ```
* **ฝึกสอน Proposed Model (BiLSTM + thai2vec):**
  ```bash
  python train_bilstm.py
  ```
* **รันเปรียบเทียบโมเดลและสร้างกราฟประเมินผล:**
  ```bash
  python compare_models.py
  ```
* **รัน System-level Evaluation & Error Analysis:**
  ```bash
  python run_system_eval_and_error_analysis.py
  ```

*(หมายเหตุ: ชุดข้อมูล `data/cleaned_train_data.csv.zip` ได้รับการบีบอัดไว้ เมื่อรันสคริปต์ระบบจะคลายซิปให้อัตโนมัติ)*

---

## 📂 โครงสร้างไดเรกทอรี (Project Structure)

```text
Kin-Nai-Dee/
├── data/                               # ชุดข้อมูล
│   ├── cleaned_train_data.csv.zip      # ข้อมูล Train ที่คลีนและตัดคำแล้ว (Auto-unzip)
│   ├── cleaned_test_data.csv           # ข้อมูล Blind Test
│   └── sample_submission.csv
├── outputs/                            # โมเดลและผลลัพธ์การประเมิน
│   ├── bilstm_model.pt                 # PyTorch Weights ของ BiLSTM
│   ├── bilstm_vocab.joblib             # Vocabulary & Index Mapping
│   ├── baseline_linearsvc.joblib       # LinearSVC Trained Model
│   ├── tfidf_vectorizer.joblib         # Fitted TF-IDF Vectorizer
│   ├── baseline_confusion_matrix.png
│   ├── bilstm_confusion_matrix.png
│   ├── model_comparison_detailed.png
│   └── tradeoff_f1_vs_latency.png
├── imbalance_compare/                  # การทดลองเจาะลึก Class Imbalance
│   ├── run_comparison.py
│   ├── confusion_matrix_comparison.png
│   └── recall_comparison_bar.png
├── web_app/                            # Interactive Web Dashboard
│   ├── app.py                          # Flask Backend
│   ├── run_app.bat                     # สคริปต์รันเว็บในคลิกเดียว
│   ├── static/                         # CSS, JS, Images
│   └── templates/                      # HTML UI
├── clean_data.py                       # Data Cleaning Pipeline
├── train_baseline.py                   # Baseline Training Script
├── train_bilstm.py                     # BiLSTM Training Script
├── compare_models.py                   # Model Comparison Script
├── run_system_eval_and_error_analysis.py
├── SLIDES_PRESENTATION.md              # โครงสร้างสไลด์และบทพูดนำเสนอ
├── requirements.txt                    # Python Dependencies
├── .gitignore                          # Git Ignore Configuration
└── README.md
```

---

## 👥 ผู้พัฒนา (Contributors)
* **โปรเจกต์:** Kin Nai Dee (กินไหนดี)
* **วิชา:** Pattern Recognition / Machine Learning
