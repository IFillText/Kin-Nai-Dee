# Kin Nai Dee (กินไหนดี) — Web Application

เว็บแอปพลิเคชันตรวจจับความขัดแย้งของรีวิวร้านอาหารด้วย NLP & Deep Learning (วิชา Pattern Recognition)

## คุณสมบัติหลัก (Features)
1. **Live Review Consistency Inspector**:
   - พิมพ์หรือวางรีวิวภาษาไทย เลือกระดับดาว (1-5 ดาว)
   - ปุ๋ม Preset ตัวอย่างเคสจริง (ด่าบริการแต่ชมอาหาร, ประชดเจ้าของร้าน, ชมเพียบแต่กดผิด, 5 ดาวแท้, 3 ดาวกลางๆ)
   - โมเดลประมวลผลคู่ขนาน:
     - **Proposed: BiLSTM + thai2vec** (Deep Learning)
     - **Baseline: LinearSVC + TF-IDF** (Machine Learning)
   - ป้ายแจ้งเตือนความขัดแย้ง (🟢 สอดคล้องปกติ, 🟡 คลาดเคลื่อนเล็กน้อย, 🔴 ขัดแย้งรุนแรง)
   - แจกแจงความน่าจะเป็นทุกคลาส (1-5 ดาว) และวัดค่า Latency (ms) สดๆ
   - ไฮไลท์คำสำคัญ (Positive / Negative tokens)

2. **Engineering Benchmark**:
   - ตารางเปรียบเทียบ Macro F1, 1-Star Recall, Model Size, Latency, Throughput
   - กฎการตัดสินใจ Trade-off เชิงวิศวกรรม

3. **Visual Charts Gallery**:
   - แสดงกราฟผลการทดลองจริง: F1 vs Latency Trade-off, Confusion Matrices, Data Imbalance Comparison

4. **Error Analysis Deep-Dive**:
   - วิเคราะห์ Success Cases vs Failure Cases พร้อม Root Cause Analysis

## วิธีรันเว็บแอปพลิเคชัน
1. ดับเบิ้ลคลิกไฟล์ `run_app.bat` หรือ
2. รันคำสั่งผ่าน Terminal:
   ```bash
   cd "d:\Kin nai dee\web_app"
   ..\venv\Scripts\python.exe app.py
   ```
3. เปิดเบราว์เซอร์ไปที่: `http://127.0.0.1:5000`
