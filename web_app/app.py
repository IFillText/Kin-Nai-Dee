import os
import sys
import re
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, send_from_directory
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

# กำหนด encoding
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
IMBALANCE_DIR = os.path.join(PROJECT_ROOT, 'imbalance_compare')

# ----------------------------------------------------
# 1. นิยามโครงสร้างโมเดล BiLSTM
# ----------------------------------------------------
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

# ----------------------------------------------------
# 2. ฟังก์ชันตัดคำภาษาไทย
# ----------------------------------------------------
def thai_tokenizer(text):
    text = normalize(text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    return [t.strip() for t in tokens if len(t.strip()) > 0]

# ----------------------------------------------------
# 3. โหลดโมเดลทั้งสองตัวเมื่อเริ่มต้นเซิร์ฟเวอร์
# ----------------------------------------------------
print("Loading LinearSVC Baseline...")
try:
    vec = joblib.load(os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.joblib'))
    baseline_model = joblib.load(os.path.join(OUTPUTS_DIR, 'baseline_linearsvc.joblib'))
    print("LinearSVC Loaded successfully!")
except Exception as e:
    print(f"Error loading LinearSVC: {e}")
    vec, baseline_model = None, None

print("Loading BiLSTM Proposed Model...")
try:
    meta = joblib.load(os.path.join(OUTPUTS_DIR, 'bilstm_vocab.joblib'))
    word2idx = meta['word2idx']
    max_len = meta['max_len']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bilstm_model = BiLSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=meta['embedding_dim'],
        hidden_dim=meta['hidden_dim'],
        num_classes=meta['num_classes'],
        num_layers=meta['num_layers'],
        dropout=meta['dropout']
    ).to(device)

    model_path = os.path.join(OUTPUTS_DIR, 'bilstm_model.pt')
    bilstm_model.load_state_dict(torch.load(model_path, map_location=device))
    bilstm_model.eval()
    print(f"BiLSTM Loaded successfully on device: {device}!")
except Exception as e:
    print(f"Error loading BiLSTM: {e}")
    meta, word2idx, max_len, bilstm_model, device = None, None, None, None, 'cpu'

# ----------------------------------------------------
# 4. โมดูลวิเคราะห์มิติรสชาติและคุณลักษณะการตัดสินใจ (Taste & Aspect Extraction)
# ----------------------------------------------------
def extract_taste_and_aspects(text, tokens):
    text_clean = text.lower()

    # A. วิเคราะห์ระดับความเผ็ด (Spiciness)
    spicy_high = ['เผ็ดมาก', 'เผ็ดจัด', 'เผ็ดแซ่บ', 'แซ่บมาก', 'ซี๊ด', 'เผ็ดร้อน', 'พริกสิบเม็ด', 'แสบปาก', 'พริกท่วม', 'เผ็ดจี๊ด', 'โคตรเผ็ด', 'เผ็ดแสบ']
    spicy_general = ['เผ็ด', 'แซ่บ', 'พริก', 'จัดจ้าน', 'รสจัด']
    spicy_low = ['เผ็ดน้อย', 'ไม่ค่อยเผ็ด', 'ไม่เผ็ดเท่าไหร่', 'เผ็ดกำลังดี', 'เผ็ดพอดี']
    spicy_neg = ['ไม่เผ็ด', 'ไม่แซ่บ', 'ไม่ใส่พริก', 'ไร้พริก']

    spiciness = {"level": 0, "label": "ไม่ระบุความเผ็ด", "score_pct": 0, "detected": False}
    if any(k in text_clean for k in spicy_neg):
        spiciness = {"level": 0, "label": "ไม่เผ็ด / รสไม่จัด", "score_pct": 10, "detected": True}
    elif any(k in text_clean for k in spicy_high):
        spiciness = {"level": 3, "label": "เผ็ดจัดจ้าน / แซ่บมาก", "score_pct": 95, "detected": True}
    elif any(k in text_clean for k in spicy_low):
        spiciness = {"level": 1, "label": "เผ็ดน้อย / รสพอดี", "score_pct": 40, "detected": True}
    elif any(k in text_clean for k in spicy_general):
        spiciness = {"level": 2, "label": "เผ็ดชัดเจน", "score_pct": 70, "detected": True}

    # B. วิเคราะห์ระดับความหวาน (Sweetness)
    sweet_high = ['หวานมาก', 'หวานเจี๊ยบ', 'หวานนำ', 'เลี่ยน', 'หวานตัดขา', 'หวานเกิน', 'หวานแสบคอ']
    sweet_low = ['หวานน้อย', 'ไม่ค่อยหวาน', 'หวานกำลังดี', 'หวานพอดี', 'ไม่เลี่ยน']
    sweet_neg = ['ไม่หวาน', 'ไม่เลี่ยนเลย', 'ไม่หวานเลย']
    sweet_general = ['หวาน']

    sweetness = {"level": 0, "label": "ไม่ระบุ", "score_pct": 0, "detected": False}
    if any(k in text_clean for k in sweet_neg):
        sweetness = {"level": 0, "label": "ไม่หวาน", "score_pct": 10, "detected": True}
    elif any(k in text_clean for k in sweet_high):
        sweetness = {"level": 2, "label": "หวานนำ / ค่อนข้างหวาน", "score_pct": 85, "detected": True}
    elif any(k in text_clean for k in sweet_low):
        sweetness = {"level": 1, "label": "หวานน้อย / พอดี", "score_pct": 40, "detected": True}
    elif any(k in text_clean for k in sweet_general):
        sweetness = {"level": 1, "label": "รสหวานพอดี", "score_pct": 55, "detected": True}

    # C. วิเคราะห์ความนัว / เข้มข้น / เค็ม (Savory & Richness)
    rich_high = ['นัว', 'นัวมาก', 'เข้มข้น', 'กลมกล่อม', 'หอมมัน', 'รสเด็ด', 'ถึงเครื่อง', 'น้ำซุปเด็ด']
    salty_high = ['เค็มมาก', 'เค็มนำ', 'เค็มเกิน', 'เค็มปี๋']
    bland = ['จืด', 'รสจืด', 'ไม่มีรสชาติ', 'จืดชืด']

    richness = {"label": "ไม่ระบุ", "score_pct": 0, "detected": False}
    if any(k in text_clean for k in rich_high):
        richness = {"label": "นัวกลมกล่อม / เข้มข้นถึงเครื่อง", "score_pct": 90, "detected": True}
    elif any(k in text_clean for k in salty_high):
        richness = {"label": "เค็มนำ / ค่อนข้างเค็ม", "score_pct": 80, "detected": True}
    elif any(k in text_clean for k in bland):
        richness = {"label": "รสจืด / ค่อนข้างเบา", "score_pct": 20, "detected": True}

    # D. วิเคราะห์ความเปรี้ยว (Sourness)
    sour_high = ['เปรี้ยวจี๊ด', 'เปรี้ยวมาก', 'มะนาวแท้', 'เปรี้ยวแซ่บ', 'รสเปรี้ยวจี๊ด']
    sour_neg = ['ไม่เปรี้ยว', 'เปรี้ยวไป']
    sourness = {"label": "ไม่ระบุ", "detected": False}
    if any(k in text_clean for k in sour_high):
        sourness = {"label": "เปรี้ยวจี๊ดสดชื่น", "detected": True}
    elif any(k in text_clean for k in sour_neg):
        sourness = {"label": "เปรี้ยวน้อย / ไม่เปรี้ยว", "detected": True}
    elif 'เปรี้ยว' in text_clean:
        sourness = {"label": "เปรี้ยวกลมกล่อม", "detected": True}

    # E. วิเคราะห์มิติประกอบการตัดสินใจ (Decision Aspects)
    # 1) ความคุ้มค่า / ราคา
    value_pos = ['คุ้ม', 'คุ้มค่า', 'คุ้มราคา', 'ราคาถูก', 'ไม่แพง', 'สมราคา', 'ให้เยอะ', 'จานใหญ่', 'อิ่มจุก', 'ปริมาณเยอะ']
    value_neg = ['แพง', 'ราคาสูง', 'แพงเกิน', 'ให้น้อย', 'ไม่คุ้ม', 'ปริมาณน้อย', 'แพงมาก']
    aspect_value = {"status": "neutral", "summary": "ราคาและปริมาณตามมาตรฐาน"}
    if any(k in text_clean for k in value_pos):
        aspect_value = {"status": "positive", "summary": "คุ้มค่าราคา / ปริมาณให้เยอะ"}
    elif any(k in text_clean for k in value_neg):
        aspect_value = {"status": "negative", "summary": "ราคาสูง / ปริมาณค่อนข้างน้อย"}

    # 2) เวลารออาหาร / คิว
    time_pos = ['เร็ว', 'ไว', 'ไม่ต้องรอ', 'อาหารไว', 'ออกไว', 'ทันใจ', 'ไม่ต้องรอนาน']
    time_neg = ['รอนาน', 'คิวยาว', 'ช้า', 'รอครึ่งชั่วโมง', 'รอนานมาก', 'อาหารช้า', 'คิวแน่น']
    aspect_time = {"status": "neutral", "summary": "เวลารออาหารปกติ"}
    if any(k in text_clean for k in time_pos):
        aspect_time = {"status": "positive", "summary": "อาหารออกไว ไม่ต้องรอนาน"}
    elif any(k in text_clean for k in time_neg):
        aspect_time = {"status": "negative", "summary": "รอนาน / ค่อนข้างใช้เวลา"}

    # 3) การบริการ
    serv_pos = ['บริการดี', 'สุภาพ', 'ยิ้มแย้ม', 'น่ารัก', 'ต้อนรับดี', 'ใส่ใจ', 'เป็นกันเอง']
    serv_neg = ['เหวี่ยง', 'ชักสีหน้า', 'บริการแย่', 'ไม่สนใจ', 'ดูถูก', 'หยาบ', 'ไม่บริการ']
    aspect_service = {"status": "neutral", "summary": "การบริการตามมาตรฐาน"}
    if any(k in text_clean for k in serv_pos):
        aspect_service = {"status": "positive", "summary": "บริการสุภาพ เป็นกันเอง"}
    elif any(k in text_clean for k in serv_neg):
        aspect_service = {"status": "negative", "summary": "การบริการบกพร่อง / มีการบ่น"}

    # 4) บรรยากาศและความสะอาด
    place_pos = ['สะอาด', 'บรรยากาศดี', 'น่านั่ง', 'ชิล', 'ติดแอร์', 'ห้องแอร์', 'มีที่จอดรถ', 'ถ่ายรูปสวย']
    place_neg = ['สกปรก', 'เหม็น', 'แออัด', 'ร้อนอบอ้าว', 'แมลงวัน', 'ไม่มีที่จอดรถ', 'หาที่จอดยาก']
    aspect_place = {"status": "neutral", "summary": "บรรยากาศทั่วไป"}
    if any(k in text_clean for k in place_pos):
        aspect_place = {"status": "positive", "summary": "สถานที่สะอาด บรรยากาศดี"}
    elif any(k in text_clean for k in place_neg):
        aspect_place = {"status": "negative", "summary": "สถานที่แออัด / มีข้อจำกัด"}

    # F. สรุปความเห็นเพื่อการตัดสินใจ (Decision Verdict Summary)
    points = []
    if spiciness["detected"]:
        if spiciness["level"] >= 2:
            points.append("เด่นรสจัดจ้านเผ็ดนำ (เหมาะกับคนชอบทานเผ็ด)")
        elif spiciness["level"] == 0:
            points.append("รสไม่เผ็ด ทานง่ายทุกวัย")
    if richness["detected"] and richness["score_pct"] >= 70:
        points.append("รสนัวเข้มข้นถึงเครื่อง")
    if sweetness["detected"] and sweetness["level"] == 2:
        points.append("รสค่อนไปทางหวานนำ")
    if aspect_value["status"] == "positive":
        points.append("จุดเด่นคือความคุ้มค่าและปริมาณ")
    elif aspect_value["status"] == "negative":
        points.append("ข้อควรพิจารณาคือราคาสูงเทียบกับปริมาณ")
    if aspect_time["status"] == "negative":
        points.append("ควรเผื่อเวลารอคิวอาหาร")
    elif aspect_time["status"] == "positive":
        points.append("อาหารออกรวดเร็วทันใจ")
    if aspect_service["status"] == "negative":
        points.append("มีข้อติติงเรื่องการบริการ")
    elif aspect_service["status"] == "positive":
        points.append("บริการประทับใจ")

    if not points:
        decision_verdict = "รีวิวเน้นบอกรสชาติและประสบการณ์ทั่วไป ไม่มีจุดติหรือรสชาติโดดเฉพาะเจาะจงเป็นพิเศษ"
    else:
        decision_verdict = " • ".join(points)

    return {
        "taste": {
            "spiciness": spiciness,
            "sweetness": sweetness,
            "richness": richness,
            "sourness": sourness
        },
        "aspects": {
            "value": aspect_value,
            "time": aspect_time,
            "service": aspect_service,
            "place": aspect_place
        },
        "decision_verdict": decision_verdict
    }

# ----------------------------------------------------
# 5. ตัวอย่าง Presets สำหรับทดสอบ
# ----------------------------------------------------
SAMPLE_PRESETS = [
    {
        "id": "complaint_service_1star",
        "title": "บริการแย่แต่รสชาติดี (1★)",
        "tag": "Success Case",
        "star": 1,
        "text": "เหวี่ยงใส่ลูกค้า ราคาสูงเกินคุณภาพก็ไม่ได้ว่าอะไรเพราะเห็นว่าวัตถุดิบหายาก แต่คิดว่าตัวเองเป็นเจ้าของกิจการจะชักสีหน้าไม่พอใจ ไม่ตอบสนองความต้องการของลูกค้า ก็อยู่บ้านทำกินเองคะไม่ต้องขาย ขายของทำตัวดูถูกลูกค้า บริการจัดว่าแย่ที่สุด"
    },
    {
        "id": "sarcastic_review_1star",
        "title": "ประชดประชันเจ้าของร้าน (1★)",
        "tag": "Success Case",
        "star": 1,
        "text": "วันนั้นที่เลือกทานเพราะเป็นวันเกิดอยากกินอะไรเบาๆ สั่งจานแรกเฝอ แค่ชิมน้ำก็ไม่ผ่านแล้วคะ ไม่ปลื้มมากขนาดปรุงเพิ่ม จานที่สองของแฟนก็ไม่โอเค มองตากันปริบๆ คิดว่าตัวเองเป็นเทวดา ไม่นานเด๋วก็เจ๊ง"
    },
    {
        "id": "praise_but_1star_noise",
        "title": "ชมอาหารแต่กดดาวผิด (1★)",
        "tag": "Failure Case / Inconsistency",
        "star": 1,
        "text": "อาหารอร่อยมาก รสชาติต้นตำรับแท้ๆ วัตถุดิบสดใหม่ กุ้งแม่น้ำเผาตัวใหญ่ มันเยิ้ม พนักงานบริการสุภาพ ยิ้มแย้ม บรรยากาศริมน้ำโรแมนติก แนะนำทุกคนต้องมาลอง"
    },
    {
        "id": "authentic_5star",
        "title": "ร้านโปรดประทับใจ (5★)",
        "tag": "Consistent",
        "star": 5,
        "text": "สุดยอดร้านโปรดในดวงใจ ไปทานซ้ำเกินสิบรอบ เนื้อย่างนุ่มละลายในปาก น้ำจิ้มเด็ดมาก บริการรวดเร็ว เจ้าของร้านน่ารักและเป็นกันเอง คุ้มค่าคุ้มราคาทุกบาททุกสตางค์"
    },
    {
        "id": "neutral_fact_3star",
        "title": "รสชาติปานกลางทั่วไป (3★)",
        "tag": "Neutral",
        "star": 3,
        "text": "รสชาติปานกลาง พอทานได้ ไม่ได้แย่แต่ก็ไม่ได้ว้าว ราคาเหมาะสมกับปริมาณ ที่จอดรถมีน้อย ต้องหาจอดริมถนน"
    },
    {
        "id": "spicy_street_food_4star",
        "title": "ส้มตำปูปลาร้าเผ็ดแซ่บ (4★ เผ็ดจัด-คุ้มค่า-รอนาน)",
        "tag": "Taste & Aspect Focus",
        "star": 4,
        "text": "ส้มตำปูปลาร้าเผ็ดแซ่บจี๊ดถึงใจ พริกสิบเม็ดสะใจมาก ไก่ย่างหนังกรอบอร่อยนัว ราคาถูกให้เยอะคุ้มค่าสุดๆ แต่โต๊ะน้อยและรอนานมาก ใครไม่กินเผ็ดต้องสั่งเผ็ดน้อยนะ"
    },
    {
        "id": "sweet_cafe_dessert_5star",
        "title": "คาเฟ่ขนมหวานละมุน (5★ บรรยากาศดี-บริการสุภาพ)",
        "tag": "Taste & Aspect Focus",
        "star": 5,
        "text": "ขนมปังเนยสดหอมฟุ้ง หวานกำลังดีไม่เลี่ยนเลย ชาไทยนัวกลมกล่อม บรรยากาศร้านสะอาดน่านั่งถ่ายรูปสวยมาก พนักงานบริการสุภาพยิ้มแย้ม แอร์เย็นสบาย"
    }
]

# ----------------------------------------------------
# 6. Route ให้บริการ
# ----------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify(SAMPLE_PRESETS)

@app.route('/api/charts/<chart_name>', methods=['GET'])
def get_chart(chart_name):
    if os.path.exists(os.path.join(OUTPUTS_DIR, chart_name)):
        return send_from_directory(OUTPUTS_DIR, chart_name)
    elif os.path.exists(os.path.join(IMBALANCE_DIR, chart_name)):
        return send_from_directory(IMBALANCE_DIR, chart_name)
    return jsonify({"error": "Chart not found"}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    given_star = data.get('given_star', None)

    if not text:
        return jsonify({"error": "กรุณาใส่ข้อความรีวิว"}), 400

    tokens = thai_tokenizer(text)

    # 1. ทำนายด้วย Baseline (LinearSVC)
    start_time_base = time.perf_counter()
    if vec is not None and baseline_model is not None:
        X_tfidf = vec.transform([text])
        base_pred_idx = int(baseline_model.predict(X_tfidf)[0])  # 0 to 4
        dec_func = baseline_model.decision_function(X_tfidf)[0]
        exp_dec = np.exp(dec_func - np.max(dec_func))
        base_probs = (exp_dec / exp_dec.sum()).tolist()
    else:
        base_pred_idx = 2
        base_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    latency_base_ms = (time.perf_counter() - start_time_base) * 1000.0

    # 2. ทำนายด้วย BiLSTM
    start_time_bi = time.perf_counter()
    if bilstm_model is not None and word2idx is not None:
        indices = [word2idx.get(w, 1) for w in tokens[:max_len]]
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = bilstm_model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            bi_pred_idx = int(np.argmax(probs))
            bi_probs = probs.tolist()
    else:
        bi_pred_idx = 2
        bi_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    latency_bi_ms = (time.perf_counter() - start_time_bi) * 1000.0

    base_pred_star = base_pred_idx + 1
    bi_pred_star = bi_pred_idx + 1

    # 3. วิเคราะห์ความขัดแย้ง (Consistency Analysis)
    consistency_status = "UNKNOWN"
    consistency_detail = ""
    difference = None

    if given_star is not None:
        try:
            given_star = int(given_star)
            diff = abs(bi_pred_star - given_star)
            difference = diff

            if diff == 0:
                consistency_status = "CONSISTENT"
                consistency_detail = f"คะแนนรีวิว ({given_star} ดาว) สอดคล้องสมบูรณ์กับเนื้อหาข้อความที่โมเดลตรวจพบ ({bi_pred_star} ดาว)"
            elif diff == 1:
                consistency_status = "SLIGHT_DIVERGENCE"
                consistency_detail = f"คะแนนรีวิว ({given_star} ดาว) ใกล้เคียงกับเนื้อหา ({bi_pred_star} ดาว) ห่างกันเพียง 1 ระดับ อยู่ในเกณฑ์ปกติของความรู้สึกบุคคล"
            elif diff >= 2:
                consistency_status = "SEVERE_INCONSISTENT"
                if given_star <= 2 and bi_pred_star >= 4:
                    consistency_detail = f"ตรวจพบความขัดแย้งรุนแรง! รีวิวให้ {given_star} ดาว แต่เนื้อหาชมหนักระดับ {bi_pred_star} ดาว (น่าสงสัย: ผู้ใช้อาจกดดาวผิด หรือมีข้อความดักฟัง)"
                elif given_star >= 4 and bi_pred_star <= 2:
                    consistency_detail = f"ตรวจพบความขัดแย้งรุนแรง! รีวิวให้ {given_star} ดาว แต่เนื้อหาบ่นด่ารุนแรงระดับ {bi_pred_star} ดาว (น่าสงสัย: รีวิวประชด, ปั่นเรตติ้ง, หรือบอท)"
                else:
                    consistency_detail = f"ตรวจพบความขัดแย้งชัดเจน! ดาวที่กด ({given_star} ดาว) ต่างจากเนื้อหาจริง ({bi_pred_star} ดาว) ถึง {diff} ระดับ"
        except Exception:
            pass

    # 4. Sentiment Polarity Tokens
    positive_words = {'อร่อย', 'สด', 'ดี', 'คุ้ม', 'ชอบ', 'ประทับใจ', 'เด็ด', 'สะอาด', 'รวดเร็ว', 'สุภาพ', 'เลิศ', 'โปรด', 'หอม', 'นุ่ม', 'นัว', 'กรอบ', 'ฟุ้ง'}
    negative_words = {'แย่', 'เหวี่ยง', 'แพง', 'ช้า', 'ดูถูก', 'เจ๊ง', 'ไม่ผ่าน', 'ไม่ปลื้ม', 'ชักสีหน้า', 'สกปรก', 'ไม่สุก', 'รอนาน', 'เลี่ยน'}

    token_highlights = []
    for t in tokens:
        if t in positive_words:
            token_highlights.append({"token": t, "type": "positive"})
        elif t in negative_words:
            token_highlights.append({"token": t, "type": "negative"})
        else:
            token_highlights.append({"token": t, "type": "neutral"})

    # 5. สกัดมิติรสชาติและข้อมูลประกอบการตัดสินใจ
    decision_insights = extract_taste_and_aspects(text, tokens)

    return jsonify({
        "input_text": text,
        "token_count": len(tokens),
        "tokens": token_highlights,
        "given_star": given_star,
        "consistency": {
            "status": consistency_status,
            "detail": consistency_detail,
            "diff": difference
        },
        "proposed_bilstm": {
            "predicted_star": bi_pred_star,
            "confidence": float(round(bi_probs[bi_pred_idx] * 100, 2)),
            "probabilities": [float(round(p * 100, 2)) for p in bi_probs],
            "latency_ms": float(round(latency_bi_ms, 2))
        },
        "baseline_linearsvc": {
            "predicted_star": base_pred_star,
            "confidence": float(round(base_probs[base_pred_idx] * 100, 2)),
            "probabilities": [float(round(p * 100, 2)) for p in base_probs],
            "latency_ms": float(round(latency_base_ms, 2))
        },
        "decision_insights": decision_insights
    })

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    return jsonify({
        "models": [
            {
                "name": "Baseline (LinearSVC + TF-IDF)",
                "type": "Traditional Machine Learning",
                "macro_f1": 0.4176,
                "star1_recall": 14.46,
                "model_size_mb": 2.18,
                "latency_mean_ms": 4.39,
                "latency_p95_ms": 5.37,
                "throughput_req_sec": 687.9,
                "notes": "โมเดลขนาดเล็ก ทำงานเร็ว แต่แพ้เรื่องบริบทสองทิศทางและคำประชด"
            },
            {
                "name": "Proposed (BiLSTM + thai2vec)",
                "type": "Deep Learning (Sequential)",
                "macro_f1": 0.4185,
                "star1_recall": 54.22,
                "model_size_mb": 63.25,
                "latency_mean_ms": 10.18,
                "latency_p95_ms": 10.85,
                "throughput_req_sec": 1667.3,
                "notes": "จับความขัดแย้งของรีวิว 1 ดาวได้ดีขึ้นเกือบ 4 เท่า ตอบสนอง Real-time ต่ำกว่า 15 ms"
            }
        ],
        "tradeoff_rule": "เกณฑ์ระบบ: Latency < 100ms พร้อม Recall คลาส 1 ดาวสูงสุด -> BiLSTM ชนะเลิศ"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n==================================================")
    print(f"🚀 Kin Nai Dee Web Application is running!")
    print(f"👉 Local URL: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
