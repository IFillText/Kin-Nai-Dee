"""
clean_data.py
----------------
สคริปต์สำหรับทำความสะอาดและตัดคำข้อมูล Wongnai Reviews Dataset
ก่อนนำไปเทรนโมเดล

ขั้นตอนที่ทำ:
1. โหลดข้อมูล (train_data.csv / test_data.csv)
2. แทนที่ Unusual line terminators (\u2028, \u2029)
3. ถอดรหัสและลบ HTML tags & URLs
4. Normalize ภาษาไทยด้วย PyThaiNLP (แก้วรรณยุกต์/สระซ้อนที่พิมพ์ผิดลำดับ)
5. ลดทอนตัวอักษรลากเสียงซ้ำซากเกิน 2 ตัว (เช่น อร่อยยยยย -> อร่อยย, 55555 -> 55) ป้องกัน Overfitting
6. กรองอักขระพิเศษ เก็บเฉพาะอักษรไทย อังกฤษ ตัวเลข และช่องว่าง
7. ปรับตัวอักษรภาษาอังกฤษเป็นตัวพิมพ์เล็ก (lowercase)
8. ลบช่องว่างซ้ำซ้อน (extra whitespace)
9. ตรวจสอบ star_rating (0-4) และลบ duplicate reviews (เฉพาะ Train Set)
10. ตัดคำภาษาไทยล่วงหน้าด้วย PyThaiNLP (engine='newmm') เพิ่มเป็นคอลัมน์ tokens
11. บันทึกไฟล์ cleaned_train_data.csv และ cleaned_test_data.csv ลงใน data/
"""

import os
import sys
import re
import html
import time
import pandas as pd
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize

# กำหนด encoding ของ stdout ให้รองรับภาษาไทยบน Windows
sys.stdout.reconfigure(encoding='utf-8')


def clean_text(text: str) -> str:
    """
    ทำความสะอาดข้อความรีวิว 1 รายการ
    - แปลง NaN / None ให้เป็นสตริงว่าง
    - ลบ Unusual line terminators (\u2028, \u2029)
    - ถอดรหัส HTML entities & ลบ HTML tags (<br>, <p>, ฯลฯ)
    - ลบ URL
    - Normalize ภาษาไทย (แก้สระ/วรรณยุกต์ซ้อน)
    - ลดทอนตัวอักษรลากเสียงซ้ำซากเกิน 2 ตัว เพื่อลดคำขยะและป้องกัน Overfitting
    - เก็บเฉพาะตัวอักษรไทย, อังกฤษ, ตัวเลข, และช่องว่าง
    - ปรับภาษาอังกฤษเป็นตัวพิมพ์เล็ก (lowercase)
    - ลบช่องว่างซ้ำซ้อน และตัดช่องว่างหน้า-หลัง
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. จัดการตัวตัดบรรทัดแปลก (\u2028 Line Separator, \u2029 Paragraph Separator)
    text = text.replace("\u2028", " ").replace("\u2029", " ")

    # 2. ถอดรหัส HTML entities (เช่น &amp; -> &) และลบ HTML tags
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. ลบ URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Normalize ภาษาไทย (แก้วรรณยุกต์และสระซ้อนที่พิมพ์ผิดลำดับ)
    text = normalize(text)

    # 5. ลดทอนตัวอักษรลากเสียงซ้ำซากเกิน 2 ตัว (เช่น อร่อยยยยย -> อร่อยย, มากกกก -> มากก)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 6. เก็บเฉพาะตัวอักษรไทย (\u0E00-\u0E7F), อังกฤษ, ตัวเลข, และช่องว่าง
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s]", " ", text)

    # 7. ปรับภาษาอังกฤษเป็นตัวพิมพ์เล็ก (lowercase)
    text = text.lower()

    # 8. ลบช่องว่างซ้ำซ้อนให้เหลือช่องว่างเดียว และตัดหน้า-หลัง
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_series(series: pd.Series, desc: str = "Train") -> pd.Series:
    """
    ตัดคำภาษาไทยล่วงหน้าด้วย PyThaiNLP (newmm engine)
    พร้อมแสดงแถบความคืบหน้า (Progress Report)
    """
    total = len(series)
    print(f"\n[เริ่มตัดคำ] กำลังตัดคำภาษาไทย ({desc} Set) ทั้งหมด {total:,} รายการ...")
    start_time = time.time()
    results = []

    for i, text in enumerate(series, 1):
        words = word_tokenize(str(text), engine='newmm', keep_whitespace=False)
        tokens_str = " ".join([w.strip() for w in words if len(w.strip()) > 0])
        results.append(tokens_str)

        if i % 5000 == 0 or i == total:
            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / speed if speed > 0 else 0
            print(f"  ความคืบหน้า: {i:,}/{total:,} ({i/total*100:.1f}%) | ความเร็ว: {speed:.0f} แถว/วิ | เวลาที่เหลือ: ~{eta:.0f} วิ")

    total_time = time.time() - start_time
    print(f"[ตัดคำเสร็จสิ้น] ใช้เวลาทั้งหมด: {total_time:.2f} วินาที ({total/total_time:.1f} แถว/วิ)\n")
    return pd.Series(results, index=series.index)


def load_data(path: str) -> pd.DataFrame:
    """โหลดไฟล์ csv ของ Wongnai Reviews Dataset"""
    df = pd.read_csv(path, encoding='utf-8')
    return df


def clean_dataframe(df: pd.DataFrame,
                    text_col: str = "review_body",
                    star_col: str = "star_rating",
                    is_train: bool = True) -> pd.DataFrame:
    """
    รับ DataFrame ดิบ แล้วคืนค่า DataFrame ที่สะอาดแล้ว พร้อมคอลัมน์ tokens ที่ตัดคำแล้ว
    - is_train=True: ตรวจสอบ star_rating (0-4) และลบ duplicate reviews
    - is_train=False: สำหรับ Test set ที่ไม่มี star_rating และต้องเก็บแถวให้ครบตาม ID
    """
    df = df.copy()
    n_before = len(df)

    # --- ขั้นที่ 1: ลบแถวที่ไม่มีข้อความรีวิว ---
    if is_train and star_col in df.columns:
        df = df.dropna(subset=[text_col, star_col])
    else:
        df[text_col] = df[text_col].fillna("")

    # --- ขั้นที่ 2: ทำความสะอาดข้อความ ---
    print(f"กำลังทำความสะอาดข้อความ ({'Train' if is_train else 'Test'} Set)...")
    df[text_col] = df[text_col].apply(clean_text)

    # --- ขั้นที่ 3: กรณี Train set ตรวจสอบข้อความว่างและ star_rating ---
    if is_train and star_col in df.columns:
        # ลบแถวที่ข้อความว่างเปล่าหลังทำความสะอาดแล้ว
        df = df[df[text_col].str.len() > 0]

        # ตรวจสอบ star_rating ให้เป็นตัวเลข 0-4 เท่านั้น (0 = 1 ดาว, ..., 4 = 5 ดาว)
        df[star_col] = pd.to_numeric(df[star_col], errors="coerce")
        df = df.dropna(subset=[star_col])
        df[star_col] = df[star_col].astype(int)
        df = df[df[star_col].between(0, 4)]

        # --- ขั้นที่ 4: ลบข้อมูลซ้ำ (รีวิวข้อความเดียวกันเป๊ะ) ---
        df = df.drop_duplicates(subset=[text_col])

    n_after = len(df)
    print(f"จำนวนแถวก่อนทำความสะอาด: {n_before:,}")
    print(f"จำนวนแถวหลังทำความสะอาด: {n_after:,}")
    print(f"ลบออกไปทั้งหมด: {n_before - n_after:,} แถว")

    # แสดงการกระจายตัวของดาว หลังทำความสะอาด (ถ้าเป็นชุด Train)
    if is_train and star_col in df.columns:
        print("\nการกระจายตัวของ star_rating (0-4) หลังทำความสะอาด:")
        dist = df[star_col].value_counts().sort_index()
        for star, count in dist.items():
            pct = (count / n_after) * 100
            print(f"  Class {star} ({star + 1} ดาว): {count:,} แถว ({pct:.2f}%)")

    # --- ขั้นที่ 5: ตัดคำภาษาไทยล่วงหน้า (Pre-tokenization) ---
    df['tokens'] = tokenize_series(df[text_col], desc="Train" if is_train else "Test")

    return df.reset_index(drop=True)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data", "train_data.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join(base_dir, "raw_backup", "train_data.csv")

    test_path = os.path.join(base_dir, "data", "test_data.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(base_dir, "raw_backup", "test_data.csv")

    text_col = "review_body"
    star_col = "star_rating"

    print("========================================")
    print("===== 1. ทำความสะอาดและตัดคำ Train Set =====")
    print("========================================")
    if os.path.exists(train_path):
        train_df = load_data(train_path)
        train_clean = clean_dataframe(train_df, text_col=text_col, star_col=star_col, is_train=True)
        train_out = os.path.join(base_dir, "data", "cleaned_train_data.csv")
        train_clean.to_csv(train_out, index=False, encoding='utf-8')
        print(f"[สำเร็จ] บันทึกไฟล์ {train_out} เรียบร้อย\n")
    else:
        print(f"[ไม่พบไฟล์] {train_path}")

    print("========================================")
    print("===== 2. ทำความสะอาดและตัดคำ Test Set ===== ")
    print("========================================")
    if os.path.exists(test_path):
        test_df = load_data(test_path)
        test_clean = clean_dataframe(test_df, text_col=text_col, is_train=False)
        test_out = os.path.join(base_dir, "data", "cleaned_test_data.csv")
        test_clean.to_csv(test_out, index=False, encoding='utf-8')
        print(f"[สำเร็จ] บันทึกไฟล์ {test_out} เรียบร้อย")
    else:
        print(f"[ไม่พบไฟล์] {test_path}")


if __name__ == "__main__":
    main()
