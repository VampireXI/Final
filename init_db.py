"""
Initialize SQLite Database from CatFood_cleaned.csv
====================================================
โหลดข้อมูลที่ทำความสะอาดแล้วเข้าสู่ SQLite database
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "catfood.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "CatFood_cleaned.csv")


def init_database():
    """สร้างและโหลดข้อมูลเข้า SQLite database"""
    print("=" * 50)
    print("Initializing SQLite Database")
    print("=" * 50)

    # อ่าน CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV: {df.shape[0]} rows x {df.shape[1]} columns")

    # สร้าง SQLite database
    conn = sqlite3.connect(DB_PATH)

    # บันทึกตาราง survey_responses
    df.to_sql("survey_responses", conn, if_exists="replace", index=True, index_label="id")
    print(f"Created table: survey_responses")

    # สร้างตาราง model_results สำหรับเก็บผลลัพธ์โมเดล
    conn.execute("DROP TABLE IF EXISTS model_results")
    conn.execute("""
        CREATE TABLE model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1_score REAL,
            cv_accuracy REAL,
            roc_auc REAL,
            is_best INTEGER DEFAULT 0
        )
    """)

    # ใส่ผลโมเดลจาก supervise.py
    model_data = [
        ("DecisionTree",       0.7119, 0.8710, 0.6750, 0.7606, 0.8367, 0.729, 0),
        ("RandomForest",       0.8814, 0.9024, 0.9250, 0.9136, 0.9864, 0.926, 1),
        ("LogisticRegression", 0.6441, 0.7879, 0.6500, 0.7123, 0.7822, 0.670, 0),
    ]

    conn.executemany("""
        INSERT INTO model_results
        (model_name, accuracy, precision_score, recall, f1_score, cv_accuracy, roc_auc, is_best)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, model_data)

    conn.commit()
    print(f"Created table: model_results (3 rows)")

    # ตรวจสอบ
    cursor = conn.execute("SELECT COUNT(*) FROM survey_responses")
    print(f"\nVerification:")
    print(f"  survey_responses: {cursor.fetchone()[0]} rows")
    cursor = conn.execute("SELECT COUNT(*) FROM model_results")
    print(f"  model_results:    {cursor.fetchone()[0]} rows")

    conn.close()
    print(f"\nDatabase saved: {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    init_database()
