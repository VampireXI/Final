"""
Flask Web Application — Supervised Learning Dashboard
======================================================
เว็บแดชบอร์ดแสดงผลการวิเคราะห์ Supervised Learning
เชื่อมต่อ SQLite Database + แสดงกราฟ + สถิติ
"""

from flask import Flask, render_template, jsonify, send_from_directory
import sqlite3
import pandas as pd
import os
import json

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "catfood.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def get_db():
    """เชื่อมต่อ SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    """หน้าหลัก — Supervised Learning Dashboard"""
    conn = get_db()

    # --- ดึงผลลัพธ์โมเดลจาก database ---
    models = conn.execute("SELECT * FROM model_results ORDER BY f1_score DESC").fetchall()
    models = [dict(row) for row in models]

    best_model = conn.execute(
        "SELECT * FROM model_results WHERE is_best = 1"
    ).fetchone()
    best_model = dict(best_model) if best_model else models[0]

    # --- สถิติข้อมูลจาก database ---
    total_rows = conn.execute("SELECT COUNT(*) as cnt FROM survey_responses").fetchone()["cnt"]

    # การกระจายตัวของ target
    target_dist = conn.execute("""
        SELECT packaging_influence, COUNT(*) as cnt
        FROM survey_responses
        GROUP BY packaging_influence
    """).fetchall()
    target_dist = {str(row["packaging_influence"]): row["cnt"] for row in target_dist}

    # การกระจายอายุ
    age_dist = conn.execute("""
        SELECT age, COUNT(*) as cnt
        FROM survey_responses
        GROUP BY age
        ORDER BY cnt DESC
    """).fetchall()
    age_dist = [dict(row) for row in age_dist]

    # การกระจายเพศ
    gender_dist = conn.execute("""
        SELECT gender, COUNT(*) as cnt
        FROM survey_responses
        GROUP BY gender
        ORDER BY cnt DESC
    """).fetchall()
    gender_dist = [dict(row) for row in gender_dist]

    # สถานภาพสมรส
    marital_dist = conn.execute("""
        SELECT marital_status, COUNT(*) as cnt
        FROM survey_responses
        GROUP BY marital_status
        ORDER BY cnt DESC
    """).fetchall()
    marital_dist = [dict(row) for row in marital_dist]

    # ค่าเฉลี่ย factor scores
    factor_avg = conn.execute("""
        SELECT
            ROUND(AVG(factor_natural), 2) as factor_natural,
            ROUND(AVG(factor_imported), 2) as factor_imported,
            ROUND(AVG(factor_taste), 2) as factor_taste,
            ROUND(AVG(factor_foreign), 2) as factor_foreign,
            ROUND(AVG(factor_brand_fame), 2) as factor_brand_fame
        FROM survey_responses
    """).fetchone()
    factor_avg = dict(factor_avg)

    # ค่าเฉลี่ย packaging attribute scores
    pkg_avg = conn.execute("""
        SELECT
            ROUND(AVG(pkg_premium), 2) as pkg_premium,
            ROUND(AVG(pkg_cat_image), 2) as pkg_cat_image,
            ROUND(AVG(pkg_kibble_image), 2) as pkg_kibble_image,
            ROUND(AVG(pkg_ingredient_image), 2) as pkg_ingredient_image,
            ROUND(AVG(pkg_eco_friendly), 2) as pkg_eco_friendly,
            ROUND(AVG(pkg_origin_symbol), 2) as pkg_origin_symbol,
            ROUND(AVG(pkg_benefit_symbol), 2) as pkg_benefit_symbol,
            ROUND(AVG(pkg_guarantee), 2) as pkg_guarantee
        FROM survey_responses
    """).fetchone()
    pkg_avg = dict(pkg_avg)

    # ค่าเฉลี่ยแต่ละ Option (want_buy)
    option_avg_query = ", ".join(
        [f"ROUND(AVG(opt{i}_want_buy), 2) as opt{i}" for i in range(1, 11)]
    )
    option_avg = conn.execute(
        f"SELECT {option_avg_query} FROM survey_responses"
    ).fetchone()
    option_avg = dict(option_avg)

    conn.close()

    # --- รายชื่อกราฟ ---
    charts = [
        {"file": "1_model_comparison.png", "title": "Model Performance Comparison", "desc": "เปรียบเทียบ Accuracy, Precision, Recall, F1, CV ทั้ง 3 โมเดล"},
        {"file": "2_confusion_matrices.png", "title": "Confusion Matrices", "desc": "แสดง Confusion Matrix ของแต่ละโมเดล"},
        {"file": "3_roc_curves.png", "title": "ROC Curves", "desc": "เปรียบเทียบ ROC-AUC ของทุกโมเดล"},
        {"file": "4_feature_importance.png", "title": "Feature Importance (Top 15)", "desc": "Feature ที่สำคัญที่สุดจาก RandomForest"},
        {"file": "5_decision_tree.png", "title": "Decision Tree Visualization", "desc": "โครงสร้าง Decision Tree (แสดง 3 ระดับ)"},
        {"file": "6_cv_boxplot.png", "title": "Cross-Validation Box Plot", "desc": "ความเสถียรของโมเดลจาก 5-Fold CV"},
        {"file": "7_classification_report_heatmap.png", "title": "Classification Report", "desc": "Heatmap ของ Precision/Recall/F1 แยกตาม Class"},
        {"file": "8_target_distribution.png", "title": "Target Distribution", "desc": "สัดส่วน packaging_influence (มีผล/ไม่มีผล)"},
    ]

    return render_template(
        "supervise.html",
        models=models,
        best_model=best_model,
        total_rows=total_rows,
        target_dist=target_dist,
        age_dist=age_dist,
        gender_dist=gender_dist,
        marital_dist=marital_dist,
        factor_avg=factor_avg,
        pkg_avg=pkg_avg,
        option_avg=option_avg,
        charts=charts,
    )


@app.route("/output/<path:filename>")
def serve_chart(filename):
    """เสิร์ฟไฟล์กราฟจากโฟลเดอร์ output"""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/survey-data")
def api_survey_data():
    """API endpoint: ดึงข้อมูลแบบสำรวจทั้งหมด"""
    conn = get_db()
    rows = conn.execute("SELECT * FROM survey_responses LIMIT 50").fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route("/api/model-results")
def api_model_results():
    """API endpoint: ผลลัพธ์โมเดล"""
    conn = get_db()
    rows = conn.execute("SELECT * FROM model_results").fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Supervised Learning Dashboard")
    print(f"  Database: {DB_PATH}")
    print(f"  Charts:   {OUTPUT_DIR}")
    print("  URL:      http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
