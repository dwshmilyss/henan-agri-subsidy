#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
farm_subsidy_ai_pipeline.py

功能（端到端）：
- 加载 Excel 数据（默认 /mnt/data/table_data_all.xlsx）
- 数据清洗与字段解析
- 探索性数据分析（EDA）输出图与汇总表
- 异常检测（IsolationForest + 基于统计的 z-score）
- 重复/疑似超录检测（基于购机者/出厂编号/时间窗口）
- 价格/补贴合理性回归模型（LightGBM）
- 时间序列预测（Prophet）按乡镇/品目预测未来需求
- 报告导出（CSV + 基本 PDF/PNG 图）
- 简单告警（发送 email 或 POST webhook）
"""

import os
import argparse
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from prophet import Prophet
from joblib import dump, load
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dateutil.parser import parse as dateparse

# ----------------------------
# 配置（可修改）
# ----------------------------
DEFAULT_INPUT = "data/table_data_all.xlsx"
DEFAULT_OUTDIR = "ai/output"
RANDOM_SEED = 42

# 用于告警的示例配置（可改）
ALERT_CONFIG = {
    "webhook_url": "",  # 如果你有告警 webhook（如钉钉/飞书/自建服务），填 URL
    "smtp": {
        "enabled": False,
        "host": "smtp.example.com",
        "port": 587,
        "username": "monitor@example.com",
        "password": "yourpassword",
        "from_addr": "monitor@example.com",
        "to_addrs": ["admin@example.com"]
    }
}

# ----------------------------
# 工具函数
# ----------------------------
def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

def load_data(path):
    """加载 Excel（若有多个 sheet，可按需改造）"""
    print(f"Loading data from {path} ...")
    df = pd.read_excel(path, engine="openpyxl", dtype=str)  # 以字符串读取避免类型混乱
    # 清理空列和空行
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df

def parse_and_cast(df):
    """解析并转换常用字段类型"""
    d = df.copy()
    # 规范列名（去空格）
    d.columns = [c.strip() for c in d.columns]

    # 尝试解析数字字段
    num_cols = ["购买数量(台)", "单台销售价格(元)", "单台中央补贴额(元)", "总补贴额(元)"]
    for col in num_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # 解析日期
    if "购机日期" in d.columns:
        d["购机日期"] = pd.to_datetime(d["购机日期"], errors="coerce")

    # 标准化：出厂编号去空格，上大写
    if "出厂编号" in d.columns:
        d["出厂编号"] = d["出厂编号"].astype(str).str.strip().replace({"None":"", "nan":""})
    # 购机者姓名修剪
    if "购机者姓名" in d.columns:
        d["购机者姓名"] = d["购机者姓名"].astype(str).str.strip()
    return d

# ----------------------------
# EDA & 可视化
# ----------------------------
def basic_eda(df, outdir):
    """生成基础统计与图表"""
    ensure_outdir(outdir)
    summary = {}
    summary['total_rows'] = len(df)
    # 地域分布
    if "所在乡(镇)" in df.columns:
        region_counts = df["所在乡(镇)"].value_counts()
        region_counts.to_csv(os.path.join(outdir, "region_counts.csv"))
        # 绘图
        plt.figure(figsize=(8,6))
        region_counts.head(20).plot(kind="bar")
        plt.title("Top 20 乡镇购机数量")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "region_top20.png"))
        plt.close()
        summary['regions_top20'] = region_counts.head(20).to_dict()
    # 品目分布
    if "机具品目" in df.columns:
        item_counts = df["机具品目"].value_counts()
        item_counts.to_csv(os.path.join(outdir, "item_counts.csv"))
        plt.figure(figsize=(8,6))
        item_counts.plot(kind="bar")
        plt.title("机具品目分布")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "item_counts.png"))
        plt.close()
        summary['item_counts'] = item_counts.to_dict()
    # 价格分布
    if "单台销售价格(元)" in df.columns:
        plt.figure(figsize=(8,6))
        df["单台销售价格(元)"].dropna().astype(float).plot(kind="hist", bins=50)
        plt.title("单台销售价格分布")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "price_hist.png"))
        plt.close()
        summary['price_stats'] = df["单台销售价格(元)"].describe().to_dict()
    # 导出 summary
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("EDA outputs saved to", outdir)

# ----------------------------
# 重复与伪造检测
# ----------------------------
def detect_duplicates(df, outdir, time_window_days=30):
    """
    检测重复购机或短期内高频购机：
    - 基于购机者姓名 + 机具品目 + 购买日期窗口
    - 基于出厂编号重复
    """
    ensure_outdir(outdir)
    d = df.copy()
    results = {}

    # 1) 出厂编号重复
    if "出厂编号" in d.columns:
        dup_serial = d[d["出厂编号"].duplicated(keep=False) & (d["出厂编号"].notna())]
        dup_serial.to_csv(os.path.join(outdir, "dup_serials.csv"), index=False)
        results['dup_serial_count'] = len(dup_serial)

    # 2) 同人同类短时重复购买（例如 30 天内多次）
    if "购机者姓名" in d.columns and "机具品目" in d.columns and "购机日期" in d.columns:
        mask = d["购机日期"].notna()
        tmp = d[mask].sort_values("购机日期")
        tmp['购_key'] = tmp['购机者姓名'].astype(str) + "||" + tmp['机具品目'].astype(str)
        repeated = []
        grouped = tmp.groupby('购_key')
        for key, group in grouped:
            dates = group["购机日期"].sort_values()
            # sliding window
            for i in range(len(dates)-1):
                if (dates.iloc[i+1] - dates.iloc[i]).days <= time_window_days:
                    # 标记所有在该 key 下的记录为可疑
                    repeated.append(group.index.tolist())
        # flatten
        repeated_idx = sorted(set([idx for sub in repeated for idx in sub]))
        suspicious = tmp.loc[repeated_idx]
        suspicious.to_csv(os.path.join(outdir, "suspicious_quick_repeat.csv"), index=False)
        results['suspicious_quick_repeat_count'] = len(suspicious)
    print("Duplicate detection results saved:", outdir)
    return results

# ----------------------------
# 基于统计的价格异常（z-score）
# ----------------------------
def price_anomaly_zscore(df, price_col="单台销售价格(元)", threshold=3.0):
    d = df.copy()
    if price_col not in d.columns:
        return d
    prices = d[price_col].astype(float)
    mean = prices.mean()
    std = prices.std()
    d['price_zscore'] = (prices - mean) / std
    d['price_stat_anomaly'] = d['price_zscore'].abs() > threshold
    return d

# ----------------------------
# IsolationForest 异常检测（多维）
# ----------------------------
def anomaly_isolation_forest(df, features, outdir, contamination=0.02):
    ensure_outdir(outdir)
    d = df.copy()
    X = d[features].fillna(0).values
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_SEED)
    clf.fit(X)
    preds = clf.predict(X)  # -1 异常, 1 正常
    d['iforest_label'] = preds
    d['iforest_anomaly'] = d['iforest_label'] == -1
    d.to_csv(os.path.join(outdir, "iforest_results.csv"), index=False)
    dump(clf, os.path.join(outdir, "iforest_model.joblib"))
    print("IsolationForest done, results:", os.path.join(outdir, "iforest_results.csv"))
    return d, clf

# ----------------------------
# 价格回归模型（LightGBM）
# ----------------------------
def train_price_model(df, outdir, target_col="单台销售价格(元)"):
    ensure_outdir(outdir)
    d = df.copy()
    # 简单特征：机具品目、生产厂家、购买机型、所在乡(镇)、经销商、购买数量
    features = []
    cat_cols = []
    for col in ["机具品目", "生产厂家", "购买机型", "所在乡(镇)", "经销商"]:
        if col in d.columns:
            d[col] = d[col].astype(str).fillna("NA")
            cat_cols.append(col)
            features.append(col)
    if "购买数量(台)" in d.columns:
        d["购买数量(台)"] = pd.to_numeric(d["购买数量(台)"], errors="coerce").fillna(1)
        features.append("购买数量(台)")

    # drop rows without target
    d = d[pd.to_numeric(d[target_col], errors="coerce").notna()]
    if len(d) < 50:
        print("Not enough rows to train price model.")
        return None, None

    # one-hot encoding for cats (lightgbm can handle categorical indices, but here we use simple encoding)
    X = pd.get_dummies(d[features], dummy_na=True)
    y = d[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': RANDOM_SEED
    }
    print("Training LightGBM model ...")
    gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=50, verbose_eval=50)
    # 预测并计算误差
    preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"Price model trained. RMSE={rmse:.2f}")

    # 保存
    dump(gbm, os.path.join(outdir, "price_model.joblib"))
    X_test.assign(pred=preds, actual=y_test).to_csv(os.path.join(outdir, "price_model_test_preds.csv"))
    return gbm, X_train.columns.tolist()

# ----------------------------
# 时间序列预测（按乡镇或按品目）
# ----------------------------
def timeseries_forecast(df, group_by_col, date_col="购机日期", value_col="购买数量(台)", outdir="./output/ts"):
    ensure_outdir(outdir)
    if group_by_col not in df.columns or date_col not in df.columns:
        print("Timeseries inputs missing")
        return
    groups = df.groupby(group_by_col)
    results = {}
    for name, g in groups:
        ts = g[[date_col, value_col]].dropna()
        if len(ts) < 10:
            continue
        ts = ts.copy()
        ts.columns = ["ds", "y"]
        ts['y'] = pd.to_numeric(ts['y'], errors='coerce').fillna(0)
        m = Prophet()
        try:
            m.fit(ts)
        except Exception as e:
            print(f"Prophet fit failed for {name}: {e}")
            continue
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        # 保存结果
        safe_name = str(name).replace("/", "_").replace(" ", "_")
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(os.path.join(outdir, f"forecast_{safe_name}.csv"), index=False)
        # plot
        fig = m.plot(forecast)
        fig.savefig(os.path.join(outdir, f"forecast_{safe_name}.png"))
        plt.close(fig)
        results[name] = forecast
    print("Time series forecasts saved to", outdir)
    return results

# ----------------------------
# 报告与告警
# ----------------------------
def send_webhook(webhook_url, payload):
    if not webhook_url:
        print("Webhook URL empty, skipping.")
        return False
    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        print("Webhook sent, status:", r.status_code)
        return r.status_code == 200
    except Exception as e:
        print("Webhook send failed:", e)
        return False

def send_email(smtp_cfg, subject, body, attachments=None):
    if not smtp_cfg or not smtp_cfg.get("enabled"):
        print("SMTP disabled or not configured.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_cfg['from_addr']
        msg['To'] = ", ".join(smtp_cfg['to_addrs'])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        # attachments: list of file paths
        for path in (attachments or []):
            with open(path, 'rb') as f:
                part = MIMEApplication(f.read())
                part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
                msg.attach(part)
        server = smtplib.SMTP(smtp_cfg['host'], smtp_cfg['port'], timeout=10)
        server.starttls()
        server.login(smtp_cfg['username'], smtp_cfg['password'])
        server.sendmail(smtp_cfg['from_addr'], smtp_cfg['to_addrs'], msg.as_string())
        server.quit()
        print("Email sent to", smtp_cfg['to_addrs'])
        return True
    except Exception as e:
        print("Email send failed:", e)
        return False

# ----------------------------
# 主流程
# ----------------------------
def main(args):
    input_path = args.input
    outdir = args.outdir
    ensure_outdir(outdir)
    df_raw = load_data(input_path)
    df = parse_and_cast(df_raw)

    # 基础 EDA
    basic_eda(df, outdir)

    # 重复/伪造检测
    dup_results = detect_duplicates(df, outdir)

    # 价格统计异常检测
    df_price = price_anomaly_zscore(df)
    df_price.to_csv(os.path.join(outdir, "price_zscore_results.csv"), index=False)

    # IsolationForest 异常检测（选择数值特征）
    features = []
    for c in ["单台销售价格(元)", "单台中央补贴额(元)", "总补贴额(元)", "购买数量(台)"]:
        if c in df.columns:
            features.append(c)
    # 将缺失值填 0
    if features:
        df_if, clf = anomaly_isolation_forest(df, features, outdir)

    # 价格模型（LightGBM）
    price_model, feature_names = train_price_model(df, outdir)

    # 时间序列预测（按乡镇）
    ts_results = timeseries_forecast(df, group_by_col="所在乡(镇)", outdir=os.path.join(outdir, "ts"))

    # 生成告警示例（当发现 > N 个异常记录时告警）
    anomalies_csv = os.path.join(outdir, "iforest_results.csv")
    try:
        a_df = pd.read_csv(anomalies_csv)
        n_anom = int(a_df['iforest_anomaly'].sum())
    except Exception:
        n_anom = 0
    if n_anom > 10:
        payload = {"title": "农机补贴异常预警", "text": f"检测到 {n_anom} 条异常记录，请及时核查。"}
        if ALERT_CONFIG.get("webhook_url"):
            send_webhook(ALERT_CONFIG["webhook_url"], payload)
        if ALERT_CONFIG.get("smtp", {}).get("enabled"):
            send_email(ALERT_CONFIG['smtp'], "农机补贴异常预警", payload["text"], attachments=[anomalies_csv])

    print("Pipeline finished. Outputs at:", outdir)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farm Subsidy AI Data Pipeline")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input Excel path")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    args = parser.parse_args()
    main(args)