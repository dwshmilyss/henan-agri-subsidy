import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import joblib  # 用于保存和加载模型及编码器
import os

# --- 全局常量和配置 ---
FILE_PATH = 'data/table_data_all.xlsx'
TARGET_COL = '单台销售价格(元)'
DATE_COL = '购机日期'
YEAR_COL = '购机年份'  # 新增：年份特征
ID_COL = '序号'

# 更新特征列，加入购机年份
FEATURE_COLS = [
    '县', '机具品目', '生产厂家', '产品名称', '购买机型',
    '购买数量(台)', '单台中央补贴额(元)', '经销商', YEAR_COL  # 关键：新增 YEAR_COL
]

MODEL_PATH = 'random_forest_model_time.pkl'  # 更改模型文件名，避免与旧模型混淆
ENCODER_PATH = 'one_hot_encoder_columns_time.pkl'
ABNORMAL_THRESHOLD_SIGMA = 2.0


# --- 1. 数据加载、预处理与模型训练 (确保模型和编码器可用) ---

def load_and_preprocess_data():
    """加载数据，清洗，并进行 One-Hot Encoding"""
    if not os.path.exists(FILE_PATH):
        print(f"错误：未找到文件 {FILE_PATH}。请确保文件路径正确。")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    try:
        df = pd.read_excel(FILE_PATH)
    except Exception:
        # 如果读取Excel失败，尝试读取CSV
        try:
            df = pd.read_csv(FILE_PATH.replace('.xlsx', '.csv'), encoding='GBK')
        except Exception as e:
            print(f"读取数据失败: {e}")
            return pd.DataFrame(), pd.Series(), pd.DataFrame()

    df.columns = [col.strip() for col in df.columns]

    # 清洗和准备序号列
    if ID_COL not in df.columns:
        df[ID_COL] = df.index + 1
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors='coerce').fillna(0).astype(int)

    # 清洗目标列
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')

    # --- 关键修改 1: 提取年份特征 ---
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df[YEAR_COL] = df[DATE_COL].dt.year.fillna(2024).astype(int)  # 缺失日期填充为2024年

    # 确保关键列都没有缺失值
    df.dropna(subset=[TARGET_COL] + FEATURE_COLS, inplace=True)

    # 准备特征和目标
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    # One-Hot Encoding (OHE) - 排除 YEAR_COL，因为它已经是数值
    categorical_cols = [col for col in FEATURE_COLS if col not in [YEAR_COL, '购买数量(台)', '单台中央补贴额(元)']]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, dummy_na=False, drop_first=True)

    # 确保数值列是数值类型，并填充 NaN
    X_encoded['购买数量(台)'] = pd.to_numeric(X_encoded['购买数量(台)'], errors='coerce').fillna(1)
    X_encoded['单台中央补贴额(元)'] = pd.to_numeric(X_encoded['单台中央补贴额(元)'], errors='coerce').fillna(0)
    # YEAR_COL 已经是整数，无需额外处理

    # 保存编码后的列名
    joblib.dump(X_encoded.columns.tolist(), ENCODER_PATH)

    # 返回编码后的特征、目标、以及包含序号的原始 DataFrame
    return X_encoded, y, df


def train_and_save_model(X, y):
    """训练模型并保存"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 增加 n_estimators 以可能更好地捕捉时间趋势
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"模型训练完成，R²得分: {r2:.4f}")

    # 保存模型
    joblib.dump(model, MODEL_PATH)

    # 计算平均绝对误差（用于异常检测）
    mae = np.mean(np.abs(y_test - y_pred))
    mae_std = np.std(np.abs(y_test - y_pred))
    print(f"平均绝对误差 (MAE): {mae:.2f} 元")
    return mae, mae_std


def load_resources():
    """加载预处理资源和模型"""
    model = joblib.load(MODEL_PATH)
    encoder_cols = joblib.load(ENCODER_PATH)
    return model, encoder_cols


# --- 2. 价格预测（含年份预测功能） ---

def predict_new_price(new_data, model, encoder_cols):
    """
    对单个新样本进行预测，包含年份。
    """
    if YEAR_COL not in new_data:
        raise ValueError(f"预测数据必须包含年份特征: '{YEAR_COL}'")

    new_df = pd.DataFrame([new_data])

    # OHE 编码的列（排除数值列和年份）
    categorical_cols = [col for col in FEATURE_COLS if col not in [YEAR_COL, '购买数量(台)', '单台中央补贴额(元)']]
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, dummy_na=False, drop_first=True)

    # 重新对齐列
    X_predict = pd.DataFrame(0, index=new_df_encoded.index, columns=encoder_cols)
    for col in new_df_encoded.columns:
        if col in X_predict.columns:
            X_predict[col] = new_df_encoded[col]

    # 确保数值列和年份是数值类型
    X_predict['购买数量(台)'] = pd.to_numeric(X_predict['购买数量(台)'], errors='coerce').fillna(1)
    X_predict['单台中央补贴额(元)'] = pd.to_numeric(X_predict['单台中央补贴额(元)'], errors='coerce').fillna(0)
    X_predict[YEAR_COL] = pd.to_numeric(X_predict[YEAR_COL], errors='coerce').fillna(new_data[YEAR_COL])

    predicted_price = model.predict(X_predict)[0]
    return predicted_price


# --- 3. 特征重要性分析（功能 2: 关键影响因素分析） ---

def analyze_feature_importance(model, encoder_cols):
    """分析并打印特征重要性，包含年份"""
    importances = model.feature_importances_
    feature_series = pd.Series(importances, index=encoder_cols)
    top_features = feature_series[feature_series > 0].sort_values(ascending=False).head(15)

    print("\n--- 关键影响因素分析 (特征重要性 Top 15) ---")

    renamed_features = {}
    for name, score in top_features.items():
        if name in FEATURE_COLS:
            # 数值特征和 YEAR_COL 直接显示
            renamed_features[name] = score
        else:
            # OHE特征解析
            parts = name.split('_', 1)
            original_feature = parts[0]
            if original_feature in FEATURE_COLS:
                display_value = parts[1][:25] + '...' if len(parts[1]) > 25 else parts[1]
                renamed_features[f"{original_feature} ({display_value})"] = score
            else:
                renamed_features[name] = score

    return pd.Series(renamed_features).sort_values(ascending=False)


# --- 4. 价格异常预警（功能 3: 销售异常检测） ---

def detect_anomalies(df_original, model, encoder_cols, mae, mae_std):
    """
    检测数据集中实际价格与预测价格偏差大于阈值的交易。
    """
    if df_original.empty:
        print("原始数据集为空，无法进行异常检测。")
        return pd.DataFrame()

    X_all = df_original[FEATURE_COLS].copy()

    # OHE 编码的列（排除数值列和年份）
    categorical_cols = [col for col in FEATURE_COLS if col not in [YEAR_COL, '购买数量(台)', '单台中央补贴额(元)']]
    X_encoded = pd.get_dummies(X_all, columns=categorical_cols, dummy_na=False, drop_first=True)

    X_final = pd.DataFrame(0, index=X_encoded.index, columns=encoder_cols)
    for col in X_encoded.columns:
        if col in X_final.columns:
            X_final[col] = X_encoded[col]

    # 确保数值列和年份是数值类型
    X_final['购买数量(台)'] = pd.to_numeric(X_final['购买数量(台)'], errors='coerce').fillna(1)
    X_final['单台中央补贴额(元)'] = pd.to_numeric(X_final['单台中央补贴额(元)'], errors='coerce').fillna(0)
    X_final[YEAR_COL] = pd.to_numeric(X_final[YEAR_COL], errors='coerce').fillna(2024)

    # 预测所有样本
    df_original['Predicted_Price'] = model.predict(X_final)
    df_original['Actual_Price'] = df_original[TARGET_COL]

    # 计算误差
    df_original['Prediction_Error'] = np.abs(df_original['Actual_Price'] - df_original['Predicted_Price'])
    df_original['Price_Difference'] = df_original['Actual_Price'] - df_original['Predicted_Price']

    # 设定异常阈值：超过平均误差 + 2倍标准差
    threshold = mae + ABNORMAL_THRESHOLD_SIGMA * mae_std
    anomalies = df_original[df_original['Prediction_Error'] > threshold].sort_values(by='Prediction_Error',
                                                                                     ascending=False)

    print("\n--- 价格异常预警 (预测误差超过阈值) ---")
    print(f"模型平均误差 (MAE): {mae:.2f} 元")
    print(f"异常检测阈值 (MAE + {ABNORMAL_THRESHOLD_SIGMA}*STD): {threshold:.2f} 元")

    if anomalies.empty:
        print("未发现显著的价格异常交易。")
        return pd.DataFrame()

    report_cols = [ID_COL, YEAR_COL, '县', '机具品目', '生产厂家', '经销商', 'Actual_Price', 'Predicted_Price',
                   'Price_Difference']

    if ID_COL not in anomalies.columns:
        anomalies[ID_COL] = anomalies.index + 1

    return anomalies[report_cols].head(10)


# --- 5. 数据驱动的决策支持（功能 4: 推荐/优化） ---

def get_top_performing_items(df_original, item_type='旋耕机', top_n=5):
    """
    分析某一品目下，哪个厂家/型号带来了最高的平均销售价格。
    """
    print(f"\n--- 数据驱动决策: 【{item_type}】品目 Top {top_n} 高价组合 ---")

    df_filtered = df_original[df_original['机具品目'] == item_type].copy()

    if df_filtered.empty:
        print(f"数据集中没有找到机具品目: {item_type}")
        return pd.DataFrame()

    # 按厂家和机型分组，计算平均价格和销售数量
    performance = df_filtered.groupby(['生产厂家', '购买机型']).agg(
        Average_Price=(TARGET_COL, 'mean'),
        Sales_Count=('购买数量(台)', 'sum'),
        Max_Central_Subsidy=('单台中央补贴额(元)', 'max')
    ).sort_values(by='Average_Price', ascending=False)

    return performance.head(top_n).reset_index()


# --- 主执行流程 ---
if __name__ == '__main__':
    # 1. 初始设置：如果模型和编码器不存在，则进行训练
    try:
        model, encoder_cols = load_resources()
        print("已加载现有模型和编码器。")
        X_all, y_all, df_original = load_and_preprocess_data()

        # 重新计算 MAE 和 STD（需要重新训练或从元数据中读取，此处简化为快速重训）
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        model_temp = joblib.load(MODEL_PATH)
        y_pred_temp = model_temp.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred_temp))
        mae_std = np.std(np.abs(y_test - y_pred_temp))

    except (FileNotFoundError, EOFError, IndexError) as e:
        print(f"模型或编码器文件不存在或加载失败 ({e})，正在进行初始训练...")
        X_all, y_all, df_original = load_and_preprocess_data()

        if not X_all.empty:
            mae, mae_std = train_and_save_model(X_all, y_all)
            model, encoder_cols = load_resources()
        else:
            print("数据加载失败或数据为空，无法执行训练和预测。")
            exit()

    # =========================================================================
    # --- 功能演示 ---
    # =========================================================================

    # 1. 价格预测（新产品定价/未来定价）
    print("\n======================== 1. 价格预测演示 (含 2026 年预测) ========================")

    # 预测样本规格（假设与原代码中的 M704-2H 轮式拖拉机规格一致）
    product_spec_2026 = {
        '县': '淮阳县',
        '机具品目': '轮式拖拉机',
        '生产厂家': '潍柴雷沃智慧农业科技股份有限公司',
        '产品名称': '轮式拖拉机',
        '购买机型': '现:M704-2H(G4)(原:M704-2H)',
        '购买数量(台)': 1,
        '单台中央补贴额(元)': 8700.00,
        '经销商': '周口群农农业机械销售有限公司',
        YEAR_COL: 2026  # 关键：将年份设置为 2026
    }

    # 进行 2026 年的预测
    predicted_price_2026 = predict_new_price(product_spec_2026, model, encoder_cols)
    print(f"\n基于当前趋势，预测轮式拖拉机 (M704-2H) 2026 年销售价格: {predicted_price_2026:.2f} 元")

    # 2. 特征重要性分析（关键影响因素）
    print("\n======================== 2. 特征重要性分析 ========================")
    importance_df = analyze_feature_importance(model, encoder_cols)
    print(importance_df.to_markdown())

    # 3. 价格异常预警（销售异常检测）
    print("\n======================== 3. 价格异常预警 ========================")
    anomalies_report = detect_anomalies(df_original.copy(), model, encoder_cols, mae, mae_std)
    if not anomalies_report.empty:
        print("\n--- 发现价格偏差 Top 10 交易，需重点审查 ---")
        print(anomalies_report.to_markdown(index=False))

    # 4. 数据驱动的决策支持（Top N 推荐）
    print("\n======================== 4. 数据驱动决策演示 ========================")
    top_items_report = get_top_performing_items(df_original, item_type='旋耕机', top_n=3)
    if not top_items_report.empty:
        print(top_items_report.to_markdown(index=False))