import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import joblib  # 用于保存和加载模型及编码器

# --- 全局常量和配置 ---
FILE_PATH = 'data/table_data_all.xlsx'
TARGET_COL = '单台销售价格(元)'
FEATURE_COLS = [
    '县', '机具品目', '生产厂家', '产品名称', '购买机型',
    '购买数量(台)', '单台中央补贴额(元)', '经销商'
]
MODEL_PATH = 'random_forest_model.pkl'
ENCODER_PATH = 'one_hot_encoder_columns.pkl'
ABNORMAL_THRESHOLD_SIGMA = 2.0  # 异常检测的阈值：超过平均绝对误差的2倍标准差


# --- 1. 数据加载、预处理与模型训练 (确保模型和编码器可用) ---

def load_and_preprocess_data():
    """加载数据，清洗，并进行 One-Hot Encoding"""
    try:
        # 尝试以GBK编码读取
        df = pd.read_excel(FILE_PATH)
    except UnicodeDecodeError:
        # 如果GBK失败，尝试UTF-8
        df = pd.read_excel(FILE_PATH, encoding='utf-8')

    df.columns = [col.strip() for col in df.columns]

    # 清洗目标列
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df.dropna(subset=[TARGET_COL] + FEATURE_COLS, inplace=True)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    # One-Hot Encoding (OHE) - 关键步骤：保存编码后的列名用于新数据预测
    X_encoded = pd.get_dummies(X, columns=['县', '机具品目', '生产厂家', '产品名称', '购买机型', '经销商'],
                               dummy_na=False, drop_first=True)

    # 保存编码后的列名
    joblib.dump(X_encoded.columns.tolist(), ENCODER_PATH)

    # 填充数值特征的NaN值
    X_encoded['购买数量(台)'] = X_encoded['购买数量(台)'].fillna(1)
    X_encoded['单台中央补贴额(元)'] = X_encoded['单台中央补贴额(元)'].fillna(0)

    return X_encoded, y, df


def train_and_save_model(X, y):
    """训练模型并保存"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
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


# --- 2. 价格预测（功能 1: 预测与定价策略） ---

def predict_new_price(new_data, model, encoder_cols):
    """
    对单个新样本进行预测。
    :param new_data: 包含所有特征值的字典或Series。
    :param model: 训练好的模型。
    :param encoder_cols: 训练时使用的 One-Hot 编码列名列表。
    :return: 预测价格。
    """
    # 转换为DataFrame
    new_df = pd.DataFrame([new_data])

    # One-Hot Encoding
    new_df_encoded = pd.get_dummies(new_df, columns=['县', '机具品目', '生产厂家', '产品名称', '购买机型', '经销商'],
                                    dummy_na=False, drop_first=True)

    # 重新对齐列：这是关键步骤，确保新数据和训练数据的特征维度一致
    X_predict = pd.DataFrame(0, index=new_df_encoded.index, columns=encoder_cols)
    for col in new_df_encoded.columns:
        if col in X_predict.columns:
            X_predict[col] = new_df_encoded[col]

    # 确保数值列是数值类型
    X_predict['购买数量(台)'] = pd.to_numeric(X_predict['购买数量(台)'], errors='coerce').fillna(1)
    X_predict['单台中央补贴额(元)'] = pd.to_numeric(X_predict['单台中央补贴额(元)'], errors='coerce').fillna(0)

    # 预测
    predicted_price = model.predict(X_predict)[0]
    return predicted_price


# --- 3. 特征重要性分析（功能 2: 关键影响因素分析） ---

def analyze_feature_importance(model, encoder_cols):
    """分析并打印特征重要性"""
    importances = model.feature_importances_
    feature_series = pd.Series(importances, index=encoder_cols)
    top_features = feature_series[feature_series > 0].sort_values(ascending=False).head(15)

    print("\n--- 关键影响因素分析 (特征重要性 Top 15) ---")
    # 为了更清晰地展示原始特征名称，进行初步解析
    renamed_features = {}
    for name, score in top_features.items():
        if name in FEATURE_COLS:
            # 数值特征直接显示
            renamed_features[name] = score
        else:
            # OHE特征解析
            parts = name.split('_', 1)
            original_feature = parts[0]
            if original_feature in FEATURE_COLS:
                renamed_features[f"{original_feature} ({parts[1]})"] = score
            else:
                renamed_features[name] = score  # 无法解析，保持原样

    return pd.Series(renamed_features).sort_values(ascending=False)


# --- 4. 价格异常预警（功能 3: 销售异常检测） ---

def detect_anomalies(df_original, model, encoder_cols, mae, mae_std):
    """
    检测数据集中实际价格与预测价格偏差大于阈值的交易。
    :param df_original: 原始数据集。
    :param mae, mae_std: 训练集上的平均绝对误差和标准差。
    """
    # 必须对整个数据集进行预测
    X_all = df_original[FEATURE_COLS].copy()

    X_encoded = pd.get_dummies(X_all, columns=['县', '机具品目', '生产厂家', '产品名称', '购买机型', '经销商'],
                               dummy_na=False, drop_first=True)

    X_final = pd.DataFrame(0, index=X_encoded.index, columns=encoder_cols)
    for col in X_encoded.columns:
        if col in X_final.columns:
            X_final[col] = X_encoded[col]

    X_final['购买数量(台)'] = pd.to_numeric(X_final['购买数量(台)'], errors='coerce').fillna(1)
    X_final['单台中央补贴额(元)'] = pd.to_numeric(X_final['单台中央补贴额(元)'], errors='coerce').fillna(0)

    # 预测所有样本
    df_original['Predicted_Price'] = model.predict(X_final)
    df_original['Actual_Price'] = df_original[TARGET_COL]
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

    return anomalies[
        ['县', '机具品目', '生产厂家', '经销商', 'Actual_Price', 'Predicted_Price', 'Price_Difference']].head(10)


# --- 5. 数据驱动的决策支持（功能 4: 推荐/优化） ---

def get_top_performing_items(df_original, item_type='旋耕机', top_n=5):
    """
    分析某一品目下，哪个厂家/型号带来了最高的平均销售价格。
    :param df_original: 原始数据集。
    :param item_type: 要分析的机具品目。
    :param top_n: 返回最高的 N 个组合。
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

    except FileNotFoundError:
        print("模型或编码器文件不存在，正在进行初始训练...")
        X_all, y_all, df_original = load_and_preprocess_data()
        mae, mae_std = train_and_save_model(X_all, y_all)
        model, encoder_cols = load_resources()

    # =========================================================================
    # --- 功能演示 ---
    # =========================================================================

    # 1. 价格预测（新产品定价）
    print("\n======================== 1. 价格预测演示 ========================")
    new_product_spec = {
        '县': '淮阳县',
        '机具品目': '轮式拖拉机',
        '生产厂家': '潍柴雷沃智慧农业科技股份有限公司',
        '产品名称': '轮式拖拉机',
        '购买机型': '现:M704-2H(G4)(原:M704-2H)',
        '购买数量(台)': 1,
        '单台中央补贴额(元)': 8700.00,
        '经销商': '周口群农农业机械销售有限公司'
    }
    predicted_price = predict_new_price(new_product_spec, model, encoder_cols)
    print(f"\n基于输入特征，预测销售价格: {predicted_price:.2f} 元")

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