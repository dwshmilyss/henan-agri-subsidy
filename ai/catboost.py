import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import sys

# 虽然 XGBoost 是一个行业标准，但对于处理高基数（High Cardinality）分类特征的表格数据，还有两个模型可能更优秀，因为它们是专门为此类数据结构优化的：
#
#  1. LightGBM (LGBM)：以更快的训练速度和更低的内存消耗著称，尤其是在大数据集上。
#  2. CatBoost：这是目前处理包含大量分类变量的数据集时最强大的模型之一。
#
# 对于您这个包含大量文本特征（如 县、机具品目、生产厂家、购买机型）的农机数据集，我们选择CatBoost (Categorical Boosting)。
#
# 为什么选择 CatBoost？
# CatBoost 的核心优势在于它原生支持分类特征，无需进行传统的 One-Hot Encoding (OHE) 转换。这有两大好处：
#   1.避免维度爆炸： 您的数据集有许多厂家和机型，OHE 会产生数千个新列，增加模型复杂度和训练时间。CatBoost 直接处理这些文本列。
#   2.更好的特征表示： CatBoost 使用一种创新的“有序提升 (Ordered Boosting)”和“有序目标统计 (Ordered Target Statistics)”方法来编码分类特征，这能更准确地捕获类别与目标价格之间的关系，通常能带来更高的预测精度。
#
# 我已经将代码更新为使用 CatBoostRegressor，并移除了原有的 One-Hot Encoding 步骤，使数据预处理更加简洁高效。

# 关键变化：从 XGBoost 切换到 CatBoost
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("--------------------------------------------------------------------------------------")
    print("错误提示：CatBoost 库未安装。")
    print("请在您的环境中运行以下命令安装：")
    print("pip install catboost")
    sys.exit(1)

from sklearn.metrics import r2_score

# --- 全局常量和配置 ---
FILE_PATH = '../data/table_data_all.xlsx'
TARGET_COL = '单台销售价格(元)'
DATE_COL = '购机日期'
YEAR_COL = '购机年份'
ID_COL = '序号'

# 特征列
FEATURE_COLS = [
    '县', '机具品目', '生产厂家', '产品名称', '购买机型',
    '购买数量(台)', '单台中央补贴额(元)', '经销商', YEAR_COL
]

# CatBoost 独有：明确定义分类特征
NUMERICAL_COLS = [YEAR_COL, '购买数量(台)', '单台中央补贴额(元)']
CATEGORICAL_COLS = [col for col in FEATURE_COLS if col not in NUMERICAL_COLS]

MODEL_PATH = 'catboost_regressor_time_v7.pkl'  # CatBoost 模型文件名
CATEGORICAL_COL_PATH = 'catboost_categorical_columns_v7.pkl'
ABNORMAL_THRESHOLD_SIGMA = 2.0


# --- 1. 数据加载与预处理 ---

def load_and_preprocess_data():
    """加载数据，清洗，并确保分类列是字符串类型"""
    if not os.path.exists(FILE_PATH):
        print(f"错误：未找到文件 {FILE_PATH}。请确保文件路径正确。")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    try:
        # 尝试以 UTF-8 或 GBK 编码读取 CSV
        df = pd.read_csv(FILE_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding='GBK')
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

    # 提取年份特征
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df[YEAR_COL] = df[DATE_COL].dt.year.fillna(2024).astype(int)

    # 确保关键列都没有缺失值
    df.dropna(subset=[TARGET_COL] + FEATURE_COLS, inplace=True)

    # 准备特征和目标
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    # ❗ 关键：将所有分类特征转换为字符串类型，CatBoost 要求
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str)
        # 确保原始数据框中的分类列也转换为字符串
        df[col] = df[col].astype(str)

    # 保存分类列名
    joblib.dump(CATEGORICAL_COLS, CATEGORICAL_COL_PATH)

    # 返回特征（未编码）、目标、以及包含序号的原始 DataFrame
    return X, y, df


def train_and_save_model(X, y):
    """训练 CatBoost 模型并保存"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- CatBoost 模型配置 ---
    model = CatBoostRegressor(
        iterations=500,  # 迭代次数 (树的数量)
        learning_rate=0.05,  # 学习率
        depth=7,  # 树的最大深度
        loss_function='RMSE',  # 损失函数 (回归使用均方根误差)
        random_seed=42,
        verbose=0,  # 不打印训练过程
        # CatBoost 核心参数：直接传入分类特征列表
        cat_features=CATEGORICAL_COLS,
        thread_count=-1
    )

    print("开始训练 CatBoost 模型...")
    # 在 fit 阶段告知模型哪些是分类特征
    model.fit(X_train, y_train, cat_features=CATEGORICAL_COLS)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"CatBoost 模型训练完成，R²得分: {r2:.4f}")

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
    # CatBoost 不需要 OHE 列，只需要分类列名
    _ = joblib.load(CATEGORICAL_COL_PATH)
    return model


# --- 2. 价格预测（含年份预测功能） ---

def predict_new_price(new_data, model):
    """
    对单个新样本进行预测，CatBoost可以直接使用原始特征。
    """
    if YEAR_COL not in new_data:
        raise ValueError(f"预测数据必须包含年份特征: '{YEAR_COL}'")

    new_df = pd.DataFrame([new_data])

    # 确保分类特征是字符串类型
    for col in CATEGORICAL_COLS:
        new_df[col] = new_df[col].astype(str)

    # 确保数值列是数值类型
    new_df['购买数量(台)'] = pd.to_numeric(new_df['购买数量(台)'], errors='coerce').fillna(1)
    new_df['单台中央补贴额(元)'] = pd.to_numeric(new_df['单台中央补贴额(元)'], errors='coerce').fillna(0)
    new_df[YEAR_COL] = pd.to_numeric(new_df[YEAR_COL], errors='coerce').fillna(new_data[YEAR_COL])

    # 确保特征顺序与训练时一致
    X_predict = new_df[FEATURE_COLS]

    predicted_price = model.predict(X_predict)[0]
    return predicted_price


# --- 3. 特征重要性分析（功能 2: 关键影响因素分析） ---

def analyze_feature_importance(model):
    """分析并打印 CatBoost 特征重要性，直接使用原始特征名"""
    importances = model.get_feature_importance()
    feature_series = pd.Series(importances, index=FEATURE_COLS)

    # 只显示 Top 15 特征
    top_features = feature_series.sort_values(ascending=False).head(15)

    print("\n--- 关键影响因素分析 (CatBoost 特征重要性 Top 15) ---")

    # 由于 CatBoost 使用原始特征名，无需进行反向 OHE 解析
    return top_features


# --- 4. 价格异常预警（功能 3: 销售异常检测） ---

def detect_anomalies(df_original, model, mae, mae_std):
    """
    检测数据集中实际价格与预测价格偏差大于阈值的交易。
    CatBoost可以直接使用原始特征的DataFrame。
    """
    if df_original.empty:
        print("原始数据集为空，无法进行异常检测。")
        return pd.DataFrame()

    X_all = df_original[FEATURE_COLS].copy()

    # 确保分类特征是字符串类型
    for col in CATEGORICAL_COLS:
        X_all[col] = X_all[col].astype(str)

    # 确保数值列是数值类型
    X_all['购买数量(台)'] = pd.to_numeric(X_all['购买数量(台)'], errors='coerce').fillna(1)
    X_all['单台中央补贴额(元)'] = pd.to_numeric(X_all['单台中央补贴额(元)'], errors='coerce').fillna(0)
    X_all[YEAR_COL] = pd.to_numeric(X_all[YEAR_COL], errors='coerce').fillna(2024)

    # 预测所有样本
    df_original['Predicted_Price'] = model.predict(X_all)
    df_original['Actual_Price'] = df_original[TARGET_COL]

    # 计算误差
    df_original['Prediction_Error'] = np.abs(df_original['Actual_Price'] - df_original['Predicted_Price'])
    df_original['Price_Difference'] = df_original['Actual_Price'] - df_original['Predicted_Price']

    # 设定异常阈值：超过平均误差 + 2倍标准差
    threshold = mae + ABNORMAL_THRESHOLD_SIGMA * mae_std
    anomalies = df_original[df_original['Prediction_Error'] > threshold].sort_values(by='Prediction_Error',
                                                                                     ascending=False)

    print("\n--- 价格异常预警 (预测误差超过阈值) ---")
    print(f"CatBoost 模型平均误差 (MAE): {mae:.2f} 元")
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

    # 1. 初始设置：如果模型和分类列文件不存在，则进行训练
    try:
        model = load_resources()
        print("已加载现有 CatBoost 模型。")
        X_all, y_all, df_original = load_and_preprocess_data()

        # 重新计算 MAE 和 STD
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        model_temp = joblib.load(MODEL_PATH)
        y_pred_temp = model_temp.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred_temp))
        mae_std = np.std(np.abs(y_test - y_pred_temp))

    except (FileNotFoundError, EOFError, IndexError) as e:
        print(f"CatBoost 模型或分类列文件不存在或加载失败 ({e})，正在进行初始训练...")
        X_all, y_all, df_original = load_and_preprocess_data()

        if not X_all.empty:
            mae, mae_std = train_and_save_model(X_all, y_all)
            model = load_resources()
        else:
            print("数据加载失败或数据为空，无法执行训练和预测。")
            sys.exit(1)

    # =========================================================================
    # --- 功能演示 ---
    # =========================================================================

    # 1. 价格预测（新产品定价/未来定价）
    print("\n======================== 1. 价格预测演示 (含 2026 年预测) ========================")

    # 预测样本规格（与上一个模型相同）
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
    predicted_price_2026 = predict_new_price(product_spec_2026, model)
    print(f"\n基于 CatBoost 模型和当前趋势，预测轮式拖拉机 (M704-2H) 2026 年销售价格: {predicted_price_2026:.2f} 元")

    # 2. 特征重要性分析（关键影响因素）
    print("\n======================== 2. 特征重要性分析 ========================")
    importance_df = analyze_feature_importance(model)
    print(importance_df.to_markdown())

    # 3. 价格异常预警（销售异常检测）
    print("\n======================== 3. 价格异常预警 ========================")
    anomalies_report = detect_anomalies(df_original.copy(), model, mae, mae_std)
    if not anomalies_report.empty:
        print("\n--- 发现价格偏差 Top 10 交易，需重点审查 ---")
        print(anomalies_report.to_markdown(index=False))

    # 4. 数据驱动的决策支持（Top N 推荐）
    print("\n======================== 4. 数据驱动决策演示 ========================")
    top_items_report = get_top_performing_items(df_original, item_type='旋耕机', top_n=3)
    if not top_items_report.empty:
        print(top_items_report.to_markdown(index=False))