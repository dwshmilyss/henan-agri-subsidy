import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- 1. 数据加载与初始查看 ---
file_path = '../../data/table_data_all.xlsx'
try:
    # 尝试以GBK编码读取，因为中文CSV文件常使用GBK或GB2312
    df = pd.read_excel(file_path)
    print("数据加载成功，前5行数据:")
    print(df.head())
    print("\n原始数据信息:")
    df.info()
except UnicodeDecodeError:
    # 如果GBK失败，尝试UTF-8
    df = pd.read_excel(file_path, encoding='utf-8')
    print("数据加载成功 (UTF-8)，前5行数据:")
    print(df.head())
    print("\n原始数据信息:")
    df.info()

# --- 2. 数据清洗与特征工程 ---

# 重命名目标列，并处理可能的空格
df.columns = [col.strip() for col in df.columns]
target_col = '单台销售价格(元)'

# 确保目标列是数值类型，并处理缺失值
# 价格列通常是干净的，但我们进行一次强制转换
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df.dropna(subset=[target_col], inplace=True)
print(f"\n清洗后样本数: {len(df)}")

# 确定特征列 (排除明显无关或重复的列)
# 排除: 序号, 所在乡(镇) (信息重复), 购机者姓名 (ID), 购机日期, 状态, 是否超录申请, 总补贴额(元) (与单台补贴额相关), 出厂编号
feature_cols = [
    '县', '机具品目', '生产厂家', '产品名称', '购买机型',
    '购买数量(台)', '单台中央补贴额(元)', '经销商'
]

# 提取特征和目标变量
X = df[feature_cols].copy()
y = df[target_col]

# 将日期转换为数值特征（由于数据片段中购机日期跨度很小，此处暂不处理，如果需要，应使用 df['购机日期'].dt.dayofyear 或年份差）
# 购买数量(台) 和 单台中央补贴额(元) 已经是数值型，无需处理

# 处理中文分类变量 (One-Hot Encoding)
# 注意：对于有大量唯一值的列（如 '生产厂家', '经销商'），One-Hot Encoding 会导致特征数量爆炸。
# 在实际应用中，可以考虑使用 Target Encoding 或只保留最常见的N个类别。
# 此处我们采用最常见的 One-Hot Encoding，并设置 handle_unknown='ignore' 以处理训练集外的新类别
X = pd.get_dummies(X, columns=['县', '机具品目', '生产厂家', '产品名称', '购买机型', '经销商'],
                   dummy_na=False, drop_first=True)

print(f"\nOne-Hot Encoding 后，特征总数: {X.shape[1]}")

# 检查并填充数值特征中的NaN值（虽然不太可能，但保险起见）
X['购买数量(台)'] = X['购买数量(台)'].fillna(1)
X['单台中央补贴额(元)'] = X['单台中央补贴额(元)'].fillna(0)


# --- 3. 模型训练 ---

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林回归模型
# n_estimators: 树的数量，越多越好但计算量大
# random_state: 保证结果可复现
# max_depth: 限制树的深度，防止过拟合
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

print("\n开始训练随机森林回归模型...")
model.fit(X_train, y_train)
print("模型训练完成。")

# --- 4. 模型评估与预测 ---

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- 模型评估结果 ---")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"决定系数 (R²): {r2:.4f}")

# 解释：R² 越接近 1，表示模型解释的方差越多，拟合越好。
# RMSE 表示预测价格与实际价格的平均差异，单位是元。

# --- 5. 特征重要性分析 ---

print("\n--- 特征重要性排序 ---")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
# 筛选出重要性大于0的特征，并按降序排列
top_features = feature_importances[feature_importances > 0].sort_values(ascending=False).head(15)

print(top_features)

# --- 6. 示例预测 ---
print("\n--- 示例预测 ---")
# 预测第一个测试样本
first_test_sample = X_test.iloc[[0]]
actual_price = y_test.iloc[0]
predicted_price = model.predict(first_test_sample)[0]

print(first_test_sample)
print(f"实际销售价格: {actual_price:.2f} 元")
print(f"模型预测价格: {predicted_price:.2f} 元")
print(f"预测误差: {predicted_price - actual_price:.2f} 元")

# 总结：
# 如果 R² 值很高 (接近 1)，表明模型对价格的解释力很强。
# 特征重要性可以告诉我们哪些因素（如机具品目、生产厂家、补贴额）对价格影响最大。