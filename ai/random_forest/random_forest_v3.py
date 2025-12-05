import pandas as pd
import plotly.express as px
from plotly.offline import plot
import numpy as np
import os

# --- 配置 ---
FILE_NAME = '../../data/table_data_all.xlsx'
OUTPUT_HTML_NAME = 'ai_farm_analysis_dashboard.html'

# 关键列名 (使用英文别名方便代码编写)
COLS_MAPPING = {
    '机具品目': 'Item_Category',
    '生产厂家': 'Manufacturer',
    '购买机型': 'Model',
    '购机日期': 'Purchase_Date',
    '购买数量(台)': 'Count',
    '单台销售价格(元)': 'Price',
    '单台中央补贴额(元)': 'Subsidy',
}


# --- 1. 数据加载与清洗 ---
def load_and_clean_data(file_path):
    """加载数据，处理列名和数值类型"""
    if not os.path.exists(file_path):
        print(f"错误：未找到文件 {file_path}")
        return None

    try:
        # 尝试以 UTF-8 或 GBK 编码读取 CSV
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='GBK')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    # 清理并映射列名
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=COLS_MAPPING, inplace=True)

    # 强制转换关键数值列
    for col in ['Price', 'Subsidy', 'Count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 过滤掉价格或补贴为零的无效交易
    df = df[(df['Price'] > 100) & (df['Count'] >= 1)].copy()

    # 计算总收入
    df['Revenue'] = df['Price'] * df['Count']

    return df


# --- 2. 需求 1: 市场竞争格局分析 (Market Positioning) ---
def analyze_market_position(df):
    """聚合数据以分析市场份额和价格定位"""

    # 聚合：按品目和厂家计算总销量、平均价格、总收入
    market_data = df.groupby(['Item_Category', 'Manufacturer']).agg(
        Total_Count=('Count', 'sum'),
        Avg_Price=('Price', 'mean'),
        Total_Revenue=('Revenue', 'sum')
    ).reset_index()

    # 过滤掉销量极低的噪音数据
    market_data = market_data[market_data['Total_Count'] >= 5]

    fig = px.scatter(
        market_data,
        x='Avg_Price',
        y='Total_Count',
        size='Total_Revenue',  # 气泡大小代表总收入，体现重要性
        color='Item_Category',
        hover_name='Manufacturer',
        log_y=True,  # 销量使用对数尺度，更好区分高低销量厂家
        title='市场竞争格局：平均价格 vs. 总销量 (气泡大小: 总收入)',
        labels={
            'Avg_Price': '平均销售价格 (元)',
            'Total_Count': '总销量 (台, 对数尺度)',
            'Total_Revenue': '总收入 (元)',
            'Item_Category': '机具品目',
            'Manufacturer': '生产厂家'
        }
    )
    fig.update_layout(height=700)
    return fig


# --- 3. 需求 2: 补贴-价格关联性分析 (Subsidy Impact) ---
def analyze_subsidy_impact(df):
    """分析补贴额与价格之间的关系"""

    # 选择销量最高的 N 个品目进行分析
    top_items = df['Item_Category'].value_counts().nlargest(6).index
    df_filtered = df[df['Item_Category'].isin(top_items)].copy()

    fig = px.scatter(
        df_filtered.sample(n=min(5000, len(df_filtered)), random_state=42),  # 采样以提高性能
        x='Subsidy',
        y='Price',
        color='Item_Category',
        facet_col='Item_Category',  # 按品目分面显示
        facet_col_wrap=3,
        trendline='ols',  # 添加线性回归趋势线
        title='补贴额与销售价格关系 (按主要品目分面)',
        labels={
            'Subsidy': '单台中央补贴额 (元)',
            'Price': '单台销售价格 (元)',
            'Item_Category': '机具品目'
        },
        height=800
    )
    fig.update_layout(showlegend=False)
    return fig


# --- 4. 需求 3: 价格异常/离群点检测 (Anomaly Detection) ---
def detect_price_anomalies(df):
    """使用 Z-Score/IQR 简化模型检测价格离群点"""

    # 核心逻辑：计算每个 Model 的平均价格和标准差
    model_stats = df.groupby('Model')['Price'].agg(['mean', 'std']).reset_index()
    model_stats.rename(columns={'mean': 'Avg_Model_Price', 'std': 'Std_Model_Price'}, inplace=True)

    df_merged = df.merge(model_stats, on='Model', how='left')

    # 设定异常阈值：价格偏离平均价格 3 个标准差（且标准差需大于0）
    df_merged['Price_Z_Score'] = (df_merged['Price'] - df_merged['Avg_Model_Price']) / df_merged[
        'Std_Model_Price'].replace(0, np.nan)

    # 筛选出 Z-Score 绝对值大于 3 的交易
    anomalies = df_merged[df_merged['Price_Z_Score'].abs() > 3].sort_values(by='Price_Z_Score', ascending=False)

    # 报告 Top 10 异常交易
    anomaly_report = anomalies[[
        'Item_Category', 'Manufacturer', 'Model', 'Price', 'Avg_Model_Price', 'Subsidy', 'Purchase_Date'
    ]].head(10).to_html(index=False, classes='table-auto w-full text-left whitespace-nowrap',
                        float_format=lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else str(x))

    return anomaly_report, len(anomalies)


def generate_dashboard_html(figs, anomaly_html, anomaly_count):
    """将所有图表和表格嵌入一个 HTML 文件"""

    # 将 Plotly 图表转换为 HTML 片段
    plot_html = ""
    for title, fig in figs.items():
        # 使用 Plotly.js 离线模式，将图表数据嵌入到 HTML 中
        plot_div = plot(fig, output_type='div', include_plotlyjs=False)
        plot_html += f'<div class="p-6 bg-white rounded-xl shadow-lg mb-8">{plot_div}</div>'

    # 使用 Tailwind CSS 结构化页面
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>农机销售 AI 驱动分析看板</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            body {{ font-family: 'Inter', sans-serif; background-color: #f4f7f9; }}
            .plotly-graph-div {{ width: 100% !important; height: auto !important; min-height: 500px; }}
            .table-auto {{ border-collapse: collapse; }}
            .table-auto th, .table-auto td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; }}
            .table-auto th {{ background-color: #4f46e5; color: white; }}
        </style>
    </head>
    <body>
        <div class="container mx-auto p-4 md:p-10">
            <header class="text-center py-8 bg-white rounded-xl shadow-xl mb-10">
                <h1 class="text-4xl font-extrabold text-indigo-700">农机销售 AI 驱动分析看板</h1>
                <p class="text-xl text-gray-600 mt-2">基于 {FILE_NAME} 的交互式数据洞察</p>
            </header>

            <!-- 需求 3: 价格异常检测 -->
            <section class="mb-10 p-6 bg-red-50 rounded-xl shadow-xl border-l-4 border-red-500">
                <h2 class="text-2xl font-bold text-red-700 mb-4">🚨 需求 3: 价格异常/离群点检测 ({anomaly_count} 笔可疑交易)</h2>
                <p class="text-gray-700 mb-4">以下是价格偏离同型号平均价格超过 3 个标准差的 Top 10 交易，建议重点审查其销售价格与补贴额的合理性。</p>
                <div class="overflow-x-auto">
                    {anomaly_html}
                </div>
            </section>

            <!-- 需求 1 & 2: 交互式图表 -->
            <section>
                <h2 class="text-3xl font-bold text-indigo-700 mb-6">📈 需求 1 & 2: 市场洞察与补贴政策分析</h2>
                {plot_html}
            </section>

            <footer class="text-center py-6 text-gray-500 text-sm">
                数据分析由 Python/Pandas/Plotly 生成，前端由 Tailwind CSS 渲染。
            </footer>
        </div>
    </body>
    </html>
    """

    with open(OUTPUT_HTML_NAME, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"\n--- 仪表板生成成功 ---\n交互式结果已保存到文件: {OUTPUT_HTML_NAME}")


# --- 主执行逻辑 ---
if __name__ == '__main__':
    # 1. 加载数据
    df_all = load_and_clean_data(FILE_NAME)

    if df_all is None:
        exit()

    # 2. 执行分析并生成图表
    figures = {}

    # 需求 1
    try:
        figures['market_position'] = analyze_market_position(df_all)
    except Exception as e:
        print(f"生成竞争格局图表失败: {e}")

    # 需求 2
    try:
        figures['subsidy_impact'] = analyze_subsidy_impact(df_all)
    except Exception as e:
        print(f"生成补贴关联性图表失败: {e}")

    # 需求 3
    try:
        anomaly_html, anomaly_count = detect_price_anomalies(df_all)
    except Exception as e:
        print(f"执行价格异常检测失败: {e}")
        anomaly_html = "<p>异常检测失败，请检查数据完整性。</p>"
        anomaly_count = 0

    # 3. 生成最终的 HTML 仪表板
    if figures:
        generate_dashboard_html(figures, anomaly_html, anomaly_count)
    else:
        print("所有图表生成失败，无法创建仪表板。")