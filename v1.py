import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Page, Bar, Pie, Line, Map, WordCloud
from pyecharts.globals import SymbolType

# -------------------------------
# 1️⃣ 读取数据
# -------------------------------
data = pd.read_excel("data/table_data_all.xlsx")
# -------------------------------
# 2️⃣ KPI 数据
# -------------------------------
total_machines = data["购买数量(台)"].sum()
total_subsidy = data["总补贴额(元)"].sum()
unique_users = data["购机者姓名"].nunique()
pending_count = data[data["状态"]=="待申请结算"].shape[0]

# KPI HTML 卡片动画
kpi_html = f"""
<div style="display:flex; justify-content:space-around; margin-bottom:20px; font-family:Arial;">
    <div style="text-align:center; font-size:32px; color:#00ff00;">总台数<br><span id='total_machines'>{total_machines}</span></div>
    <div style="text-align:center; font-size:32px; color:#00ff00;">总补贴<br><span id='total_subsidy'>{total_subsidy}</span>元</div>
    <div style="text-align:center; font-size:32px; color:#00ff00;">购机用户数<br><span id='unique_users'>{unique_users}</span></div>
    <div style="text-align:center; font-size:32px; color:#ff0000;">待申请结算<br><span id='pending_count'>{pending_count}</span></div>
</div>
<script>
function animateValue(id, start, end, duration) {{
    var range = end - start;
    var current = start;
    var increment = range / (duration/50);
    var obj = document.getElementById(id);
    var timer = setInterval(function(){{
        current += increment;
        if((increment>0 && current>=end) || (increment<0 && current<=end)){{
            current = end;
            clearInterval(timer);
        }}
        obj.innerHTML = Math.floor(current);
    }}, 50);
}}
animateValue("total_machines", 0, {total_machines}, 1500);
animateValue("total_subsidy", 0, {total_subsidy}, 1500);
animateValue("unique_users", 0, {unique_users}, 1500);
animateValue("pending_count", 0, {pending_count}, 1500);
</script>
"""

# -------------------------------
# 3️⃣ 农机品目分布饼图
# -------------------------------
machine_counts = data.groupby("机具品目")["购买数量(台)"].sum().reset_index()
pie_machine = (
    Pie()
    .add("", [list(z) for z in zip(machine_counts["机具品目"], machine_counts["购买数量(台)"])])
    .set_global_opts(title_opts=opts.TitleOpts(title="农机品目分布"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}台"))
)

# -------------------------------
# 4️⃣ 县级购买分布柱状图
# -------------------------------
county_data = data.groupby("县")["购买数量(台)"].sum().reset_index()
bar_county = (
    Bar()
    .add_xaxis(county_data["县"].tolist())
    .add_yaxis("购买数量", county_data["购买数量(台)"].tolist())
    .set_global_opts(title_opts=opts.TitleOpts(title="县级农机购买分布"))
)

# -------------------------------
# 5️⃣ 购机趋势折线图
# -------------------------------
data['购机日期'] = pd.to_datetime(data['购机日期'])
trend_data = data.groupby('购机日期')['购买数量(台)'].sum().reset_index()
line_trend = (
    Line()
    .add_xaxis([d.strftime("%Y-%m-%d") for d in trend_data['购机日期']])
    .add_yaxis("购机数量", trend_data['购买数量(台)'].tolist(), is_smooth=True)
    .set_global_opts(title_opts=opts.TitleOpts(title="购机趋势"))
)

# -------------------------------
# 6️⃣ 农机状态分布环形图
# -------------------------------
status_counts = data.groupby("状态")["购买数量(台)"].sum().reset_index()
pie_status = (
    Pie()
    .add("", [list(z) for z in zip(status_counts["状态"], status_counts["购买数量(台)"])],
         radius=["40%", "70%"])
    .set_global_opts(title_opts=opts.TitleOpts(title="农机状态分布"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}台"))
)

# -------------------------------
# 7️⃣ 地图展示（县级分布）
# -------------------------------
# 这里用示例数据，实际可用高德坐标/GeoJSON
map_data = [(row["县"], row["购买数量(台)"]) for idx,row in county_data.iterrows()]
map_chart = (
    Map()
    .add("购买数量", map_data, "china-cities")  # 可替换为具体省市 GeoJSON
    .set_global_opts(
        title_opts=opts.TitleOpts(title="县级农机分布地图"),
        visualmap_opts=opts.VisualMapOpts(max_=max(county_data["购买数量(台)"]))
    )
)

# -------------------------------
# 8️⃣ 词云图（购机者热度）
# -------------------------------
user_counts = data.groupby("购机者姓名")["购买数量(台)"].sum().reset_index()
wordcloud = (
    WordCloud()
    .add("", [list(z) for z in zip(user_counts["购机者姓名"], user_counts["购买数量(台)"])],
         word_size_range=[20,80], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="购机者热力词云"))
)

# -------------------------------
# 9️⃣ 合并大屏页面
# -------------------------------
page = Page(layout=Page.SimplePageLayout, page_title="农机大屏可视化")
page.add(bar_county, pie_machine, line_trend, pie_status, map_chart, wordcloud)
page.render("agri_dashboard_full_map_wordcloud.html")

# -------------------------------
# 10️⃣ 输出提示
# -------------------------------
print("大屏 HTML 已生成：agri_dashboard_full_map_wordcloud.html")
print("KPI 数字 HTML（可嵌入页面显示）：")
print(kpi_html)