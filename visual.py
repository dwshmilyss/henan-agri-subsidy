import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def ensure_dir(directory):
    """确保输出目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_to_excel(data_dict, output_dir):
    """将所有统计数据保存到Excel文件"""
    writer = pd.ExcelWriter(f'{output_dir}/统计分析数据.xlsx', engine='openpyxl')
    
    # 保存各个数据表
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name)
    
    writer.close()
    print(f"\n统计数据已保存至：{output_dir}/统计分析数据.xlsx")

def plot_equipment_distribution(df, output_dir):
    """机具品目分布分析"""
    equipment_counts = df['机具品目'].value_counts()
    
    # 创建饼图
    plt.figure(figsize=(15, 10))
    plt.pie(equipment_counts, labels=equipment_counts.index, autopct='%1.1f%%', 
            textprops={'fontsize': 8},
            pctdistance=0.85,
            labeldistance=1.1)
    plt.title('农机具品目分布', pad=20)
    plt.savefig(f'{output_dir}/机具分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 准备Excel数据
    equipment_df = pd.DataFrame({
        '机具品目': equipment_counts.index,
        '数量': equipment_counts.values,
        '占比': equipment_counts.values / len(df) * 100
    })
    
    return {'机具品目分布': equipment_df}

def plot_county_analysis(df, output_dir):
    """各县数据分析"""
    county_stats = df.groupby('县').agg({
        '单台中央补贴额(元)': ['sum', 'mean', 'count']
    }).round(2)
    
    county_stats.columns = ['总补贴金额', '平均补贴金额', '设备数量']
    county_stats = county_stats.sort_values('总补贴金额', ascending=True)
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    ax = county_stats['总补贴金额'].plot(kind='barh')
    plt.title('各县补贴金额总和', pad=20)
    plt.xlabel('补贴金额（元）')
    plt.yticks(fontsize=8)
    
    for i, v in enumerate(county_stats['总补贴金额']):
        ax.text(v, i, f'{v:,.0f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/各县补贴金额.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'各县统计': county_stats}

def plot_time_analysis(df, output_dir):
    """时间趋势分析"""
    df['购机日期'] = pd.to_datetime(df['购机日期'])
    
    # 按月统计
    monthly_stats = df.groupby(df['购机日期'].dt.to_period('M')).agg({
        '单台中央补贴额(元)': ['sum', 'count']
    })
    monthly_stats.columns = ['月度补贴金额', '月度购置数量']
    
    # 补贴金额趋势图
    plt.figure(figsize=(15, 6))
    monthly_stats['月度补贴金额'].plot(kind='line', marker='o')
    plt.title('月度补贴金额趋势')
    plt.xlabel('月份')
    plt.ylabel('补贴金额（元）')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/月度补贴趋势.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 设备数量趋势图
    plt.figure(figsize=(15, 6))
    monthly_stats['月度购置数量'].plot(kind='bar')
    plt.title('月度设备购置数量')
    plt.xlabel('月份')
    plt.ylabel('数量（台）')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/月度购置数量.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'月度统计': monthly_stats}

def plot_manufacturer_analysis(df, output_dir):
    """生产厂家分析"""
    # 准备数据
    manufacturer_counts = df['生产厂家'].value_counts().head(20)
    manufacturer_subsidy = df.groupby('生产厂家')['单台中央补贴额(元)'].sum().sort_values(ascending=False).head(20)
    
    # 厂家分布图
    plt.figure(figsize=(15, 10))
    ax = manufacturer_counts.plot(kind='bar')
    plt.title('主要生产厂家分布（Top 20）', pad=20)
    plt.xlabel('生产厂家')
    plt.ylabel('设备数量（台）')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    for i, v in enumerate(manufacturer_counts):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/生产厂家分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 厂家补贴金额图
    plt.figure(figsize=(15, 10))
    ax = manufacturer_subsidy.plot(kind='bar')
    plt.title('主要生产厂家补贴金额（Top 20）', pad=20)
    plt.xlabel('生产厂家')
    plt.ylabel('补贴金额（元）')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    for i, v in enumerate(manufacturer_subsidy):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/生产厂家补贴金额.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 准备Excel数据
    manufacturer_df = pd.DataFrame({
        '设备数量': manufacturer_counts,
        '补贴金额': manufacturer_subsidy
    })
    
    return {'生产厂家分析': manufacturer_df}

def generate_summary_statistics(df):
    # 计算总补贴金额
    total_subsidy = df['单台中央补贴额(元)'].sum()
    
    # 计算平均补贴金额
    average_subsidy = df['单台中央补贴额(元)'].mean()
    
    # 计算设备数量
    equipment_count = len(df)
    
    # 打印统计摘要
    print("\n统计摘要:")

def analyze_data():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel('table_data_all.xlsx')
    
    # 创建输出目录
    output_dir = 'analysis_results_2025'
    ensure_dir(output_dir)
    
    # 收集所有统计数据
    all_stats = {}
    
    # 执行各项分析并收集数据
    all_stats.update(plot_equipment_distribution(df, output_dir))
    all_stats.update(plot_county_analysis(df, output_dir))
    all_stats.update(plot_time_analysis(df, output_dir))
    all_stats.update(plot_manufacturer_analysis(df, output_dir))
    
    # 保存统计数据到Excel
    save_to_excel(all_stats, output_dir)
    
    # 生成统计摘要
    generate_summary_statistics(df)
        
    print(f"\n分析完成！所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    analyze_data()