import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

def get_table_data(url):
    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Referer': 'http://222.143.21.233:20181/21To23/pub/gongshi'
    }
    
    try:
        # 发送GET请求
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到表格
        table = soup.find('table')
        
        if table:
            # 提取表头
            headers = []
            for th in table.find_all('th'):
                headers.append(th.text.strip())
            
            # 提取表格数据
            rows = []
            for tr in table.find_all('tr')[1:]:  # 跳过表头行
                row = []
                for td in tr.find_all('td'):
                    row.append(td.text.strip())
                if row:  # 确保行不为空
                    rows.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(rows, columns=headers)
            return df
        else:
            print("未找到表格数据")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def process_page(page, base_url, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            url = f"{base_url}?pageIndex={page}"
            print(f"\n正在获取第 {page} 页...")
            df = get_table_data(url)
            if df is not None and not df.empty:
                print(f"第 {page} 页数据获取成功，获取到 {len(df)} 条记录")
                return page, df
            retries += 1
            time.sleep(1)
        except Exception as e:
            print(f"处理第 {page} 页时发生错误: {e}")
            retries += 1
            time.sleep(1)
    return page, None

def save_batch_data(data_list, start_page, end_page):
    if data_list:
        df = pd.concat(data_list, ignore_index=True)
        filename = f'table_data_batch_{start_page}_to_{end_page}.xlsx'
        df.to_excel(filename, index=False)
        print(f"已保存批次数据到文件: {filename}")
        return df
    return None

def main():
    base_url = "http://222.143.21.233:20181/21To23/pub/gongshi/GongShiSearch"
    total_pages = 15567
    max_workers = 5  # 同时运行的线程数
    batch_size = 100  # 每批处理的页数  
    all_data = []
    
    # 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 按批次处理页面
        for batch_start in range(1, total_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_pages)
            print(f"\n开始处理第 {batch_start} 到 {batch_end} 页...")
            
            # 创建当前批次的任务
            future_to_page = {
                executor.submit(process_page, page, base_url): page
                for page in range(batch_start, batch_end + 1)
            }
            
            # 收集当前批次的结果
            batch_data = []
            for future in as_completed(future_to_page):
                page, df = future.result()
                if df is not None:
                    batch_data.append(df)
            
            # 保存当前批次的数据
            if batch_data:
                batch_df = save_batch_data(batch_data, batch_start, batch_end)
                if batch_df is not None:
                    all_data.append(batch_df)
            
            # 批次间短暂休息
            time.sleep(2)
    
    # 合并所有批次的数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        output_file = 'data/table_data_all.xlsx'
        final_df.to_excel(output_file, index=False)
        print(f"\n所有数据已成功保存到 {output_file}")
        print(f"总共获取了 {len(final_df)} 条数据")
    else:
        print("未能获取任何数据")

if __name__ == "__main__":
    main()