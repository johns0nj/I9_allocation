import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    return True

setup_chinese_fonts()

def load_and_process_data():
    """加载并处理数据"""
    print("正在加载数据...")
    
    # 读取美国3年期国债收益率数据
    treasury_data = pd.read_excel('cn_us_3y.xlsx')
    treasury_data.columns = ['date', 'us_3y_yield', 'cn_3y_yield', 'spread']
    treasury_data['date'] = pd.to_datetime(treasury_data['date'])
    treasury_data = treasury_data[['date', 'us_3y_yield']].dropna()
    
    # 读取IG债券指数数据
    ig_data = pd.read_excel('IG - 20.xlsx')
    ig_data.columns = ['date', 'ig_index']
    ig_data['date'] = pd.to_datetime(ig_data['date'])
    ig_data = ig_data.dropna()
    
    # 合并数据
    merged_data = pd.merge(treasury_data, ig_data, on='date', how='inner')
    print(f"合并后数据: {len(merged_data)} 行")
    
    return merged_data

def calculate_ig_returns_with_dates(merged_data):
    """计算IG债券年化收益率并保留日期信息"""
    print("正在计算IG债券年化收益率...")
    
    # 计算日收益率
    merged_data['ig_daily_return'] = merged_data['ig_index'].pct_change()
    
    # 计算1年期收益率
    results = []
    
    for i in range(len(merged_data)):
        start_date = merged_data['date'].iloc[i]
        end_date = start_date + timedelta(days=365)
        
        # 找到1年后的数据点
        future_data = merged_data[merged_data['date'] > start_date]
        end_data = future_data[future_data['date'] <= end_date]
        
        if len(end_data) == 0:
            continue
            
        # 获取1年期间的所有日收益率
        period_data = merged_data[
            (merged_data['date'] >= start_date) & 
            (merged_data['date'] <= end_data['date'].iloc[-1])
        ]
        
        if len(period_data) > 250:  # 确保至少有250个交易日
            # 计算累积收益率
            daily_returns = period_data['ig_daily_return'].dropna()
            if len(daily_returns) > 0:
                cumulative_return = np.prod(1 + daily_returns) - 1
                
                results.append({
                    'start_date': start_date,
                    'treasury_yield': merged_data['us_3y_yield'].iloc[i],
                    'ig_annual_return': cumulative_return,
                    'year': start_date.year,
                    'month': start_date.month
                })
            
        # 如果剩余数据不足一年，停止计算
        if len(merged_data) - i < 250:
            break
    
    return pd.DataFrame(results)

def analyze_yield_inversion(results_df):
    """分析收益率倒挂现象"""
    print("\n=== 分析收益率倒挂现象 ===")
    
    # 定义两个关键区间
    range_2_3 = results_df[
        (results_df['treasury_yield'] >= 2.0) & 
        (results_df['treasury_yield'] < 3.0)
    ]
    
    range_3_4 = results_df[
        (results_df['treasury_yield'] >= 3.0) & 
        (results_df['treasury_yield'] < 4.0)
    ]
    
    print(f"2-3%区间样本数: {len(range_2_3)}")
    print(f"3-4%区间样本数: {len(range_3_4)}")
    
    if len(range_2_3) > 0 and len(range_3_4) > 0:
        avg_2_3 = range_2_3['ig_annual_return'].mean()
        avg_3_4 = range_3_4['ig_annual_return'].mean()
        
        print(f"2-3%区间平均IG收益率: {avg_2_3*100:.2f}%")
        print(f"3-4%区间平均IG收益率: {avg_3_4*100:.2f}%")
        print(f"差值: {(avg_2_3 - avg_3_4)*100:.2f}%")
        
        # 时间分布分析
        print(f"\n2-3%区间时间分布:")
        year_counts_2_3 = range_2_3['year'].value_counts().sort_index()
        for year, count in year_counts_2_3.items():
            avg_return = range_2_3[range_2_3['year'] == year]['ig_annual_return'].mean()
            print(f"  {year}年: {count}个样本, 平均收益率: {avg_return*100:.2f}%")
        
        print(f"\n3-4%区间时间分布:")
        year_counts_3_4 = range_3_4['year'].value_counts().sort_index()
        for year, count in year_counts_3_4.items():
            avg_return = range_3_4[range_3_4['year'] == year]['ig_annual_return'].mean()
            print(f"  {year}年: {count}个样本, 平均收益率: {avg_return*100:.2f}%")
        
        # 分析可能的原因
        print(f"\n可能的原因分析:")
        
        # 1. 时间分布差异
        median_year_2_3 = range_2_3['year'].median()
        median_year_3_4 = range_3_4['year'].median()
        print(f"1. 时间分布差异:")
        print(f"   2-3%区间中位年份: {median_year_2_3}")
        print(f"   3-4%区间中位年份: {median_year_3_4}")
        
        # 2. 市场环境差异
        print(f"2. 市场环境分析:")
        
        # 检查2008-2010年（金融危机）的影响
        crisis_2_3 = range_2_3[
            (range_2_3['year'] >= 2008) & (range_2_3['year'] <= 2010)
        ]
        crisis_3_4 = range_3_4[
            (range_3_4['year'] >= 2008) & (range_3_4['year'] <= 2010)
        ]
        
        if len(crisis_2_3) > 0:
            print(f"   金融危机期间2-3%区间: {len(crisis_2_3)}样本, 平均收益率: {crisis_2_3['ig_annual_return'].mean()*100:.2f}%")
        if len(crisis_3_4) > 0:
            print(f"   金融危机期间3-4%区间: {len(crisis_3_4)}样本, 平均收益率: {crisis_3_4['ig_annual_return'].mean()*100:.2f}%")
        
        # 检查2020-2023年（疫情和通胀）的影响
        recent_2_3 = range_2_3[range_2_3['year'] >= 2020]
        recent_3_4 = range_3_4[range_3_4['year'] >= 2020]
        
        if len(recent_2_3) > 0:
            print(f"   近期2-3%区间: {len(recent_2_3)}样本, 平均收益率: {recent_2_3['ig_annual_return'].mean()*100:.2f}%")
        if len(recent_3_4) > 0:
            print(f"   近期3-4%区间: {len(recent_3_4)}样本, 平均收益率: {recent_3_4['ig_annual_return'].mean()*100:.2f}%")
    
    return range_2_3, range_3_4

def create_detailed_analysis_charts(results_df, range_2_3, range_3_4):
    """创建详细分析图表"""
    print("\n正在创建详细分析图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('IG债券收益率倒挂现象深度分析', fontsize=16, fontweight='bold')
    
    # 1. 整体散点图，突出两个区间
    axes[0, 0].scatter(results_df['treasury_yield'], results_df['ig_annual_return']*100, 
                      alpha=0.3, s=10, color='lightgray', label='所有数据')
    axes[0, 0].scatter(range_2_3['treasury_yield'], range_2_3['ig_annual_return']*100, 
                      alpha=0.7, s=30, color='blue', label='2-3%区间')
    axes[0, 0].scatter(range_3_4['treasury_yield'], range_3_4['ig_annual_return']*100, 
                      alpha=0.7, s=30, color='red', label='3-4%区间')
    
    # 添加平均线
    if len(range_2_3) > 0:
        avg_2_3 = range_2_3['ig_annual_return'].mean() * 100
        axes[0, 0].axhline(y=avg_2_3, xmin=0.2, xmax=0.4, color='blue', 
                          linestyle='--', linewidth=2, label=f'2-3%平均: {avg_2_3:.2f}%')
    if len(range_3_4) > 0:
        avg_3_4 = range_3_4['ig_annual_return'].mean() * 100
        axes[0, 0].axhline(y=avg_3_4, xmin=0.4, xmax=0.6, color='red', 
                          linestyle='--', linewidth=2, label=f'3-4%平均: {avg_3_4:.2f}%')
    
    axes[0, 0].set_xlabel('美国3年期国债收益率 (%)')
    axes[0, 0].set_ylabel('IG债券年化收益率 (%)')
    axes[0, 0].set_title('收益率倒挂现象可视化')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 时间序列分析 - 2-3%区间
    if len(range_2_3) > 0:
        axes[0, 1].scatter(range_2_3['start_date'], range_2_3['ig_annual_return']*100, 
                          alpha=0.7, color='blue', s=20)
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('IG债券年化收益率 (%)')
        axes[0, 1].set_title('2-3%区间时间序列')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 时间序列分析 - 3-4%区间
    if len(range_3_4) > 0:
        axes[0, 2].scatter(range_3_4['start_date'], range_3_4['ig_annual_return']*100, 
                          alpha=0.7, color='red', s=20)
        axes[0, 2].set_xlabel('时间')
        axes[0, 2].set_ylabel('IG债券年化收益率 (%)')
        axes[0, 2].set_title('3-4%区间时间序列')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. 按年份对比平均收益率
    if len(range_2_3) > 0 and len(range_3_4) > 0:
        # 计算每年的平均收益率
        yearly_2_3 = range_2_3.groupby('year')['ig_annual_return'].mean() * 100
        yearly_3_4 = range_3_4.groupby('year')['ig_annual_return'].mean() * 100
        
        # 找到共同年份
        common_years = set(yearly_2_3.index) & set(yearly_3_4.index)
        common_years = sorted(list(common_years))
        
        if len(common_years) > 0:
            x = np.arange(len(common_years))
            width = 0.35
            
            bars1 = axes[1, 0].bar(x - width/2, [yearly_2_3[year] for year in common_years], 
                                  width, label='2-3%区间', alpha=0.7, color='blue')
            bars2 = axes[1, 0].bar(x + width/2, [yearly_3_4[year] for year in common_years], 
                                  width, label='3-4%区间', alpha=0.7, color='red')
            
            axes[1, 0].set_xlabel('年份')
            axes[1, 0].set_ylabel('平均IG债券收益率 (%)')
            axes[1, 0].set_title('各年份收益率对比')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(common_years, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 收益率分布对比
    if len(range_2_3) > 0 and len(range_3_4) > 0:
        axes[1, 1].hist(range_2_3['ig_annual_return']*100, bins=20, alpha=0.7, 
                       color='blue', label='2-3%区间', density=True)
        axes[1, 1].hist(range_3_4['ig_annual_return']*100, bins=20, alpha=0.7, 
                       color='red', label='3-4%区间', density=True)
        axes[1, 1].set_xlabel('IG债券年化收益率 (%)')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].set_title('收益率分布对比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 箱线图对比
    if len(range_2_3) > 0 and len(range_3_4) > 0:
        data_for_box = [range_2_3['ig_annual_return']*100, range_3_4['ig_annual_return']*100]
        bp = axes[1, 2].boxplot(data_for_box, labels=['2-3%区间', '3-4%区间'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.7)
        
        axes[1, 2].set_ylabel('IG债券年化收益率 (%)')
        axes[1, 2].set_title('收益率分布箱线图')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('IG债券收益率倒挂分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("分析图表已保存为 'IG债券收益率倒挂分析.png'")

def main():
    try:
        # 1. 加载和处理数据
        merged_data = load_and_process_data()
        
        # 2. 计算IG债券年化收益率（保留日期信息）
        results_df = calculate_ig_returns_with_dates(merged_data)
        
        # 3. 分析收益率倒挂现象
        range_2_3, range_3_4 = analyze_yield_inversion(results_df)
        
        # 4. 创建详细分析图表
        create_detailed_analysis_charts(results_df, range_2_3, range_3_4)
        
        # 5. 保存详细分析结果
        analysis_results = []
        
        if len(range_2_3) > 0:
            analysis_results.append({
                '区间': '2-3%',
                '样本数': len(range_2_3),
                '平均IG收益率(%)': range_2_3['ig_annual_return'].mean() * 100,
                '标准差(%)': range_2_3['ig_annual_return'].std() * 100,
                '中位年份': range_2_3['year'].median(),
                '最早年份': range_2_3['year'].min(),
                '最晚年份': range_2_3['year'].max()
            })
        
        if len(range_3_4) > 0:
            analysis_results.append({
                '区间': '3-4%',
                '样本数': len(range_3_4),
                '平均IG收益率(%)': range_3_4['ig_annual_return'].mean() * 100,
                '标准差(%)': range_3_4['ig_annual_return'].std() * 100,
                '中位年份': range_3_4['year'].median(),
                '最早年份': range_3_4['year'].min(),
                '最晚年份': range_3_4['year'].max()
            })
        
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_excel('IG债券收益率倒挂分析结果.xlsx', index=False)
        print("\n详细分析结果已保存到 'IG债券收益率倒挂分析结果.xlsx'")
        
        # 6. 结论
        print("\n" + "="*80)
        print("收益率倒挂现象分析结论")
        print("="*80)
        
        if len(range_2_3) > 0 and len(range_3_4) > 0:
            diff = (range_2_3['ig_annual_return'].mean() - range_3_4['ig_annual_return'].mean()) * 100
            print(f"确认存在倒挂现象: 2-3%区间比3-4%区间高 {diff:.2f}%")
            
            print("\n可能的解释:")
            print("1. 时间效应: 不同利率区间主要出现在不同的经济周期")
            print("2. 市场环境: 高利率时期往往伴随经济不确定性，影响IG债券表现")
            print("3. 流动性因素: 不同利率环境下的市场流动性差异")
            print("4. 信用利差: 经济周期对信用利差的影响")
            print("5. 数据清理: 之前移除的垂直线数据可能影响了统计结果")
        
        print("="*80)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
