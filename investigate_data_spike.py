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

setup_chinese_fonts()

def load_and_process_data():
    """加载和处理数据"""
    print("正在加载IG债券数据...")
    
    # 读取IG债券数据
    ig_data = pd.read_excel('IG - 20.xlsx')
    print(f"IG债券数据形状: {ig_data.shape}")
    print("IG债券数据列名:", ig_data.columns.tolist())
    
    # 重命名列
    ig_data.columns = ['date', 'price']
    
    # 确保日期格式正确
    ig_data['date'] = pd.to_datetime(ig_data['date'])
    
    # 按日期排序
    ig_data = ig_data.sort_values('date').reset_index(drop=True)
    
    # 计算日收益率
    ig_data['daily_return'] = ig_data['price'].pct_change()
    
    # 删除第一行（因为没有前一日价格）
    ig_data = ig_data.dropna().reset_index(drop=True)
    
    print(f"处理后数据: {len(ig_data)} 行")
    print(f"日期范围: {ig_data['date'].min()} 到 {ig_data['date'].max()}")
    
    return ig_data

def calculate_annual_returns_detailed(data):
    """详细计算年化收益率，保留更多信息"""
    print("正在计算年化收益率...")
    
    results = []
    
    for i in range(len(data)):
        start_date = data['date'].iloc[i]
        end_date = start_date + timedelta(days=365)
        
        # 找到最接近一年后的数据点
        future_data = data[data['date'] > start_date]
        end_data = future_data[future_data['date'] <= end_date]
        
        if len(end_data) == 0:
            continue
            
        # 获取一年期间的所有日收益率
        period_data = data[(data['date'] >= start_date) & (data['date'] <= end_data['date'].iloc[-1])]
        
        if len(period_data) > 250:  # 确保至少有250个交易日
            # 计算累积收益率
            daily_returns = period_data['daily_return'].values
            cumulative_return = np.prod(1 + daily_returns) - 1
            
            # 记录详细信息
            results.append({
                'start_date': start_date,
                'end_date': end_data['date'].iloc[-1],
                'annual_return': cumulative_return,
                'start_price': data['price'].iloc[i],
                'end_price': end_data['price'].iloc[-1],
                'days_count': len(period_data),
                'year': start_date.year,
                'month': start_date.month
            })
            
        # 如果剩余数据不足一年，停止计算
        if len(data) - i < 250:
            break
    
    return pd.DataFrame(results)

def analyze_spike_phenomenon(results_df):
    """分析3%附近的数据集中现象"""
    print("\n=== 分析3%附近的数据集中现象 ===")
    
    # 找出收益率在2.5%到3.5%之间的数据
    spike_data = results_df[(results_df['annual_return'] >= 0.025) & (results_df['annual_return'] <= 0.035)]
    
    print(f"总样本数: {len(results_df)}")
    print(f"2.5%-3.5%区间内的样本数: {len(spike_data)}")
    print(f"占比: {len(spike_data)/len(results_df)*100:.2f}%")
    
    if len(spike_data) > 0:
        print(f"\n3%附近数据的时间分布:")
        year_counts = spike_data['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}年: {count} 个样本")
        
        print(f"\n3%附近数据的统计信息:")
        print(f"  平均收益率: {spike_data['annual_return'].mean()*100:.3f}%")
        print(f"  标准差: {spike_data['annual_return'].std()*100:.3f}%")
        print(f"  最小值: {spike_data['annual_return'].min()*100:.3f}%")
        print(f"  最大值: {spike_data['annual_return'].max()*100:.3f}%")
        
        # 检查价格变化
        print(f"\n价格变化分析:")
        print(f"  起始价格范围: {spike_data['start_price'].min():.2f} - {spike_data['start_price'].max():.2f}")
        print(f"  结束价格范围: {spike_data['end_price'].min():.2f} - {spike_data['end_price'].max():.2f}")
    
    return spike_data

def create_detailed_visualizations(results_df, spike_data):
    """创建详细的可视化分析"""
    print("\n正在创建详细分析图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('美国IG债券指数收益率异常现象分析', fontsize=16, fontweight='bold')
    
    # 1. 收益率分布直方图（高分辨率）
    axes[0, 0].hist(results_df['annual_return']*100, bins=100, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    axes[0, 0].axvline(x=3.0, color='red', linestyle='--', linewidth=2, label='3%基准线')
    axes[0, 0].set_xlabel('年化收益率 (%)')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('年化收益率分布（高分辨率）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 时间序列图
    axes[0, 1].plot(results_df['start_date'], results_df['annual_return']*100, alpha=0.7, linewidth=1)
    axes[0, 1].scatter(spike_data['start_date'], spike_data['annual_return']*100, 
                      color='red', s=20, alpha=0.8, label='3%附近数据')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('年化收益率 (%)')
    axes[0, 1].set_title('年化收益率时间序列')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 按年份统计
    yearly_stats = results_df.groupby('year')['annual_return'].agg(['mean', 'std', 'count']).reset_index()
    axes[0, 2].bar(yearly_stats['year'], yearly_stats['mean']*100, alpha=0.7, color='green')
    axes[0, 2].set_xlabel('年份')
    axes[0, 2].set_ylabel('平均年化收益率 (%)')
    axes[0, 2].set_title('各年份平均收益率')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. 3%附近数据的年份分布
    if len(spike_data) > 0:
        spike_yearly = spike_data['year'].value_counts().sort_index()
        axes[1, 0].bar(spike_yearly.index, spike_yearly.values, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('年份')
        axes[1, 0].set_ylabel('3%附近数据点数量')
        axes[1, 0].set_title('3%附近数据的年份分布')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 收益率vs起始日期散点图
    scatter = axes[1, 1].scatter(results_df['start_date'], results_df['annual_return']*100, 
                                c=results_df['year'], cmap='viridis', alpha=0.6, s=15)
    axes[1, 1].set_xlabel('起始日期')
    axes[1, 1].set_ylabel('年化收益率 (%)')
    axes[1, 1].set_title('收益率vs时间（按年份着色）')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.colorbar(scatter, ax=axes[1, 1], label='年份')
    
    # 6. 收益率密度图
    axes[1, 2].hist(results_df['annual_return']*100, bins=50, density=True, alpha=0.7, color='blue', label='所有数据')
    if len(spike_data) > 0:
        axes[1, 2].hist(spike_data['annual_return']*100, bins=20, density=True, alpha=0.8, color='red', label='3%附近数据')
    axes[1, 2].set_xlabel('年化收益率 (%)')
    axes[1, 2].set_ylabel('密度')
    axes[1, 2].set_title('收益率密度分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('IG债券收益率异常分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("分析图表已保存为 'IG债券收益率异常分析.png'")

def investigate_price_patterns(ig_data):
    """调查价格模式"""
    print("\n=== 调查价格模式 ===")
    
    # 检查价格的变化模式
    price_changes = ig_data['price'].diff()
    
    # 找出价格完全不变的连续期间
    no_change_periods = []
    current_start = None
    
    for i, change in enumerate(price_changes):
        if change == 0:  # 价格没有变化
            if current_start is None:
                current_start = i-1  # 记录开始位置
        else:
            if current_start is not None:
                # 结束一个无变化期间
                no_change_periods.append({
                    'start_idx': current_start,
                    'end_idx': i-1,
                    'duration': i - current_start,
                    'start_date': ig_data['date'].iloc[current_start],
                    'end_date': ig_data['date'].iloc[i-1],
                    'price': ig_data['price'].iloc[current_start]
                })
                current_start = None
    
    # 输出长期无变化的时期
    long_periods = [p for p in no_change_periods if p['duration'] > 10]  # 超过10天
    
    print(f"发现 {len(long_periods)} 个超过10天的价格无变化期间:")
    for period in long_periods:
        print(f"  {period['start_date'].strftime('%Y-%m-%d')} 到 {period['end_date'].strftime('%Y-%m-%d')}: "
              f"{period['duration']} 天，价格 {period['price']:.4f}")
    
    return no_change_periods

def main():
    try:
        # 1. 加载和处理数据
        ig_data = load_and_process_data()
        
        # 2. 计算详细的年化收益率
        results_df = calculate_annual_returns_detailed(ig_data)
        
        # 3. 分析3%附近的异常现象
        spike_data = analyze_spike_phenomenon(results_df)
        
        # 4. 调查价格模式
        no_change_periods = investigate_price_patterns(ig_data)
        
        # 5. 创建详细分析图表
        create_detailed_visualizations(results_df, spike_data)
        
        # 6. 保存详细结果
        results_df.to_excel('IG债券收益率详细分析.xlsx', index=False)
        print("\n详细分析结果已保存到 'IG债券收益率详细分析.xlsx'")
        
        # 7. 结论
        print("\n=== 分析结论 ===")
        print("3%附近的数据集中现象可能的原因：")
        print("1. 债券市场的利率环境相对稳定")
        print("2. 1-3年期投资级债券的收益率波动较小")
        print("3. 某些时期可能存在价格数据重复或缺失")
        print("4. 市场结构性因素导致收益率向某个水平收敛")
        
        if len(no_change_periods) > 0:
            total_no_change_days = sum(p['duration'] for p in no_change_periods)
            print(f"5. 发现总计 {total_no_change_days} 天的价格无变化期间，这可能影响收益率计算")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
