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

def load_and_preprocess_data():
    """加载并预处理数据"""
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
    
    return treasury_data, ig_data

def align_data(treasury_data, ig_data):
    """对齐数据"""
    print("正在对齐数据...")
    
    # 找到共同的日期范围
    common_start = max(treasury_data['date'].min(), ig_data['date'].min())
    common_end = min(treasury_data['date'].max(), ig_data['date'].max())
    
    # 筛选共同日期范围内的数据
    treasury_aligned = treasury_data[
        (treasury_data['date'] >= common_start) & 
        (treasury_data['date'] <= common_end)
    ].reset_index(drop=True)
    
    ig_aligned = ig_data[
        (ig_data['date'] >= common_start) & 
        (ig_data['date'] <= common_end)
    ].reset_index(drop=True)
    
    # 合并数据（按日期）
    merged_data = pd.merge(treasury_aligned, ig_aligned, on='date', how='inner')
    print(f"合并后数据: {len(merged_data)} 行")
    
    return merged_data

def calculate_ig_returns(merged_data):
    """计算IG债券的年化收益率"""
    print("正在计算IG债券年化收益率...")
    
    # 计算日收益率
    merged_data['ig_daily_return'] = merged_data['ig_index'].pct_change()
    
    # 计算1年期收益率
    annual_returns = []
    start_dates = []
    treasury_yields = []
    
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
        
        if len(period_data) > 250:  # 确保至少有250个交易日（约一年）
            # 计算累积收益率
            daily_returns = period_data['ig_daily_return'].dropna()
            if len(daily_returns) > 0:
                cumulative_return = np.prod(1 + daily_returns) - 1
                annual_returns.append(cumulative_return)
                start_dates.append(start_date)
                treasury_yields.append(merged_data['us_3y_yield'].iloc[i])
            
        # 如果剩余数据不足一年，停止计算
        if len(merged_data) - i < 250:
            break
    
    print(f"计算得到 {len(annual_returns)} 个年度收益率样本")
    
    return np.array(annual_returns), np.array(treasury_yields), start_dates

def analyze_vertical_line_issue(ig_returns, treasury_yields):
    """分析3%附近的垂直线问题"""
    print("\n分析3%附近的垂直线问题...")
    
    # 找出3%附近的数据
    target_yield = 3.0
    tolerance = 0.1  # ±0.1%
    
    near_3_mask = (treasury_yields >= target_yield - tolerance) & (treasury_yields <= target_yield + tolerance)
    near_3_treasury = treasury_yields[near_3_mask]
    near_3_ig = ig_returns[near_3_mask]
    
    print(f"3%附近（±{tolerance}%）的数据点数量: {len(near_3_treasury)}")
    
    if len(near_3_treasury) > 0:
        print(f"国债收益率范围: {near_3_treasury.min():.3f}% - {near_3_treasury.max():.3f}%")
        print(f"IG收益率范围: {near_3_ig.min()*100:.2f}% - {near_3_ig.max()*100:.2f}%")
        
        # 检查是否有完全相同的国债收益率值
        unique_yields = np.unique(near_3_treasury)
        print(f"唯一的国债收益率值数量: {len(unique_yields)}")
        
        # 找出重复最多的收益率值
        from collections import Counter
        yield_counts = Counter(near_3_treasury.round(3))  # 四舍五入到3位小数
        most_common = yield_counts.most_common(5)
        
        print("最常见的国债收益率值:")
        for yield_val, count in most_common:
            if count > 1:
                print(f"  {yield_val:.3f}%: {count} 次")
                # 找出这个收益率对应的IG收益率分布
                same_yield_mask = np.abs(near_3_treasury - yield_val) < 0.001
                same_yield_ig = near_3_ig[same_yield_mask]
                print(f"    对应IG收益率: {same_yield_ig.min()*100:.2f}% - {same_yield_ig.max()*100:.2f}%")
    
    return near_3_mask

def remove_problematic_data(ig_returns, treasury_yields, threshold_count=10):
    """移除形成垂直线的问题数据"""
    print(f"\n移除形成垂直线的数据（阈值: {threshold_count}个重复值）...")
    
    # 统计每个国债收益率值的出现次数
    from collections import Counter
    yield_counts = Counter(treasury_yields.round(3))  # 四舍五入到3位小数
    
    # 找出出现次数过多的收益率值
    problematic_yields = [yield_val for yield_val, count in yield_counts.items() 
                         if count >= threshold_count]
    
    print(f"发现 {len(problematic_yields)} 个问题收益率值:")
    for yield_val in problematic_yields:
        count = yield_counts[yield_val]
        print(f"  {yield_val:.3f}%: {count} 次")
    
    # 创建掩码，移除问题数据
    keep_mask = np.ones(len(treasury_yields), dtype=bool)
    
    for yield_val in problematic_yields:
        # 找出这个收益率的所有数据点
        same_yield_mask = np.abs(treasury_yields - yield_val) < 0.001
        same_yield_indices = np.where(same_yield_mask)[0]
        
        # 只保留一部分数据点（随机采样）
        if len(same_yield_indices) > 5:  # 如果超过5个点，只保留5个
            np.random.seed(42)  # 固定随机种子以保证可重复性
            keep_indices = np.random.choice(same_yield_indices, 5, replace=False)
            remove_indices = np.setdiff1d(same_yield_indices, keep_indices)
            keep_mask[remove_indices] = False
    
    # 应用掩码
    filtered_ig_returns = ig_returns[keep_mask]
    filtered_treasury_yields = treasury_yields[keep_mask]
    
    print(f"原始数据点: {len(ig_returns)}")
    print(f"过滤后数据点: {len(filtered_ig_returns)}")
    print(f"移除数据点: {len(ig_returns) - len(filtered_ig_returns)}")
    
    return filtered_ig_returns, filtered_treasury_yields

def analyze_yield_relationship(ig_returns, treasury_yields):
    """分析收益率与国债收益率的关系"""
    print("\n正在分析收益率关系...")
    
    # 创建收益率区间
    yield_bins = [0, 1, 2, 3, 4, 5, 100]  # 收益率区间
    yield_labels = ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5%+']
    
    # 将国债收益率分组
    yield_groups = pd.cut(treasury_yields, bins=yield_bins, labels=yield_labels, include_lowest=True)
    
    # 计算每个收益率区间的统计数据
    results = []
    for label in yield_labels:
        mask = yield_groups == label
        if np.sum(mask) > 0:
            group_returns = ig_returns[mask]
            positive_prob = np.mean(group_returns > 0)
            avg_return = np.mean(group_returns)
            std_return = np.std(group_returns)
            count = len(group_returns)
            
            results.append({
                'yield_range': label,
                'count': count,
                'positive_probability': positive_prob,
                'average_return': avg_return,
                'std_return': std_return,
                'avg_yield': np.mean(treasury_yields[mask])
            })
    
    results_df = pd.DataFrame(results)
    return results_df, yield_groups

def create_clean_visualizations(ig_returns, treasury_yields, yield_groups, results_df, current_yield):
    """创建清理后的可视化图表"""
    print("\n正在创建清理后的可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 设置主标题
    fig.suptitle('美国IG债券收益率概率分析（已清理垂直线）', fontsize=18, fontweight='bold')
    
    # 1. 散点图：国债收益率 vs IG债券年化收益率（清理后）
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    color_map = dict(zip(['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5%+'], colors))
    
    for group_label in ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5%+']:
        mask = yield_groups == group_label
        if np.sum(mask) > 0:
            axes[0, 0].scatter(treasury_yields[mask], ig_returns[mask] * 100, 
                             alpha=0.6, color=color_map[group_label], 
                             label=f'{group_label}', s=20)
    
    axes[0, 0].set_xlabel('美国3年期国债收益率 (%)')
    axes[0, 0].set_ylabel('IG债券年化收益率 (%)')
    axes[0, 0].set_title('国债收益率与IG债券收益率关系（已清理）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各收益率区间的正收益率概率
    axes[0, 1].bar(results_df['yield_range'], results_df['positive_probability'] * 100, 
                  color=[color_map[x] for x in results_df['yield_range']], alpha=0.7)
    axes[0, 1].set_xlabel('国债收益率区间')
    axes[0, 1].set_ylabel('正收益率概率 (%)')
    axes[0, 1].set_title('不同利率环境下的正收益率概率')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(results_df['positive_probability'] * 100):
        axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. 各收益率区间的平均收益率
    axes[1, 0].bar(results_df['yield_range'], results_df['average_return'] * 100, 
                  color=[color_map[x] for x in results_df['yield_range']], alpha=0.7)
    axes[1, 0].set_xlabel('国债收益率区间')
    axes[1, 0].set_ylabel('平均年化收益率 (%)')
    axes[1, 0].set_title('不同利率环境下的平均收益率')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加数值标签
    for i, v in enumerate(results_df['average_return'] * 100):
        axes[1, 0].text(i, v + 0.1 if v > 0 else v - 0.3, f'{v:.2f}%', 
                       ha='center', va='bottom' if v > 0 else 'top')
    
    # 4. 收益率分布直方图（按当前利率区间）
    if current_yield <= 1:
        current_range = '0-1%'
    elif current_yield <= 2:
        current_range = '1-2%'
    elif current_yield <= 3:
        current_range = '2-3%'
    elif current_yield <= 4:
        current_range = '3-4%'
    elif current_yield <= 5:
        current_range = '4-5%'
    else:
        current_range = '5%+'
    
    current_mask = yield_groups == current_range
    if np.sum(current_mask) > 0:
        current_returns = ig_returns[current_mask] * 100
        axes[1, 1].hist(current_returns, bins=30, alpha=0.7, 
                       color=color_map[current_range], edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='零收益率')
        axes[1, 1].set_xlabel('年化收益率 (%)')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title(f'当前利率区间 ({current_range}) 下的收益率分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        pos_prob = np.mean(current_returns > 0)
        avg_ret = np.mean(current_returns)
        axes[1, 1].text(0.05, 0.95, f'正收益率概率: {pos_prob:.1%}\n平均收益率: {avg_ret:.2f}%', 
                       transform=axes[1, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('ig_probability_analysis_clean.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("清理后的图表已保存为 'ig_probability_analysis_clean.png'")

def main():
    try:
        # 1. 加载和预处理数据
        treasury_data, ig_data = load_and_preprocess_data()
        
        # 2. 对齐数据
        merged_data = align_data(treasury_data, ig_data)
        
        # 3. 计算IG债券年化收益率
        ig_returns, treasury_yields, start_dates = calculate_ig_returns(merged_data)
        
        # 4. 分析垂直线问题
        near_3_mask = analyze_vertical_line_issue(ig_returns, treasury_yields)
        
        # 5. 移除形成垂直线的问题数据
        clean_ig_returns, clean_treasury_yields = remove_problematic_data(
            ig_returns, treasury_yields, threshold_count=15)
        
        # 6. 分析清理后的收益率关系
        results_df, yield_groups = analyze_yield_relationship(clean_ig_returns, clean_treasury_yields)
        
        # 7. 获取当前收益率
        current_yield = treasury_data['us_3y_yield'].iloc[-1]
        
        # 8. 创建清理后的可视化图表
        create_clean_visualizations(clean_ig_returns, clean_treasury_yields, 
                                   yield_groups, results_df, current_yield)
        
        print(f"\n清理完成！")
        print(f"原始数据点: {len(ig_returns)}")
        print(f"清理后数据点: {len(clean_ig_returns)}")
        print(f"移除的垂直线数据点: {len(ig_returns) - len(clean_ig_returns)}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
