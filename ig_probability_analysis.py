import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """强化版中文字体设置"""
    import matplotlib.font_manager as fm
    import matplotlib
    
    # 清除matplotlib字体缓存
    try:
        matplotlib.font_manager._rebuild()
    except:
        pass
    
    # 获取系统可用字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 扩展的中文字体优先级列表
    preferred_fonts = [
        'Microsoft YaHei',  # 微软雅黑
        'SimHei',           # 黑体
        'KaiTi',           # 楷体
        'FangSong',        # 仿宋
        'STSong',          # 华文宋体
        'STKaiti',         # 华文楷体
        'STHeiti',         # 华文黑体
        'STFangsong',      # 华文仿宋
        'STXihei',         # 华文细黑
        'STZhongsong',     # 华文中宋
        'Microsoft JhengHei', # 微软正黑体
        'PingFang SC',     # 苹方
        'Hiragino Sans GB', # 冬青黑体
        'Source Han Sans CN', # 思源黑体
        'Noto Sans CJK SC',   # Noto中文
        'WenQuanYi Micro Hei', # 文泉驿微米黑
        'Arial Unicode MS'  # Arial Unicode MS
    ]
    
    # 找到所有可用的中文字体
    available_fonts = []
    for font in preferred_fonts:
        if font in font_list:
            available_fonts.append(font)
    
    if available_fonts:
        selected_font = available_fonts[0]
        print(f"使用字体: {selected_font}")
        
        # 强化字体设置
        plt.rcParams.update({
            'font.sans-serif': available_fonts,
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            'font.size': 10
        })
        
        # 额外设置matplotlib的默认字体
        matplotlib.rcParams['font.sans-serif'] = available_fonts
        matplotlib.rcParams['axes.unicode_minus'] = False
        
    else:
        print("警告: 未找到中文字体，使用系统默认字体")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    return True

# 设置中文字体
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
    
    print(f"美国3年期国债收益率数据: {len(treasury_data)} 行")
    print(f"IG债券指数数据: {len(ig_data)} 行")
    print(f"国债收益率日期范围: {treasury_data['date'].min()} 到 {treasury_data['date'].max()}")
    print(f"IG债券日期范围: {ig_data['date'].min()} 到 {ig_data['date'].max()}")
    
    return treasury_data, ig_data

def align_data(treasury_data, ig_data):
    """对齐数据"""
    print("\n正在对齐数据...")
    
    # 找到共同的日期范围
    common_start = max(treasury_data['date'].min(), ig_data['date'].min())
    common_end = min(treasury_data['date'].max(), ig_data['date'].max())
    
    print(f"共同日期范围: {common_start} 到 {common_end}")
    
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
    print("\n正在计算IG债券年化收益率...")
    
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

def calculate_current_probability(results_df, current_yield):
    """基于当前收益率水平计算概率"""
    print(f"\n正在计算当前收益率水平 ({current_yield:.2f}%) 下的概率...")
    
    # 找到最接近的收益率区间
    if current_yield <= 1:
        target_range = '0-1%'
    elif current_yield <= 2:
        target_range = '1-2%'
    elif current_yield <= 3:
        target_range = '2-3%'
    elif current_yield <= 4:
        target_range = '3-4%'
    elif current_yield <= 5:
        target_range = '4-5%'
    else:
        target_range = '5%+'
    
    # 获取对应区间的统计数据
    target_data = results_df[results_df['yield_range'] == target_range]
    
    if len(target_data) == 0:
        print(f"警告: 未找到收益率区间 {target_range} 的历史数据")
        return None, None
    
    target_data = target_data.iloc[0]
    
    positive_prob = target_data['positive_probability']
    avg_return = target_data['average_return']
    
    print(f"基于历史数据 ({target_data['count']} 个样本):")
    print(f"  收益率为正的概率: {positive_prob:.2%}")
    print(f"  平均年化收益率: {avg_return:.2%}")
    print(f"  标准差: {target_data['std_return']:.2%}")
    
    return positive_prob, avg_return

def create_visualizations(ig_returns, treasury_yields, yield_groups, results_df, current_yield):
    """创建可视化图表"""
    print("\n正在创建可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 设置主标题
    fig.suptitle('美国IG债券收益率概率分析', fontsize=18, fontweight='bold')
    
    # 1. 散点图：国债收益率 vs IG债券年化收益率
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    color_map = dict(zip(['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5%+'], colors))
    
    for group_label in ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5%+']:
        mask = yield_groups == group_label
        if np.sum(mask) > 0:
            axes[0, 0].scatter(treasury_yields[mask], ig_returns[mask] * 100, 
                             alpha=0.6, color=color_map[group_label], 
                             label=f'{group_label}', s=20)
    
    # 添加当前收益率水平的垂直线
    axes[0, 0].axvline(x=current_yield, color='red', linestyle='--', 
                      linewidth=2, label=f'当前收益率: {current_yield:.2f}%')
    
    axes[0, 0].set_xlabel('美国3年期国债收益率 (%)')
    axes[0, 0].set_ylabel('IG债券年化收益率 (%)')
    axes[0, 0].set_title('国债收益率与IG债券收益率关系')
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
    # 确定当前利率对应的区间
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
    plt.savefig('ig_probability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 'ig_probability_analysis.png'")

def main():
    try:
        # 1. 加载和预处理数据
        treasury_data, ig_data = load_and_preprocess_data()
        
        # 2. 对齐数据
        merged_data = align_data(treasury_data, ig_data)
        
        # 3. 计算IG债券年化收益率
        ig_returns, treasury_yields, start_dates = calculate_ig_returns(merged_data)
        
        # 4. 分析收益率关系
        results_df, yield_groups = analyze_yield_relationship(ig_returns, treasury_yields)
        
        # 5. 获取当前美国3年期国债收益率（使用最新数据）
        current_yield = treasury_data['us_3y_yield'].iloc[-1]
        print(f"\n当前美国3年期国债收益率: {current_yield:.2f}%")
        
        # 6. 计算当前利率水平下的概率
        positive_prob, avg_return = calculate_current_probability(results_df, current_yield)
        
        # 7. 输出详细结果
        print("\n" + "="*80)
        print("美国IG债券收益率概率分析结果")
        print("="*80)
        print(f"分析期间: {len(ig_returns)} 个年度收益率样本")
        print(f"当前美国3年期国债收益率: {current_yield:.2f}%")
        print()
        
        print("各利率区间统计数据:")
        for _, row in results_df.iterrows():
            print(f"{row['yield_range']} (样本数: {row['count']}):")
            print(f"  正收益率概率: {row['positive_probability']:.2%}")
            print(f"  平均年化收益率: {row['average_return']:.2%}")
            print(f"  标准差: {row['std_return']:.2%}")
            print(f"  平均国债收益率: {row['avg_yield']:.2f}%")
            print()
        
        if positive_prob is not None:
            print("基于当前利率水平的预测:")
            print(f"  未来1年IG债券收益率为正的概率: {positive_prob:.2%}")
            print(f"  预期年化收益率: {avg_return:.2%}")
            print(f"  增量收益（净值边际变化）预期: {avg_return:.2%}")
        
        print("="*80)
        
        # 8. 保存结果到CSV
        results_df_export = results_df.copy()
        results_df_export['positive_probability'] = results_df_export['positive_probability'] * 100
        results_df_export['average_return'] = results_df_export['average_return'] * 100
        results_df_export['std_return'] = results_df_export['std_return'] * 100
        
        # 添加当前预测
        if positive_prob is not None:
            current_prediction = pd.DataFrame({
                'yield_range': ['当前预测'],
                'count': [0],
                'positive_probability': [positive_prob * 100],
                'average_return': [avg_return * 100],
                'std_return': [0],
                'avg_yield': [current_yield]
            })
            results_df_export = pd.concat([results_df_export, current_prediction], ignore_index=True)
        
        results_df_export.to_csv('ig_probability_analysis_results.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存到 'ig_probability_analysis_results.csv'")
        
        # 9. 创建可视化图表
        create_visualizations(ig_returns, treasury_yields, yield_groups, results_df, current_yield)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

