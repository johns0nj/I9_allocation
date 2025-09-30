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
    
    # 测试中文显示
    try:
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.text(0.5, 0.5, '中文字体测试成功', fontsize=12, ha='center', va='center')
        plt.close(fig)
        print("中文字体设置成功")
        return True
    except Exception as e:
        print(f"中文字体设置可能有问题: {e}")
        return False

# 设置中文字体
setup_chinese_fonts()

def load_data():
    """加载IG债券指数数据"""
    print("正在读取IG债券指数数据...")
    
    # 读取IG债券指数数据
    ig_data = pd.read_excel('IG - 20.xlsx')
    print(f"IG债券指数数据形状: {ig_data.shape}")
    print("IG债券指数数据列名:", ig_data.columns.tolist())
    print("IG债券前5行数据:")
    print(ig_data.head())
    
    return ig_data

def preprocess_data(ig_data):
    """预处理数据，提取日期和价格，并计算日收益率"""
    print("\n正在预处理数据...")
    
    # 重命名列以便于处理
    ig_data.columns = ['date', 'price']
    
    # 确保日期格式正确
    ig_data['date'] = pd.to_datetime(ig_data['date'])
    
    # 按日期排序
    ig_data = ig_data.sort_values('date').reset_index(drop=True)
    
    # 计算日收益率
    ig_data['daily_return'] = ig_data['price'].pct_change()
    
    # 删除第一行（因为没有前一日价格）
    ig_data = ig_data.dropna().reset_index(drop=True)
    
    print(f"IG债券处理后数据: {len(ig_data)} 行")
    print(f"IG债券日期范围: {ig_data['date'].min()} 到 {ig_data['date'].max()}")
    
    return ig_data

def calculate_annual_returns(data):
    """计算从第一日起算1年后的收益率"""
    print(f"\n正在计算年化收益率...")
    
    annual_returns = []
    start_dates = []
    
    # 从第一日开始，计算每一年的收益率
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
        
        if len(period_data) > 250:  # 确保至少有250个交易日（约一年）
            # 计算累积收益率：(1+r1)*(1+r2)*...*(1+rn) - 1
            daily_returns = period_data['daily_return'].values
            cumulative_return = np.prod(1 + daily_returns) - 1
            annual_returns.append(cumulative_return)
            start_dates.append(start_date)
            
        # 如果剩余数据不足一年，停止计算
        if len(data) - i < 250:
            break
    
    print(f"计算得到 {len(annual_returns)} 个年度收益率")
    return np.array(annual_returns), start_dates

def calculate_ig_statistics(ig_returns, start_dates):
    """计算IG债券统计数据"""
    print(f"\n正在计算IG债券统计数据...")
    
    # 计算统计数据
    ig_mean = np.mean(ig_returns)
    ig_std = np.std(ig_returns, ddof=1)
    
    # 计算负收益率概率
    ig_negative_prob = np.sum(ig_returns < 0) / len(ig_returns)
    
    # 计算正收益率概率
    ig_positive_prob = np.sum(ig_returns > 0) / len(ig_returns)
    
    # 计算最大回撤
    cumulative_returns = np.cumprod(1 + ig_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # 计算夏普比率（假设无风险利率为2%）
    risk_free_rate = 0.02
    excess_returns = ig_returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    # 添加时间阶段分类
    time_periods = []
    for date in start_dates:
        if date.year <= 2010:
            time_periods.append('2005-2010')
        elif date.year <= 2015:
            time_periods.append('2011-2015')
        elif date.year <= 2020:
            time_periods.append('2016-2020')
        else:
            time_periods.append('2021-2025')
    
    return {
        'ig_mean': ig_mean,
        'ig_std': ig_std,
        'ig_returns': ig_returns,
        'ig_negative_prob': ig_negative_prob,
        'ig_positive_prob': ig_positive_prob,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'start_dates': start_dates,
        'time_periods': time_periods
    }

def create_visualizations(stats):
    """创建可视化图表"""
    print("\n正在创建可视化图表...")
    
    # 强制重新设置字体
    font_success = setup_chinese_fonts()
    if not font_success:
        print("警告: 中文字体设置可能存在问题")
    
    # 设置图表样式（避免覆盖字体设置）
    try:
        plt.style.use('seaborn-v0_8')
        # 重新应用字体设置，因为样式可能会覆盖
        setup_chinese_fonts()
    except:
        try:
            plt.style.use('seaborn')
            setup_chinese_fonts()
        except:
            print("使用默认样式")
    
    # 创建图表，明确指定字体参数
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 设置主标题，明确指定字体
    fig.suptitle('美国IG债券指数分析结果', fontsize=18, fontweight='bold', 
                 fontfamily='sans-serif')
    
    # 1. 年化收益率分布直方图（转换为百分比）
    axes[0, 0].hist(stats['ig_returns']*100, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=stats['ig_mean']*100, color='red', linestyle='--', linewidth=2, 
                      label=f'平均收益率: {stats["ig_mean"]*100:.2f}%')
    axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='零收益率')
    axes[0, 0].set_xlabel('年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontfamily='sans-serif', fontsize=12)
    axes[0, 0].set_title('IG债券年化收益率分布', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[0, 0].legend(prop={'family': 'sans-serif', 'size': 10})
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 收益率时间序列（转换为百分比）
    dates = range(len(stats['ig_returns']))
    axes[0, 1].plot(dates, stats['ig_returns']*100, color='blue', alpha=0.7, linewidth=1)
    axes[0, 1].axhline(y=stats['ig_mean']*100, color='red', linestyle='--', linewidth=2, 
                      label=f'平均收益率: {stats["ig_mean"]*100:.2f}%')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='零收益率')
    axes[0, 1].set_xlabel('时间（年）', fontfamily='sans-serif', fontsize=12)
    axes[0, 1].set_ylabel('年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[0, 1].set_title('IG债券年化收益率时间序列', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[0, 1].legend(prop={'family': 'sans-serif', 'size': 10})
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 按时间阶段分组的收益率箱线图
    time_periods = stats['time_periods']
    unique_periods = ['2005-2010', '2011-2015', '2016-2020', '2021-2025']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 红、青、蓝、绿
    color_map = dict(zip(unique_periods, colors))
    
    # 准备箱线图数据
    box_data = []
    box_labels = []
    box_colors = []
    
    for period in unique_periods:
        mask = [p == period for p in time_periods]
        if any(mask):
            period_returns = np.array(stats['ig_returns'])[mask] * 100
            box_data.append(period_returns)
            box_labels.append(period)
            box_colors.append(color_map[period])
    
    if box_data:
        bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='零收益率')
    axes[1, 0].set_xlabel('时间阶段', fontfamily='sans-serif', fontsize=12)
    axes[1, 0].set_ylabel('年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[1, 0].set_title('不同时间阶段的收益率分布', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[1, 0].legend(prop={'family': 'sans-serif', 'size': 10})
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累积收益率曲线
    cumulative_returns = np.cumprod(1 + stats['ig_returns']) * 100  # 转换为百分比
    axes[1, 1].plot(range(len(cumulative_returns)), cumulative_returns, color='blue', linewidth=2)
    axes[1, 1].set_xlabel('时间（年）', fontfamily='sans-serif', fontsize=12)
    axes[1, 1].set_ylabel('累积收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[1, 1].set_title('IG债券累积收益率曲线', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f'''统计摘要:
平均年化收益率: {stats['ig_mean']*100:.2f}%
标准差: {stats['ig_std']*100:.2f}%
正收益率概率: {stats['ig_positive_prob']*100:.1f}%
负收益率概率: {stats['ig_negative_prob']*100:.1f}%
最大回撤: {stats['max_drawdown']*100:.2f}%
夏普比率: {stats['sharpe_ratio']:.3f}'''
    
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                   fontfamily='sans-serif', fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('ig_bond_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 'ig_bond_analysis.png'")

def main():
    try:
        # 1. 加载数据
        ig_data = load_data()
        
        # 2. 预处理数据
        ig_processed = preprocess_data(ig_data)
        
        # 3. 计算年化收益率
        print("\n计算IG债券年化收益率...")
        ig_annual_returns, ig_dates = calculate_annual_returns(ig_processed)
        
        # 4. 计算统计数据
        stats = calculate_ig_statistics(ig_annual_returns, ig_dates)
        
        # 5. 输出结果
        print("\n" + "="*60)
        print("美国IG债券指数分析结果")
        print("="*60)
        print(f"分析期间: {len(ig_annual_returns)} 个年度收益率样本")
        print(f"数据时间范围: {ig_dates[0].year} - {ig_dates[-1].year}")
        print()
        print("IG债券指数统计:")
        print(f"  平均年化收益率: {stats['ig_mean']:.4f} ({stats['ig_mean']*100:.2f}%)")
        print(f"  标准差: {stats['ig_std']:.4f} ({stats['ig_std']*100:.2f}%)")
        print(f"  正收益率概率: {stats['ig_positive_prob']:.4f} ({stats['ig_positive_prob']*100:.2f}%)")
        print(f"  负收益率概率: {stats['ig_negative_prob']:.4f} ({stats['ig_negative_prob']*100:.2f}%)")
        print(f"  最大回撤: {stats['max_drawdown']:.4f} ({stats['max_drawdown']*100:.2f}%)")
        print(f"  夏普比率: {stats['sharpe_ratio']:.4f}")
        print()
        
        # 按时间阶段分析
        print("按时间阶段分析:")
        time_periods = stats['time_periods']
        unique_periods = ['2005-2010', '2011-2015', '2016-2020', '2021-2025']
        
        for period in unique_periods:
            mask = [p == period for p in time_periods]
            if any(mask):
                period_returns = np.array(stats['ig_returns'])[mask]
                period_mean = np.mean(period_returns)
                period_std = np.std(period_returns)
                period_positive_prob = np.sum(period_returns > 0) / len(period_returns)
                print(f"  {period}:")
                print(f"    平均收益率: {period_mean*100:.2f}%")
                print(f"    标准差: {period_std*100:.2f}%")
                print(f"    正收益率概率: {period_positive_prob*100:.1f}%")
                print(f"    样本数: {len(period_returns)}")
        
        print("="*60)
        
        # 6. 保存结果到CSV和Excel
        results_df = pd.DataFrame({
            '指标': ['平均年化收益率(%)', '标准差(%)', '正收益率概率(%)', '负收益率概率(%)', '最大回撤(%)', '夏普比率'],
            '数值': [stats['ig_mean']*100, stats['ig_std']*100, stats['ig_positive_prob']*100, 
                    stats['ig_negative_prob']*100, stats['max_drawdown']*100, stats['sharpe_ratio']]
        })
        
        # 保存为CSV
        results_df.to_csv('ig_bond_analysis_results.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存到 'ig_bond_analysis_results.csv'")
        
        # 保存为Excel，包含多个工作表
        with pd.ExcelWriter('IG债券分析结果.xlsx', engine='openpyxl') as writer:
            # 主要统计结果
            results_df.to_excel(writer, sheet_name='统计摘要', index=False)
            
            # 详细收益率数据
            detailed_df = pd.DataFrame({
                '年份': [date.year for date in stats['start_dates']],
                'IG债券年化收益率(%)': stats['ig_returns']*100,
                '时间阶段': stats['time_periods']
            })
            detailed_df.to_excel(writer, sheet_name='详细收益率数据', index=False)
            
            # 按时间阶段分析
            period_analysis_data = []
            for period in unique_periods:
                mask = [p == period for p in time_periods]
                if any(mask):
                    period_returns = np.array(stats['ig_returns'])[mask]
                    period_analysis_data.append({
                        '时间阶段': period,
                        '样本数': len(period_returns),
                        '平均收益率(%)': np.mean(period_returns)*100,
                        '标准差(%)': np.std(period_returns)*100,
                        '正收益率概率(%)': np.sum(period_returns > 0) / len(period_returns) * 100,
                        '最大收益率(%)': np.max(period_returns)*100,
                        '最小收益率(%)': np.min(period_returns)*100
                    })
            
            period_analysis_df = pd.DataFrame(period_analysis_data)
            period_analysis_df.to_excel(writer, sheet_name='时间阶段分析', index=False)
        
        print("详细结果已保存到 'IG债券分析结果.xlsx'")
        
        # 7. 创建可视化图表
        create_visualizations(stats)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
