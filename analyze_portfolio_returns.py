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
    """加载恒生指数和IG债券指数数据"""
    print("正在读取数据文件...")
    
    # 读取恒生指数数据
    hsi_data = pd.read_excel('HSI - 20.xlsx')
    print(f"恒生指数数据形状: {hsi_data.shape}")
    print("恒生指数数据列名:", hsi_data.columns.tolist())
    print("恒生指数前5行数据:")
    print(hsi_data.head())
    
    # 读取IG债券指数数据
    ig_data = pd.read_excel('IG - 20.xlsx')
    print(f"\nIG债券指数数据形状: {ig_data.shape}")
    print("IG债券指数数据列名:", ig_data.columns.tolist())
    print("IG债券前5行数据:")
    print(ig_data.head())
    
    return hsi_data, ig_data

def preprocess_data(hsi_data, ig_data):
    """预处理数据，提取日期和价格，并计算日收益率"""
    print("\n正在预处理数据...")
    
    # 重命名列以便于处理
    hsi_data.columns = ['date', 'price']
    ig_data.columns = ['date', 'price']
    
    # 确保日期格式正确
    hsi_data['date'] = pd.to_datetime(hsi_data['date'])
    ig_data['date'] = pd.to_datetime(ig_data['date'])
    
    # 按日期排序
    hsi_data = hsi_data.sort_values('date').reset_index(drop=True)
    ig_data = ig_data.sort_values('date').reset_index(drop=True)
    
    # 计算日收益率
    hsi_data['daily_return'] = hsi_data['price'].pct_change()
    ig_data['daily_return'] = ig_data['price'].pct_change()
    
    # 删除第一行（因为没有前一日价格）
    hsi_data = hsi_data.dropna().reset_index(drop=True)
    ig_data = ig_data.dropna().reset_index(drop=True)
    
    print(f"恒生指数处理后数据: {len(hsi_data)} 行")
    print(f"IG债券处理后数据: {len(ig_data)} 行")
    print(f"恒生指数日期范围: {hsi_data['date'].min()} 到 {hsi_data['date'].max()}")
    print(f"IG债券日期范围: {ig_data['date'].min()} 到 {ig_data['date'].max()}")
    
    return hsi_data, ig_data

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

def align_data_by_date(hsi_data, ig_data):
    """按日期对齐两个数据集"""
    print("\n正在对齐数据...")
    
    # 找到共同的日期范围
    common_start = max(hsi_data['date'].min(), ig_data['date'].min())
    common_end = min(hsi_data['date'].max(), ig_data['date'].max())
    
    print(f"共同日期范围: {common_start} 到 {common_end}")
    
    # 筛选共同日期范围内的数据
    hsi_aligned = hsi_data[(hsi_data['date'] >= common_start) & (hsi_data['date'] <= common_end)].reset_index(drop=True)
    ig_aligned = ig_data[(ig_data['date'] >= common_start) & (ig_data['date'] <= common_end)].reset_index(drop=True)
    
    return hsi_aligned, ig_aligned

def calculate_portfolio_statistics(hsi_returns, ig_returns, hsi_weight=0.23, ig_weight=0.77):
    """计算投资组合统计数据"""
    print(f"\n正在计算投资组合统计数据（恒生指数权重: {hsi_weight*100}%, IG债券权重: {ig_weight*100}%）...")
    
    # 确保两个收益率数组长度相同
    min_length = min(len(hsi_returns), len(ig_returns))
    hsi_returns = hsi_returns[:min_length]
    ig_returns = ig_returns[:min_length]
    
    # 计算组合收益率
    portfolio_returns = hsi_weight * hsi_returns + ig_weight * ig_returns
    
    # 计算统计数据
    hsi_mean = np.mean(hsi_returns)
    hsi_std = np.std(hsi_returns, ddof=1)
    
    ig_mean = np.mean(ig_returns)
    ig_std = np.std(ig_returns, ddof=1)
    
    portfolio_mean = np.mean(portfolio_returns)
    portfolio_std = np.std(portfolio_returns, ddof=1)
    
    # 计算相关系数
    correlation = np.corrcoef(hsi_returns, ig_returns)[0, 1]
    
    # 计算负收益率概率
    hsi_negative_prob = np.sum(hsi_returns < 0) / len(hsi_returns)
    ig_negative_prob = np.sum(ig_returns < 0) / len(ig_returns)
    portfolio_negative_prob = np.sum(portfolio_returns < 0) / len(portfolio_returns)
    
    return {
        'hsi_mean': hsi_mean,
        'hsi_std': hsi_std,
        'ig_mean': ig_mean,
        'ig_std': ig_std,
        'portfolio_mean': portfolio_mean,
        'portfolio_std': portfolio_std,
        'correlation': correlation,
        'hsi_returns': hsi_returns,
        'ig_returns': ig_returns,
        'portfolio_returns': portfolio_returns,
        'hsi_negative_prob': hsi_negative_prob,
        'ig_negative_prob': ig_negative_prob,
        'portfolio_negative_prob': portfolio_negative_prob
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
    fig.suptitle('投资组合分析结果', fontsize=18, fontweight='bold', 
                 fontfamily='sans-serif')
    
    # 1. 年化收益率分布直方图（转换为百分比）
    axes[0, 0].hist(stats['hsi_returns']*100, bins=30, alpha=0.7, label='恒生指数', color='red')
    axes[0, 0].hist(stats['ig_returns']*100, bins=30, alpha=0.7, label='IG债券', color='blue')
    axes[0, 0].hist(stats['portfolio_returns']*100, bins=30, alpha=0.7, label='投资组合', color='green')
    axes[0, 0].set_xlabel('年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontfamily='sans-serif', fontsize=12)
    axes[0, 0].set_title('年化收益率分布', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[0, 0].legend(prop={'family': 'sans-serif', 'size': 10})
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 收益率时间序列（转换为百分比）
    dates = range(len(stats['hsi_returns']))
    axes[0, 1].plot(dates, stats['hsi_returns']*100, label='恒生指数', color='red', alpha=0.7)
    axes[0, 1].plot(dates, stats['ig_returns']*100, label='IG债券', color='blue', alpha=0.7)
    axes[0, 1].plot(dates, stats['portfolio_returns']*100, label='投资组合', color='green', linewidth=2)
    axes[0, 1].set_xlabel('时间（天）', fontfamily='sans-serif', fontsize=12)
    axes[0, 1].set_ylabel('年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[0, 1].set_title('年化收益率时间序列', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[0, 1].legend(prop={'family': 'sans-serif', 'size': 10})
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 散点图显示相关性（转换为百分比）
    axes[1, 0].scatter(stats['hsi_returns']*100, stats['ig_returns']*100, alpha=0.6, color='purple')
    axes[1, 0].set_xlabel('恒生指数年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[1, 0].set_ylabel('IG债券年化收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[1, 0].set_title(f'收益率相关性 (相关系数: {stats["correlation"]:.3f})', 
                         fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 风险收益图（转换为百分比，用权重表示面积大小）
    assets = ['恒生指数', 'IG债券', '投资组合']
    means = [stats['hsi_mean']*100, stats['ig_mean']*100, stats['portfolio_mean']*100]
    stds = [stats['hsi_std']*100, stats['ig_std']*100, stats['portfolio_std']*100]
    colors = ['red', 'blue', 'green']
    weights = [0.23, 0.77, 1.0]  # 权重：23%恒生指数，77%IG债券，100%投资组合
    base_size = 1000  # 基础圆圈大小
    
    for i, (asset, mean, std, color, weight) in enumerate(zip(assets, means, stds, colors, weights)):
        # 圆圈大小与权重成正比
        size = base_size * weight
        axes[1, 1].scatter(std, mean, s=size, label=f'{asset} ({weight*100:.0f}%)', 
                          color=color, alpha=0.7, edgecolors='black', linewidth=1)
        
        # 添加详细的数字标注（收益率和标准差）
        annotation_text = f'{asset}\n收益率: {mean:.2f}%\n标准差: {std:.2f}%'
        axes[1, 1].annotate(annotation_text, (std, mean), 
                           xytext=(10, 10), textcoords='offset points',
                           fontfamily='sans-serif', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                           ha='left', va='bottom')
    
    axes[1, 1].set_xlabel('标准差（风险）(%)', fontfamily='sans-serif', fontsize=12)
    axes[1, 1].set_ylabel('平均收益率 (%)', fontfamily='sans-serif', fontsize=12)
    axes[1, 1].set_title('风险-收益图（圆圈大小代表权重）', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(prop={'family': 'sans-serif', 'size': 10}, loc='upper left')
    
    # 添加权重说明
    axes[1, 1].text(0.02, 0.98, '圆圈面积 ∝ 投资权重', transform=axes[1, 1].transAxes, 
                   fontfamily='sans-serif', fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('portfolio_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 'portfolio_analysis_visualization.png'")

def main():
    try:
        # 1. 加载数据
        hsi_data, ig_data = load_data()
        
        # 2. 预处理数据
        hsi_processed, ig_processed = preprocess_data(hsi_data, ig_data)
        
        # 3. 对齐数据
        hsi_aligned, ig_aligned = align_data_by_date(hsi_processed, ig_processed)
        
        # 4. 计算年化收益率
        print("\n计算恒生指数年化收益率...")
        hsi_annual_returns, hsi_dates = calculate_annual_returns(hsi_aligned)
        
        print("\n计算IG债券年化收益率...")
        ig_annual_returns, ig_dates = calculate_annual_returns(ig_aligned)
        
        # 5. 计算统计数据
        stats = calculate_portfolio_statistics(hsi_annual_returns, ig_annual_returns)
        
        # 6. 输出结果
        print("\n" + "="*60)
        print("投资组合分析结果")
        print("="*60)
        print(f"分析期间: {len(hsi_annual_returns)} 个年度收益率样本")
        print()
        print("恒生指数:")
        print(f"  平均年化收益率: {stats['hsi_mean']:.4f} ({stats['hsi_mean']*100:.2f}%)")
        print(f"  标准差: {stats['hsi_std']:.4f} ({stats['hsi_std']*100:.2f}%)")
        print(f"  负收益率概率: {stats['hsi_negative_prob']:.4f} ({stats['hsi_negative_prob']*100:.2f}%)")
        print()
        print("美国IG债券指数:")
        print(f"  平均年化收益率: {stats['ig_mean']:.4f} ({stats['ig_mean']*100:.2f}%)")
        print(f"  标准差: {stats['ig_std']:.4f} ({stats['ig_std']*100:.2f}%)")
        print(f"  负收益率概率: {stats['ig_negative_prob']:.4f} ({stats['ig_negative_prob']*100:.2f}%)")
        print()
        print("投资组合 (23%恒生指数 + 77%IG债券):")
        print(f"  平均年化收益率: {stats['portfolio_mean']:.4f} ({stats['portfolio_mean']*100:.2f}%)")
        print(f"  标准差: {stats['portfolio_std']:.4f} ({stats['portfolio_std']*100:.2f}%)")
        print(f"  负收益率概率: {stats['portfolio_negative_prob']:.4f} ({stats['portfolio_negative_prob']*100:.2f}%)")
        print()
        print(f"恒生指数与IG债券相关系数: {stats['correlation']:.4f}")
        print("="*60)
        
        # 7. 保存结果到CSV和Excel
        results_df = pd.DataFrame({
            '资产': ['恒生指数', 'IG债券指数', '投资组合'],
            '平均年化收益率(%)': [stats['hsi_mean']*100, stats['ig_mean']*100, stats['portfolio_mean']*100],
            '标准差(%)': [stats['hsi_std']*100, stats['ig_std']*100, stats['portfolio_std']*100],
            '负收益率概率(%)': [stats['hsi_negative_prob']*100, stats['ig_negative_prob']*100, stats['portfolio_negative_prob']*100]
        })
        
        # 保存为CSV
        results_df.to_csv('portfolio_analysis_results.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存到 'portfolio_analysis_results.csv'")
        
        # 保存为Excel，包含多个工作表
        with pd.ExcelWriter('投资组合分析结果.xlsx', engine='openpyxl') as writer:
            # 主要统计结果
            results_df.to_excel(writer, sheet_name='统计摘要', index=False)
            
            # 详细收益率数据
            detailed_df = pd.DataFrame({
                '恒生指数年化收益率(%)': stats['hsi_returns']*100,
                'IG债券年化收益率(%)': stats['ig_returns']*100,
                '投资组合年化收益率(%)': stats['portfolio_returns']*100
            })
            detailed_df.to_excel(writer, sheet_name='详细收益率数据', index=False)
            
            # 负收益率详细分析
            negative_analysis_df = pd.DataFrame({
                '资产': ['恒生指数', 'IG债券指数', '投资组合'],
                '总样本数': [len(stats['hsi_returns']), len(stats['ig_returns']), len(stats['portfolio_returns'])],
                '负收益率次数': [np.sum(stats['hsi_returns'] < 0), 
                              np.sum(stats['ig_returns'] < 0), 
                              np.sum(stats['portfolio_returns'] < 0)],
                '负收益率概率(%)': [stats['hsi_negative_prob']*100, 
                                stats['ig_negative_prob']*100, 
                                stats['portfolio_negative_prob']*100],
                '平均负收益率(%)': [np.mean(stats['hsi_returns'][stats['hsi_returns'] < 0])*100 if np.any(stats['hsi_returns'] < 0) else 0,
                                np.mean(stats['ig_returns'][stats['ig_returns'] < 0])*100 if np.any(stats['ig_returns'] < 0) else 0,
                                np.mean(stats['portfolio_returns'][stats['portfolio_returns'] < 0])*100 if np.any(stats['portfolio_returns'] < 0) else 0]
            })
            negative_analysis_df.to_excel(writer, sheet_name='负收益率分析', index=False)
        
        print("详细结果已保存到 '投资组合分析结果.xlsx'")
        
        # 8. 创建可视化图表
        create_visualizations(stats)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
