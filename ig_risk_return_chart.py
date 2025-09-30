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
            'font.size': 12
        })
        
        # 额外设置matplotlib的默认字体
        matplotlib.rcParams['font.sans-serif'] = available_fonts
        matplotlib.rcParams['axes.unicode_minus'] = False
        
    else:
        print("警告: 未找到中文字体，使用系统默认字体")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    return True

def load_and_process_data():
    """加载并处理IG债券数据"""
    print("正在加载IG债券数据...")
    
    # 读取IG债券指数数据
    ig_data = pd.read_excel('IG - 20.xlsx')
    ig_data.columns = ['date', 'price']
    ig_data['date'] = pd.to_datetime(ig_data['date'])
    ig_data = ig_data.sort_values('date').reset_index(drop=True)
    
    # 计算日收益率
    ig_data['daily_return'] = ig_data['price'].pct_change()
    ig_data = ig_data.dropna().reset_index(drop=True)
    
    print(f"IG债券数据: {len(ig_data)} 行")
    print(f"日期范围: {ig_data['date'].min()} 到 {ig_data['date'].max()}")
    
    return ig_data

def calculate_annual_returns(data):
    """计算年化收益率"""
    print("正在计算年化收益率...")
    
    annual_returns = []
    start_dates = []
    
    for i in range(len(data)):
        start_date = data['date'].iloc[i]
        end_date = start_date + timedelta(days=365)
        
        future_data = data[data['date'] > start_date]
        end_data = future_data[future_data['date'] <= end_date]
        
        if len(end_data) == 0:
            continue
            
        period_data = data[(data['date'] >= start_date) & (data['date'] <= end_data['date'].iloc[-1])]
        
        if len(period_data) > 250:  # 确保至少有250个交易日
            daily_returns = period_data['daily_return'].values
            cumulative_return = np.prod(1 + daily_returns) - 1
            annual_returns.append(cumulative_return)
            start_dates.append(start_date)
            
        if len(data) - i < 250:
            break
    
    print(f"计算得到 {len(annual_returns)} 个年度收益率")
    return np.array(annual_returns), start_dates

def create_ig_risk_return_chart():
    """创建IG债券风险收益图（仅显示风险-收益图格式）"""
    print("正在创建IG债券风险收益图...")
    
    # 设置中文字体
    setup_chinese_fonts()
    
    # 加载数据
    ig_data = load_and_process_data()
    ig_returns, start_dates = calculate_annual_returns(ig_data)
    
    # 计算统计数据
    ig_mean = np.mean(ig_returns)
    ig_std = np.std(ig_returns, ddof=1)
    
    # 创建图表 - 模仿原始投资组合分析中右下角的格式
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 设置图表样式（模仿原始样式）
    try:
        plt.style.use('seaborn-v0_8')
        setup_chinese_fonts()
    except:
        try:
            plt.style.use('seaborn')
            setup_chinese_fonts()
        except:
            print("使用默认样式")
    
    # 绘制IG债券风险收益点（模仿原始格式）
    # 使用与原始图表相同的颜色和样式
    ax.scatter(ig_std * 100, ig_mean * 100, s=1000, color='blue', 
              alpha=0.7, edgecolors='black', linewidth=1, label='IG债券')
    
    # 添加详细的数字标注（模仿原始格式）
    annotation_text = f'IG债券\n收益率: {ig_mean*100:.2f}%\n标准差: {ig_std*100:.2f}%'
    ax.annotate(annotation_text, (ig_std * 100, ig_mean * 100), 
               xytext=(10, 10), textcoords='offset points',
               fontfamily='sans-serif', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
               ha='left', va='bottom')
    
    # 设置坐标轴标签（模仿原始格式）
    ax.set_xlabel('标准差（风险）(%)', fontfamily='sans-serif', fontsize=12)
    ax.set_ylabel('平均收益率 (%)', fontfamily='sans-serif', fontsize=12)
    ax.set_title('风险-收益图（圆圈大小代表权重）', fontfamily='sans-serif', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围（确保IG债券点可见）
    ax.set_xlim(0, max(ig_std * 100 * 2, 5))
    ax.set_ylim(min(ig_mean * 100 - 1, 0), max(ig_mean * 100 + 1, 5))
    
    # 添加网格（模仿原始格式）
    ax.grid(True, alpha=0.3)
    
    # 添加图例（模仿原始格式）
    ax.legend(prop={'family': 'sans-serif', 'size': 10}, loc='upper left')
    
    # 添加权重说明（模仿原始格式）
    ax.text(0.02, 0.98, '圆圈面积 ∝ 投资权重', transform=ax.transAxes, 
           fontfamily='sans-serif', fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('ig_bond_risk_return_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("IG债券风险收益图已保存为 'ig_bond_risk_return_chart.png'")
    
    # 输出详细统计信息
    print("\n" + "="*60)
    print("美国IG债券指数风险收益分析结果")
    print("="*60)
    print(f"分析期间: {len(ig_returns)} 个年度收益率样本")
    print(f"数据时间范围: {start_dates[0].year} - {start_dates[-1].year}")
    print()
    print("风险收益指标:")
    print(f"  平均年化收益率: {ig_mean:.4f} ({ig_mean*100:.2f}%)")
    print(f"  标准差（风险）: {ig_std:.4f} ({ig_std*100:.2f}%)")
    print(f"  夏普比率: {sharpe_ratio:.4f}")
    print(f"  最大回撤: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    print(f"  95% VaR: {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"  95% CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"  风险收益比: {ig_mean/ig_std:.4f}")
    print()
    print("收益率分布:")
    print(f"  正收益率概率: {np.sum(ig_returns > 0)/len(ig_returns)*100:.1f}%")
    print(f"  负收益率概率: {np.sum(ig_returns < 0)/len(ig_returns)*100:.1f}%")
    print(f"  最大收益率: {np.max(ig_returns)*100:.2f}%")
    print(f"  最小收益率: {np.min(ig_returns)*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    create_ig_risk_return_chart()
