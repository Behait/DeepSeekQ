"""
激进交易策略配置文件 - 针对100 USDT小资金优化
"""

AGGRESSIVE_CONFIG = {
    # 基础交易配置
    'symbol': 'BTC/USDT:USDT',
    'timeframe': '15m',
    'leverage': 20,  # 使用20倍杠杆
    'test_mode': True,  # 测试模式，实际交易时设为False
    
    # 激进风险参数
    'max_risk_per_trade': 0.10,  # 单笔交易最大风险10%（原2%）
    'max_position_ratio': 0.50,  # 最大持仓比例50%（原10%）
    'daily_loss_limit': 0.20,    # 单日最大亏损20%
    
    # 信心等级乘数调整（更激进）
    'confidence_multipliers': {
        'HIGH': 1.5,    # 高信心时使用150%仓位（原100%）
        'MEDIUM': 1.0,  # 中等信心100%（原70%）
        'LOW': 0.5      # 低信心50%（原30%）
    },
    
    # 波动率调整策略
    'volatility_adjustment': {
        'high_volatility_threshold': 0.04,  # 高波动率阈值4%（原3%）
        'medium_volatility_threshold': 0.025,  # 中等波动率2.5%（原2%）
        'high_vol_multiplier': 0.8,    # 高波动率时80%（原50%）
        'medium_vol_multiplier': 1.0,  # 中等波动率时100%（原80%）
        'low_vol_multiplier': 1.2      # 低波动率时120%（原100%）
    },
    
    # 趋势跟随增强
    'trend_following': {
        'strong_trend_multiplier': 1.3,  # 强势趋势时额外30%加成
        'breakout_multiplier': 1.5,     # 突破关键位时50%加成
        'momentum_threshold': 0.03      # 动量阈值3%
    },
    
    # 执行参数
    'min_trade_size': 0.001,  # 最小交易量0.001 BTC
    'position_adjust_threshold': 0.005,  # 仓位调整阈值0.5%
    'auto_compound': True     # 自动复利模式
}

# 激进信号过滤规则
AGGRESSIVE_SIGNAL_RULES = {
    # 允许的信号组合
    'allowed_combinations': [
        # 强势上涨 + 任何RSI → BUY
        {'trend': 'strong_uptrend', 'signal': 'BUY'},
        # 强势下跌 + 任何RSI → SELL  
        {'trend': 'strong_downtrend', 'signal': 'SELL'},
        # 突破阻力 + 放量 → BUY
        {'breakout': 'resistance', 'volume': 'high', 'signal': 'BUY'},
        # 跌破支撑 + 放量 → SELL
        {'breakout': 'support', 'volume': 'high', 'signal': 'SELL'}
    ],
    
    # 禁止的信号（在激进模式下减少HOLD）
    'restricted_signals': [
        # 减少过度保守的HOLD信号
        {'condition': 'rsi_range', 'min': 30, 'max': 70, 'signal': 'HOLD'},
        {'condition': 'bollinger_position', 'min': 0.2, 'max': 0.8, 'signal': 'HOLD'}
    ],
    
    # 强制执行的信号
    'forced_signals': [
        # 明确趋势中强制跟随
        {'trend': 'strong_uptrend', 'force_signal': 'BUY'},
        {'trend': 'strong_downtrend', 'force_signal': 'SELL'}
    ]
}

def get_aggressive_config():
    """获取激进配置"""
    return AGGRESSIVE_CONFIG

def get_aggressive_rules():
    """获取激进信号规则"""
    return AGGRESSIVE_SIGNAL_RULES

def calculate_aggressive_position(account_balance, signal_confidence, market_volatility, trend_strength):
    """
    计算激进仓位大小
    
    Args:
        account_balance: 账户余额
        signal_confidence: 信号信心等级
        market_volatility: 市场波动率
        trend_strength: 趋势强度
    """
    config = get_aggressive_config()
    
    # 基础风险金额
    max_risk_amount = account_balance * config['max_risk_per_trade']
    max_position_amount = account_balance * config['max_position_ratio']
    
    # 信心乘数
    confidence_multiplier = config['confidence_multipliers'].get(signal_confidence, 1.0)
    
    # 波动率调整
    if market_volatility > config['volatility_adjustment']['high_volatility_threshold']:
        vol_multiplier = config['volatility_adjustment']['high_vol_multiplier']
    elif market_volatility > config['volatility_adjustment']['medium_volatility_threshold']:
        vol_multiplier = config['volatility_adjustment']['medium_vol_multiplier']
    else:
        vol_multiplier = config['volatility_adjustment']['low_vol_multiplier']
    
    # 趋势加成
    trend_multiplier = 1.0
    if trend_strength > config['trend_following']['momentum_threshold']:
        trend_multiplier = config['trend_following']['strong_trend_multiplier']
    
    # 计算建议金额
    suggested_amount = max_risk_amount * confidence_multiplier * vol_multiplier * trend_multiplier
    
    # 确保不超过最大持仓限制
    final_amount = min(suggested_amount, max_position_amount)
    
    # 确保不低于最小交易量
    if final_amount < account_balance * 0.01:  # 至少1%仓位
        final_amount = account_balance * 0.01
    
    return final_amount