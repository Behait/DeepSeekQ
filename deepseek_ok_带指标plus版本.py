import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# 导入风险管理系统
from risk_management import initialize_risk_manager, get_risk_manager

# 设置日志系统
def setup_logging():
    """设置日志系统"""
    logger = logging.getLogger('crypto_trading_bot')
    logger.setLevel(logging.INFO)
    
    # 文件处理器 - 滚动日志文件
    file_handler = RotatingFileHandler(
        'trading_bot.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 全局日志对象
trading_logger = setup_logging()

def log_trade_signal(signal_data, price_data):
    """记录交易信号"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': 'trade_signal',
        'signal': signal_data.get('signal'),
        'confidence': signal_data.get('confidence'),
        'price': price_data.get('price'),
        'price_change': price_data.get('price_change'),
        'stop_loss': signal_data.get('stop_loss'),
        'take_profit': signal_data.get('take_profit'),
        'reason': signal_data.get('reason')
    }
    
    if TRADE_CONFIG.get('verbose', False):
        trading_logger.info(f"交易信号: {json.dumps(log_entry, ensure_ascii=False)}")

def log_api_call(api_type, success, details):
    """记录API调用"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': 'api_call',
        'api_type': api_type,
        'success': success,
        'details': details
    }
    
    if TRADE_CONFIG.get('verbose', False):
        trading_logger.info(f"API调用: {json.dumps(log_entry, ensure_ascii=False)}")

def log_error(error_type, error_message, traceback_info=None):
    """记录错误信息"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': 'error',
        'error_type': error_type,
        'message': error_message,
        'traceback': traceback_info
    }
    
    trading_logger.error(f"错误信息: {json.dumps(log_entry, ensure_ascii=False)}")

def get_performance_stats():
    """获取性能统计"""
    return {
        'api_calls': api_call_stats,
        'total_trades': len(signal_history),
        'last_signal': signal_history[-1] if signal_history else None,
        'current_position': position,
        'timestamp': datetime.now().isoformat()
    }

def monitor_performance():
    """性能监控函数，定期记录性能指标"""
    stats = get_performance_stats()
    
    # 计算成功率
    total_calls = int(stats['api_calls']['total'])
    success_calls = int(stats['api_calls']['success'])
    success_rate = (success_calls / total_calls * 100) if total_calls > 0 else 0
    
    performance_log = {
        'timestamp': datetime.now().isoformat(),
        'api_calls_total': total_calls,
        'api_calls_success': success_calls,
        'api_success_rate': f"{success_rate:.2f}%",
        'total_trades': stats['total_trades'],
        'exchange_calls': stats['api_calls']['exchange'],
        'deepseek_calls': stats['api_calls']['deepseek']
    }
    
    if TRADE_CONFIG.get('verbose', False):
        trading_logger.info(f"性能监控: {json.dumps(performance_log, ensure_ascii=False)}")
    
    # 每10次API调用记录一次性能
    if total_calls % 10 == 0:
        print(f"📊 性能统计 - API调用: {total_calls}, 成功率: {success_rate:.2f}%, 交易次数: {stats['total_trades']}")

load_dotenv()

# 全局变量初始化
position = None
signal_history = []
api_call_stats = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'exchange': 0,
    'deepseek': 0
}

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 初始化OKX交易所
exchange = None  # 将在setup_exchange中初始化

# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'amount': 0.01,  # 交易数量 (BTC)
    'leverage': 10,  # 杠杆倍数
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': False,  # 测试模式
    'data_points': 96,  # 24小时数据（96根15分钟K线）
    'analysis_periods': {
        'short_term': 20,  # 短期均线
        'medium_term': 50,  # 中期均线
        'long_term': 96  # 长期趋势
    },
    'verbose': False  # 是否输出调试与原始JSON信息
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

# 添加缓存和性能优化相关的全局变量
price_data_cache = None
last_calculation_time = 0
calculation_cache = {}

# 添加API调用统计
api_call_stats = {
    'exchange': {'success': 0, 'fail': 0, 'last_call': 0},
    'deepseek': {'success': 0, 'fail': 0, 'last_call': 0}
}

# 受控调试输出
def vprint(*args, **kwargs):
    if TRADE_CONFIG.get('verbose', False):
        print(*args, **kwargs)

def get_cached_technical_data():
    """获取缓存的技术数据，避免重复计算"""
    global price_data_cache, last_calculation_time, calculation_cache
    
    current_time = time.time()
    
    # 如果缓存有效且未过期（15分钟内），直接返回缓存数据
    if (price_data_cache and 
        current_time - last_calculation_time < 900 and  # 15分钟
        not TRADE_CONFIG['test_mode']):
        return price_data_cache
    
    # 需要重新获取数据
    price_data = get_btc_ohlcv_enhanced()
    if price_data:
        price_data_cache = price_data
        last_calculation_time = current_time
    
    return price_data


def get_current_position():
    """获取当前持仓信息"""
    try:
        if TRADE_CONFIG['test_mode']:
            # 测试模式下返回模拟持仓
            return {
                'side': 'none',
                'size': 0.0,
                'entry_price': 0.0,
                'unrealized_pnl': 0.0,
                'leverage': TRADE_CONFIG['leverage']
            }
        
        # 确保exchange已初始化
        global exchange
        if exchange is None:
            if not setup_exchange():
                raise Exception("交易所未初始化")
        
        # 获取持仓信息
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])
        
        if positions:
            position = positions[0]
            return {
                'side': position['side'],
                'size': float(position['contracts']),
                'entry_price': float(position['entryPrice']),
                'unrealized_pnl': float(position['unrealizedPnl']),
                'leverage': TRADE_CONFIG['leverage']
            }
        else:
            return {
                'side': 'none',
                'size': 0.0,
                'entry_price': 0.0,
                'unrealized_pnl': 0.0,
                'leverage': TRADE_CONFIG['leverage']
            }
            
    except Exception as e:
        error_msg = f"获取持仓失败: {e}"
        print(error_msg)
        log_error('POSITION_FETCH_ERROR', error_msg, str(e))
        return {
            'side': 'none',
            'size': 0.0,
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'leverage': TRADE_CONFIG['leverage']
        }


def format_position_text(pos):
    """将持仓信息格式化为友好的摘要文本"""
    try:
        side = (pos or {}).get('side', 'none')
        size = float((pos or {}).get('size', 0) or 0)
        entry_price = float((pos or {}).get('entry_price', 0) or 0)
        pnl = float((pos or {}).get('unrealized_pnl', 0) or 0)
        leverage = (pos or {}).get('leverage', TRADE_CONFIG.get('leverage', 1))

        if size <= 0 or side == 'none':
            return f"当前持仓: 无持仓 | 杠杆 {leverage}x"

        side_cn = '多仓' if side == 'long' else ('空仓' if side == 'short' else side)
        return (
            f"当前持仓: {side_cn} {size:.6f} BTC | 开仓价 ${entry_price:,.2f} "
            f"| 未实现盈亏 ${pnl:,.2f} | 杠杆 {leverage}x"
        )
    except Exception:
        return "当前持仓: 信息不可用"

def setup_exchange():
    """设置交易所参数"""
    global exchange
    
    if TRADE_CONFIG['test_mode']:
        print("测试模式：跳过真实交易所连接")
        print("将使用模拟数据进行分析")
        return True
    
    try:
        # 只在非测试模式下初始化真实交易所
        exchange = ccxt.okx({
            'options': {
                'defaultType': 'swap',  # OKX使用swap表示永续合约
            },
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET'),
            'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
        })
        
        # OKX设置杠杆
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}  # 全仓模式
        )
        print(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 获取余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"当前USDT余额: {usdt_balance:.2f}")

        return True
    except Exception as e:
        print(f"交易所设置失败: {e}")
        return False


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        # 调试信息 - 检查数据类型
        vprint(f"Debug RSI: gain type={type(gain.iloc[-1]) if len(gain) > 0 else 'empty'}, loss type={type(loss.iloc[-1]) if len(loss) > 0 else 'empty'}")
        vprint(f"Debug RSI: gain value={gain.iloc[-1] if len(gain) > 0 else 'N/A'}, loss value={loss.iloc[-1] if len(loss) > 0 else 'N/A'}")
        
        # 确保类型安全，转换为浮点数
        gain_float = gain.astype(float)
        loss_float = loss.astype(float)
        
        # 避免除零错误
        rs = gain_float / loss_float.where(loss_float != 0, 1)
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        # 计算布林带位置，避免分母为0导致错误
        bb_denom = (df['bb_upper'] - df['bb_lower']).replace(0, pd.NA)
        df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_denom).fillna(0)

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        # 成交量比率，避免分母为0或NaN
        vol_denom = df['volume_ma'].replace(0, pd.NA)
        df['volume_ratio'] = (df['volume'] / vol_denom).fillna(1)

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        # 调试信息：检查类型
        vprint(f"Debug support_resistance: current_price={current_price} (type: {type(current_price)})")
        vprint(f"Debug support_resistance: resistance_level={resistance_level} (type: {type(resistance_level)})")
        vprint(f"Debug support_resistance: support_level={support_level} (type: {type(support_level)})")
        
        # 确保类型安全
        current_price_float = float(current_price)
        resistance_level_float = float(resistance_level)
        support_level_float = float(support_level)
        
        return {
            'static_resistance': resistance_level_float,
            'static_support': support_level_float,
            'dynamic_resistance': float(bb_upper),
            'dynamic_support': float(bb_lower),
            'price_vs_resistance': ((resistance_level_float - current_price_float) / current_price_float) * 100 if current_price_float != 0 else 0,
            'price_vs_support': ((current_price_float - support_level_float) / support_level_float) * 100 if support_level_float != 0 else 0
        }
    except Exception as e:
        print(f"支撑阻力计算失败: {e}")
        return {}


def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}


def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        if TRADE_CONFIG['test_mode']:
            # 测试模式：生成模拟数据
            import random
            import numpy as np
            
            # 生成模拟的K线数据
            base_price = 45000  # 基础价格
            timestamps = []
            ohlcv_data = []
            
            current_time = datetime.now()
            for i in range(TRADE_CONFIG['data_points']):
                # 生成时间戳（每15分钟一个）
                timestamp = current_time - pd.Timedelta(minutes=15 * (TRADE_CONFIG['data_points'] - i - 1))
                timestamps.append(int(timestamp.timestamp() * 1000))
                
                # 生成价格数据（随机波动）
                price_change = random.uniform(-0.02, 0.02)  # ±2%的随机波动
                current_price = base_price * (1 + price_change * i / 100)
                
                high = current_price * (1 + random.uniform(0, 0.01))
                low = current_price * (1 - random.uniform(0, 0.01))
                open_price = current_price * (1 + random.uniform(-0.005, 0.005))
                close_price = current_price
                volume = random.uniform(100, 1000)
                
                ohlcv_data.append([timestamps[-1], open_price, high, low, close_price, volume])
            
            print("测试模式：使用模拟市场数据")
        else:
            # 真实模式：从交易所获取数据
            ohlcv_data = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                         limit=TRADE_CONFIG['data_points'])

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 调试信息：检查数据类型
        vprint(f"Debug get_btc_ohlcv_enhanced: current_data['close']={current_data['close']} (type: {type(current_data['close'])})")
        vprint(f"Debug get_btc_ohlcv_enhanced: previous_data['close']={previous_data['close']} (type: {type(previous_data['close'])})")

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        return {
            'price': float(current_data['close']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': float(current_data['high']),
            'low': float(current_data['low']),
            'volume': float(current_data['volume']),
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((float(current_data['close']) - float(previous_data['close'])) / float(previous_data['close'])) * 100 if float(previous_data['close']) != 0 else 0,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
            'technical_data': {
                'sma_5': current_data.get('sma_5', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_50': current_data.get('sma_50', 0),
                'rsi': current_data.get('rsi', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'macd_histogram': current_data.get('macd_histogram', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0),
                'bb_position': current_data.get('bb_position', 0),
                'volume_ratio': current_data.get('volume_ratio', 0)
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }
    except Exception as e:
        print(f"获取增强K线数据失败: {e}")
        return None


def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    
    # 检查 price_data 是否为空
    if not price_data:
        return "【技术指标分析】\n数据获取失败，无法进行技术分析"
    
    if 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})
    levels = price_data.get('levels_analysis', {})

    # 检查数据有效性
    def safe_float(value, default=0):
        try:
            if value is None:
                return default
            # 处理pandas NaN和其他NaN情况
            if hasattr(value, '__float__'):
                return float(value)
            if str(value).lower() in ['nan', 'null', 'none', '']:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    # 计算相对价格百分比，避免除零错误
    def calc_relative_percent(price, ma_value):
        # 确保类型安全，转换为浮点数
        price_float = safe_float(price, 0)
        ma_float = safe_float(ma_value, 0)
        if ma_float == 0:
            return 0.0
        
        # 调试信息（受控）
        vprint(f"Debug calc_relative_percent: price={price}, ma_value={ma_value}")
        vprint(f"Debug calc_relative_percent: price_float={price_float}, ma_float={ma_float}")
        
        return (price_float - ma_float) / ma_float * 100
    
    sma_5_val = safe_float(tech['sma_5'])
    sma_20_val = safe_float(tech['sma_20'])
    sma_50_val = safe_float(tech['sma_50'])
    
    # 调试信息（受控）：检查calc_relative_percent调用前的参数
    vprint(f"Debug generate_technical_analysis_text: price_data['price']={price_data['price']} (type: {type(price_data['price'])})")
    vprint(f"Debug generate_technical_analysis_text: sma_5_val={sma_5_val} (type: {type(sma_5_val)})")
    vprint(f"Debug generate_technical_analysis_text: sma_20_val={sma_20_val} (type: {type(sma_20_val)})")
    vprint(f"Debug generate_technical_analysis_text: sma_50_val={sma_50_val} (type: {type(sma_50_val)})")
    
    analysis_text = f"""
    【技术指标分析】
    📈 移动平均线:
    - 5周期: {sma_5_val:.2f} | 价格相对: {calc_relative_percent(price_data['price'], sma_5_val):+.2f}%
    - 20周期: {sma_20_val:.2f} | 价格相对: {calc_relative_percent(price_data['price'], sma_20_val):+.2f}%
    - 50周期: {sma_50_val:.2f} | 价格相对: {calc_relative_percent(price_data['price'], sma_50_val):+.2f}%

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向: {trend.get('macd', 'N/A')}

    📊 动量指标:
    - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f}
    - 信号线: {safe_float(tech['macd_signal']):.4f}

    🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

    💰 关键水平:
    - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
    - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}
    """
    return analysis_text


def safe_api_call(func, max_retries=3, retry_delay=1, api_type='exchange'):
    """安全的API调用包装器，带重试机制"""
    for attempt in range(max_retries):
        try:
            result = func()
            api_call_stats[api_type]['success'] += 1
            api_call_stats[api_type]['last_call'] = time.time()
            log_api_call(api_type, True, f"调用成功 - 尝试 {attempt + 1}")
            return result
        except Exception as e:
            api_call_stats[api_type]['fail'] += 1
            error_msg = f"{api_type.upper()} API调用第{attempt + 1}次失败: {e}"
            print(error_msg)
            log_api_call(api_type, False, error_msg)
            
            if attempt == max_retries - 1:
                log_error("API_FINAL_FAILURE", f"{api_type} API调用最终失败", str(e))
                raise e
            
            # 指数退避等待
            wait_time = retry_delay * (2 ** attempt)
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    return None

def get_exchange_data_with_retry():
    """带重试的交易所数据获取"""
    def fetch_data():
        return exchange.fetch_ohlcv(
            TRADE_CONFIG['symbol'], 
            TRADE_CONFIG['timeframe'],
            limit=TRADE_CONFIG['data_points']
        )
    
    return safe_api_call(fetch_data, api_type='exchange')

def calculate_position_size(price_data, signal_data, account_balance):
    """动态计算仓位大小，基于风险管理和市场波动率"""
    
    # 导入激进配置
    try:
        from aggressive_config import get_aggressive_config
        config = get_aggressive_config()
        
        # 使用激进配置参数
        RISK_PARAMS = {
            'max_risk_per_trade': config['max_risk_per_trade'],
            'max_position_ratio': config['max_position_ratio'],
            'volatility_multiplier': 1.0
        }
        
        # 使用激进信心乘数
        confidence_multiplier = config['confidence_multipliers'].get(
            signal_data.get('confidence', 'MEDIUM'), 1.0
        )
        
    except ImportError:
        # 回退到保守配置
        RISK_PARAMS = {
            'max_risk_per_trade': 0.02,  # 单笔交易最大风险2%
            'max_position_ratio': 0.2,   # 最大持仓比例10%
            'volatility_multiplier': 1.0
        }
        
        confidence_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.7,
            'LOW': 0.3
        }.get(signal_data.get('confidence', 'MEDIUM'), 0.5)
    
    # 根据市场波动率调整仓位
    volatility = calculate_market_volatility(price_data)
    
    try:
        from aggressive_config import get_aggressive_config
        config = get_aggressive_config()
        if volatility > config['volatility_adjustment']['high_volatility_threshold']:
            RISK_PARAMS['volatility_multiplier'] = config['volatility_adjustment']['high_vol_multiplier']
        elif volatility > config['volatility_adjustment']['medium_volatility_threshold']:
            RISK_PARAMS['volatility_multiplier'] = config['volatility_adjustment']['medium_vol_multiplier']
        else:
            RISK_PARAMS['volatility_multiplier'] = config['volatility_adjustment']['low_vol_multiplier']
    except ImportError:
        # 回退到保守波动率调整
        if volatility > 0.03:  # 高波动率市场
            RISK_PARAMS['volatility_multiplier'] = 0.5
        elif volatility > 0.02:  # 中等波动率
            RISK_PARAMS['volatility_multiplier'] = 0.8
        else:  # 低波动率
            RISK_PARAMS['volatility_multiplier'] = 1.0
    
    # 趋势强度加成
    trend_multiplier = 1.0
    if 'trend_analysis' in price_data:
        trend_strength = 0.0
        trend_text = price_data['trend_analysis'].get('overall', '')
        
        if 'strong' in trend_text.lower():
            trend_strength = 0.05  # 强势趋势
        elif 'uptrend' in trend_text.lower() or 'downtrend' in trend_text.lower():
            trend_strength = 0.03  # 普通趋势
            
        try:
            from aggressive_config import get_aggressive_config
            config = get_aggressive_config()
            if trend_strength > 0:
                trend_multiplier = config['trend_following']['strong_trend_multiplier']
        except ImportError:
            # 保守模式下不使用趋势加成
            pass
    
    # 计算最大可投入金额
    max_risk_amount = account_balance * RISK_PARAMS['max_risk_per_trade']
    max_position_amount = account_balance * RISK_PARAMS['max_position_ratio']
    
    # 应用调整因子
    suggested_amount = max_risk_amount * confidence_multiplier * RISK_PARAMS['volatility_multiplier'] * trend_multiplier
    
    # 确保不超过最大持仓限制
    final_amount = min(suggested_amount, max_position_amount)
    
    # 计算合约数量
    contract_size = final_amount / price_data['price']
    
    print(
        f"📊 仓位计算: 余额=${account_balance:,.2f}, 风险=${max_risk_amount:,.2f}, "
        f"信心×{confidence_multiplier}, 波动×{RISK_PARAMS['volatility_multiplier']}, 趋势×{trend_multiplier} "
        f"→ 投入=${final_amount:,.2f}, 数量={contract_size:.6f} BTC"
    )
    
    return contract_size

def calculate_market_volatility(price_data):
    """计算市场波动率"""
    if 'technical_data' not in price_data:
        return 0.02  # 默认2%波动率
    
    # 使用ATR或价格标准差计算波动率
    # 这里简化处理，使用价格变化率的绝对值
    volatility = abs(price_data.get('price_change', 0)) / 100
    
    # 确保波动率在合理范围内
    return max(0.005, min(volatility, 0.1))

def check_risk_limits():
    """检查风险限制"""
    # 获取账户信息
    try:
        balance = exchange.fetch_balance()
        total_balance = balance['total'].get('USDT', 0)
        
        # 检查单日最大亏损
        daily_loss_limit = total_balance * 0.05  # 5%
        
        # TODO: 实现实际亏损跟踪
        
        return True, total_balance
        
    except Exception as e:
        print(f"风险检查失败: {e}")
        return False, 0


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            vprint(f"JSON解析失败，原始内容: {json_str}")
            vprint(f"错误详情: {e}")
            return None


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    # 如果 price_data 为空，使用默认价格
    if not price_data:
        default_price = 100000  # 默认BTC价格
        return {
            "signal": "HOLD",
            "reason": "数据获取失败，采取保守策略",
            "stop_loss": default_price * 0.98,  # -2%
            "take_profit": default_price * 1.02,  # +2%
            "confidence": "LOW",
            "is_fallback": True
        }
    
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": float(price_data['price']) * 0.98,  # -2%
        "take_profit": float(price_data['price']) * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }


def analyze_with_deepseek(ohlcv_data, indicators, current_price, current_pos=None):
    """使用DeepSeek API进行市场分析"""
    print("🔍 开始DeepSeek分析...")
    
    # 确保current_price是数字类型
    try:
        current_price = float(current_price)
    except (ValueError, TypeError):
        current_price = 0.0
        print("⚠️ current_price参数类型错误，使用默认值0.0")
    
    try:
        # 生成技术分析文本 - 需要构建price_data格式的参数
        price_data_for_analysis = {
            'price': current_price,
            'technical_data': indicators,
            'trend_analysis': {},
            'levels_analysis': {}
        }
        analysis_text = generate_technical_analysis_text(price_data_for_analysis)
        print("📊 技术分析文本生成完成")
        
        # 构建K线数据文本（兼容DataFrame与列表格式）
        kline_text = ""
        try:
            if isinstance(ohlcv_data, pd.DataFrame):
                # 期望列: timestamp(open/high/low/close/volume)
                df_tail = ohlcv_data.tail(10)
                for _, row in df_tail.iterrows():
                    ts = row.get('timestamp')
                    # 处理不同类型的时间戳
                    try:
                        if isinstance(ts, pd.Timestamp):
                            ts_str = ts.strftime('%Y-%m-%d %H:%M')
                        elif isinstance(ts, (int, float)):
                            ts_str = datetime.fromtimestamp(float(ts)/1000).strftime('%Y-%m-%d %H:%M')
                        else:
                            # 字符串或其他类型
                            ts_val = float(ts)
                            ts_str = datetime.fromtimestamp(ts_val/1000).strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        ts_str = str(ts)
                    kline_text += (
                        f"{ts_str}: 开={row.get('open')}, 高={row.get('high')}, 低={row.get('low')}, "
                        f"收={row.get('close')}, 量={row.get('volume')}\n"
                    )
            else:
                # 退化为原始列表格式 [ts_ms, open, high, low, close, volume]
                for candle in ohlcv_data[-10:]:  # 显示最近10根K线
                    try:
                        ts_ms = float(candle[0])
                        ts_str = datetime.fromtimestamp(ts_ms/1000).strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        ts_str = str(candle[0])
                    kline_text += f"{ts_str}: 开={candle[1]}, 高={candle[2]}, 低={candle[3]}, 收={candle[4]}, 量={candle[5]}\n"
        except Exception as e:
            print(f"⚠️ 构建K线数据文本失败: {e}")
            kline_text = "(K线数据格式错误)"
        
        print("📈 构建K线数据文本...")
        
        # 构建持仓信息
        position_text = "无持仓"
        pnl_text = ""
        if current_pos and current_pos.get('size', 0) > 0:
            position_text = f"当前持仓: {current_pos['size']} BTC, 成本价: {current_pos['entry_price']}"
            pnl_text = f", 未实现盈亏: {current_pos['unrealized_pnl']}"
        elif current_pos is None:
            position_text = "持仓查询失败"
        
        print("📊 持仓信息构建完成")
        
        # 构建完整的提示词
        prompt = f"""
作为专业的加密货币量化交易分析师，请分析以下市场数据并提供交易建议：

当前价格: {current_price} USDT

技术指标分析:
{analysis_text}

最近K线数据:
{kline_text}

持仓状态: {position_text}{pnl_text}

请提供:
1. 交易信号 (BUY/SELL/HOLD)
2. 信心级别 (HIGH/MEDIUM/LOW) 
3. 详细的分析理由
4. 建议的止损价格
5. 建议的止盈价格

请用JSON格式返回，包含以下字段:
signal, confidence, reason, stop_loss, take_profit
"""
        
        print("🔄 正在调用DeepSeek API...")
        vprint(f"📋 提示词长度: {len(prompt)} 字符")
        
        # 调用DeepSeek API
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的加密货币量化交易分析师，擅长技术分析和风险管理。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            print("✅ DeepSeek API调用成功")
            log_api_call('deepseek', True, 'API调用成功')
            
            # 解析响应
            content = response.choices[0].message.content
            vprint(f"📄 DeepSeek API响应内容: {content[:200]}...")  # 显示前200个字符
            
        except Exception as api_error:
            print(f"❌ DeepSeek API调用失败: {api_error}")
            raise api_error
        
        # 提取JSON部分
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            vprint(f"📋 提取到的JSON内容: {json_str[:100]}...")
            
            # 解析JSON
            try:
                analysis_result = safe_json_parse(json_str)
                if analysis_result:
                    vprint("✅ JSON解析成功")
                    
                    # 确保所有数值字段都是正确的类型
                    signal = analysis_result.get('signal', 'HOLD')
                    confidence = analysis_result.get('confidence', 'MEDIUM')
                    reason = analysis_result.get('reason', '')
                    
                    # 处理止损止盈
                    stop_loss = analysis_result.get('stop_loss')
                    take_profit = analysis_result.get('take_profit')
                    
                    # 调试信息：检查原始值类型
                    vprint(f"🔍 原始 stop_loss 类型: {type(stop_loss)}, 值: {stop_loss}")
                    vprint(f"🔍 原始 take_profit 类型: {type(take_profit)}, 值: {take_profit}")
                    
                    # 将可能包含千位分隔符或空格的数字安全转换为float
                    def to_float_safe(x):
                        if x is None:
                            return None
                        if isinstance(x, (int, float)):
                            return float(x)
                        if isinstance(x, str):
                            s = x.strip().replace(',', '')
                            # 清除非数字字符（保留负号、小数点、科学计数法）
                            s = re.sub(r"[^0-9eE+\-.]", "", s)
                            try:
                                return float(s)
                            except Exception:
                                return None
                        return None

                    stop_loss = to_float_safe(stop_loss)
                    take_profit = to_float_safe(take_profit)
                    
                    # 如果缺少止损止盈，使用默认值
                    if stop_loss is None:
                        stop_loss = float(current_price) * 0.98
                    if take_profit is None:
                        take_profit = float(current_price) * 1.03
                    
                    # 构建最终结果
                    result = {
                        "signal": signal,
                        "confidence": confidence,
                        "reason": reason,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    }
                    
                    # 验证必需字段
                    required_fields = ['signal', 'confidence', 'reason']
                    for field in required_fields:
                        if field not in result:
                            result[field] = "未知"
                    
                    # 记录交易信号
                    price_data = {
                        'price': current_price,
                        'price_change': None  # 可以添加价格变化计算
                    }
                    log_trade_signal(result, price_data)
                    
                    return result
                    
            except Exception as e:
                error_msg = f"JSON解析错误: {e}"
                print(error_msg)
                vprint(f"原始响应: {content}")
                log_error('JSON_PARSE_ERROR', error_msg, content)
                
                # 返回默认的HOLD信号
                return {
                    "signal": "HOLD",
                    "confidence": "LOW", 
                    "reason": "API响应解析失败",
                    "stop_loss": float(current_price) * 0.98,
                    "take_profit": float(current_price) * 1.03
                }
        else:
            error_msg = "未找到JSON格式的响应"
            print(error_msg)
            vprint(f"原始响应: {content}")
            log_error('NO_JSON_RESPONSE', error_msg, content)
            
            return {
                "signal": "HOLD",
                "confidence": "LOW",
                "reason": "API响应格式错误",
                "stop_loss": float(current_price) * 0.98,
                "take_profit": float(current_price) * 1.03
            }
            
    except Exception as e:
        error_msg = f"DeepSeek分析过程中发生错误: {e}"
        print(error_msg)
        vprint(f"Debug: current_price type: {type(current_price)}, value: {current_price}")
        log_error('DEEPSEEK_ANALYSIS_ERROR', error_msg, str(e))
        
        # 确保current_price是数字类型
        try:
            current_price_float = float(current_price)
        except (ValueError, TypeError):
            current_price_float = 10000.0  # 默认价格
            print("⚠️ current_price转换失败，使用默认值10000.0")
        
        return {
            "signal": "HOLD",
            "confidence": "LOW",
            "reason": f"分析过程错误: {str(e)}",
            "stop_loss": float(current_price_float) * 0.98,
            "take_profit": float(current_price_float) * 1.03
        }


def execute_trade(signal_data, price_data):
    """执行交易 - OKX版本（修复保证金检查）"""
    global position

    current_position = get_current_position()

    # 🔴 紧急修复：防止频繁反转
    if current_position and signal_data['signal'] != 'HOLD':
        current_side = current_position['side']
        # 修正：正确处理HOLD情况
        if signal_data['signal'] == 'BUY':
            new_side = 'long'
        elif signal_data['signal'] == 'SELL':
            new_side = 'short'
        else:  # HOLD
            new_side = None

        # 如果只是方向反转，需要高信心才执行
        if new_side != current_side:
            if signal_data['confidence'] != 'HIGH':
                print(f"🔒 非高信心反转信号，保持现有{current_side}仓")
                return

            # 检查最近信号历史，避免频繁反转
            if len(signal_history) >= 2:
                last_signals = [s['signal'] for s in signal_history[-2:]]
                if signal_data['signal'] in last_signals:
                    print(f"🔒 近期已出现{signal_data['signal']}信号，避免频繁反转")
                    return

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"理由: {signal_data['reason']}")
    print(f"止损: ${signal_data['stop_loss']:,.2f}")
    print(f"止盈: ${signal_data['take_profit']:,.2f}")
    print(format_position_text(current_position))

    # 记录交易执行
    trade_log = {
        'signal': signal_data['signal'],
        'confidence': signal_data['confidence'],
        'price': price_data['price'],
        'stop_loss': signal_data['stop_loss'],
        'take_profit': signal_data['take_profit'],
        'timestamp': datetime.now().isoformat()
    }
    if TRADE_CONFIG.get('verbose', False):
        trading_logger.info(f"交易执行: {json.dumps(trade_log, ensure_ascii=False)}")

    # 风险管理：低信心信号不执行
    if signal_data['confidence'] == 'LOW' and not TRADE_CONFIG['test_mode']:
        print("⚠️ 低信心信号，跳过执行")
        log_error('LOW_CONFIDENCE_SKIP', '低信心信号，跳过执行')
        return

    if TRADE_CONFIG['test_mode']:
        print("测试模式 - 仅模拟交易")
        return

    try:
        # 获取账户余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        
        # 确保price是数字类型
        try:
            price_float = float(price_data['price'])
        except (ValueError, TypeError):
            price_float = 10000.0  # 默认价格
            print("⚠️ price_data['price']转换失败，使用默认值10000.0")
        
        # 动态仓位计算，避免因保证金不足而跳过交易
        try:
            sizing_price_data = {
                'price': price_float,
                'price_change': price_data.get('price_change', 0)
            }
            dynamic_amount = calculate_position_size(sizing_price_data, signal_data, usdt_balance)
        except Exception as size_err:
            print(f"⚠️ 动态仓位计算失败，使用配置量: {size_err}")
            dynamic_amount = TRADE_CONFIG['amount']

        # 市场限制的最小下单量
        min_amount = 0.001
        try:
            market = exchange.market(TRADE_CONFIG['symbol'])
            limits = market.get('limits', {})
            if limits and 'amount' in limits and limits['amount'].get('min'):
                min_amount = float(limits['amount']['min'])
        except Exception:
            pass

        # 计算可承受的最大数量（占用最多80%余额作为保证金）
        max_affordable_amount = (usdt_balance * 0.8 * TRADE_CONFIG['leverage']) / price_float
        order_amount = max(min(dynamic_amount, max_affordable_amount), min_amount)

        # 若为HOLD信号，直接退出，不进行下单与保证金计算
        if signal_data['signal'] not in ('BUY', 'SELL'):
            print("⏸️ HOLD信号，暂不下单。将于下一周期重新评估。")
            return

        required_margin = price_float * order_amount / TRADE_CONFIG['leverage']

        # 风险管理检查
        risk_manager = get_risk_manager()
        risk_results = risk_manager.update_balance(usdt_balance)
        
        # 检查硬性止损
        if risk_results['hard_stop_triggered']:
            for msg in risk_results['messages']:
                print(f"🚨 {msg}")
            log_error('HARD_STOP_LOSS', '硬性止损触发，停止所有交易')
            return
            
        # 检查单日亏损限制
        if risk_results['daily_limit_triggered']:
            for msg in risk_results['messages']:
                print(f"⚠️ {msg}")
            log_error('DAILY_LOSS_LIMIT', '单日亏损限制触发，暂停交易')
            return

        if required_margin > usdt_balance * 0.8:  # 使用不超过80%的余额
            # 再次收敛数量以满足保证金限制
            order_amount = max_affordable_amount
            required_margin = price_float * order_amount / TRADE_CONFIG['leverage']
            print(f"🔧 已调整数量以控制保证金占用: {order_amount:.6f} BTC，保证金≈{required_margin:.2f} USDT")

        # 记录最终下单信息
        print(f"📌 最终下单数量: {order_amount:.6f} BTC (价格≈{price_float:.2f}, 杠杆×{TRADE_CONFIG['leverage']})")

        # 执行交易逻辑   tag 是我的经纪商api（不拿白不拿），不会影响大家返佣，介意可以删除
        order_executed = False
        order_response = None
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("平空仓并开多仓...")
                # 平空仓
                try:
                    exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE'}
                    )
                    order_executed = True
                except Exception as close_err:
                    print(f"❌ 平空仓失败: {close_err}")
                time.sleep(1)
                # 开多仓
                order_response = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    order_amount,
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
                order_executed = True
            elif current_position and current_position['side'] == 'long':
                print("已有多头持仓，保持现状")
            else:
                # 无持仓时开多仓
                print("开多仓...")
                order_response = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    order_amount,
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
                order_executed = True

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("平多仓并开空仓...")
                # 平多仓
                try:
                    exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE'}
                    )
                    order_executed = True
                except Exception as close_err:
                    print(f"❌ 平多仓失败: {close_err}")
                time.sleep(1)
                # 开空仓
                order_response = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    order_amount,
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
                order_executed = True
            elif current_position and current_position['side'] == 'short':
                print("已有空头持仓，保持现状")
            else:
                # 无持仓时开空仓
                print("开空仓...")
                order_response = exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    order_amount,
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
                order_executed = True

        if order_executed:
            print("✅ 订单执行成功")
            if order_response:
                try:
                    # 友好摘要输出（存在字段缺失时自动忽略）
                    summary = {
                        'id': order_response.get('id'),
                        'status': order_response.get('status'),
                        'side': order_response.get('side'),
                        'amount': order_response.get('amount'),
                        'filled': order_response.get('filled'),
                        'average': order_response.get('average'),
                        'cost': order_response.get('cost'),
                    }
                    summary_str = ", ".join(
                        [f"{k}={v}" for k, v in summary.items() if v is not None]
                    )
                    print(f"🧾 订单摘要: {summary_str}")
                except Exception:
                    pass
        else:
            print("ℹ️ 未执行任何订单")
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")
        
        # 利润提取检查
        risk_manager = get_risk_manager()
        risk_results = risk_manager.update_balance(usdt_balance)
        
        if risk_results['profit_extraction']:
            for msg in risk_results['messages']:
                print(f"💰 {msg}")
            # 在实际交易中，这里可以添加提取利润到安全钱包的逻辑
            print("📤 利润提取功能已触发（模拟）")

    except Exception as e:
        error_msg = f"订单执行失败: {e}"
        print(error_msg)
        log_error('TRADE_EXECUTION_ERROR', error_msg, str(e))
        import traceback
        traceback.print_exc()


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            # 准备分析所需的参数
            ohlcv_data = price_data.get('full_data', [])
            indicators = price_data.get('technical_data', {})
            current_price = price_data.get('price', 0)
            current_pos = get_current_position()
            
            signal_data = analyze_with_deepseek(ohlcv_data, indicators, current_price, current_pos)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            print(f"第{attempt + 1}次尝试失败，进行重试...")
            time.sleep(1)

        except Exception as e:
            print(f"第{attempt + 1}次尝试异常: {e}")
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)


def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（00, 15, 30, 45分钟）
    next_period_minute = ((current_minute // 15) + 1) * 15
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return seconds_to_wait


def trading_bot():
    """主交易循环"""
    print(f"🚀 启动交易机器人 - 交易对: {TRADE_CONFIG['symbol']}, 时间周期: {TRADE_CONFIG['timeframe']}")
    print(f"📊 测试模式: {'开启' if TRADE_CONFIG['test_mode'] else '关闭'}")
    
    # 记录启动信息
    startup_log = {
        'timestamp': datetime.now().isoformat(),
        'symbol': TRADE_CONFIG['symbol'],
        'timeframe': TRADE_CONFIG['timeframe'],
        'test_mode': TRADE_CONFIG['test_mode'],
        'status': 'started'
    }
    if TRADE_CONFIG.get('verbose', False):
        trading_logger.info(f"交易机器人启动: {json.dumps(startup_log, ensure_ascii=False)}")
    
    while True:
        try:
            print(f"\n⏰ 开始新一轮分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 获取市场数据
            print("📡 获取市场数据...")
            price_data = get_btc_ohlcv_enhanced()
            
            if not price_data:
                print("⚠️ 获取市场数据失败，等待下一轮")
                log_error('MARKET_DATA_FAILURE', '获取市场数据失败')
                wait_for_next_period()
                continue
            
            # 市场异常波动监控
            risk_manager = get_risk_manager()
            volatility_alert = risk_manager.monitor_volatility(price_data, price_data['price'])
            if volatility_alert:
                print("⚠️ 检测到市场异常波动，建议谨慎操作")
                log_error('HIGH_VOLATILITY_ALERT', '市场异常波动检测')
            
            # 使用DeepSeek分析
            print("🤖 使用DeepSeek进行市场分析...")
            signal_data = analyze_with_deepseek_with_retry(price_data)
            
            # 执行交易
            execute_trade(signal_data, price_data)
            
            # 记录循环完成
            cycle_log = {
                'timestamp': datetime.now().isoformat(),
                'price': price_data['price'],
                'signal': signal_data['signal'],
                'confidence': signal_data['confidence'],
                'status': 'completed'
            }
            if TRADE_CONFIG.get('verbose', False):
                trading_logger.info(f"交易循环完成: {json.dumps(cycle_log, ensure_ascii=False)}")
            
            # 等待下一周期
            wait_seconds = wait_for_next_period()
            try:
                if isinstance(wait_seconds, (int, float)) and wait_seconds > 0:
                    time.sleep(wait_seconds)
            except Exception as sleep_err:
                print(f"⚠️ 等待下一周期休眠失败: {sleep_err}")
            
        except KeyboardInterrupt:
            print("\n🛑 用户中断程序")
            shutdown_log = {
                'timestamp': datetime.now().isoformat(),
                'status': 'shutdown',
                'reason': 'user_interrupt'
            }
            if TRADE_CONFIG.get('verbose', False):
                trading_logger.info(f"交易机器人关闭: {json.dumps(shutdown_log, ensure_ascii=False)}")
            break
        except Exception as e:
            error_msg = f"交易循环发生错误: {e}"
            print(f"❌ {error_msg}")
            log_error('TRADING_LOOP_ERROR', error_msg, str(e))
            import traceback
            traceback.print_exc()
            print("⏸️  等待30秒后重试...")
            try:
                time.sleep(30)
            except Exception:
                pass
            time.sleep(30)


def main():
    """主函数"""
    print("BTC/USDT OKX自动交易机器人启动成功！")
    print("融合技术指标策略 + OKX实盘接口")

    if TRADE_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"交易周期: {TRADE_CONFIG['timeframe']}")
    print("已启用完整技术指标分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    if TRADE_CONFIG['test_mode']:
        print("测试模式：立即执行一次分析演示")
        # 测试模式下立即执行一次分析
        try:
            print("\n" + "="*50)
            print("开始执行市场分析...")
            
            # 获取市场数据
            price_data = get_btc_ohlcv_enhanced()
            if price_data:
                print(f"当前BTC价格: ${price_data['price']:,.2f}")
                print(f"24小时涨跌: {price_data['price_change']:+.2f}%")
                
                # 执行AI分析
                signal_data = analyze_with_deepseek_with_retry(price_data)
                if signal_data:
                    print(f"\nAI分析结果:")
                    print(f"交易信号: {signal_data['signal']}")
                    print(f"信心程度: {signal_data['confidence']}")
                    print(f"分析理由: {signal_data['reason']}")
                    
                    # 执行交易（测试模式下只是打印）
                    execute_trade(signal_data, price_data)
                else:
                    print("AI分析失败")
            else:
                print("获取市场数据失败")
                
            print("="*50)
            print("测试演示完成！")
            return
            
        except Exception as e:
            print(f"测试执行出错: {e}")
            import traceback
            traceback.print_exc()
            return

    print("执行频率: 每15分钟整点执行")

    # 循环执行（不使用schedule）
    while True:
        trading_bot()  # 函数内部会自己等待整点

        # 执行完后等待一段时间再检查（避免频繁循环）
        time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    main()