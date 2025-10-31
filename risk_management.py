"""
风险管理系统 - 硬止损、利润提取和波动监控
"""

import json
import time
from datetime import datetime, timedelta
import logging

# 配置日志
logger = logging.getLogger('risk_management')

class RiskManager:
    def __init__(self, initial_balance=100.0):
        # 风险参数
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_loss_limit = 0.20  # 单日最大亏损20%
        self.hard_stop_loss = 0.30   # 硬性止损30%
        self.profit_take_ratio = 0.50  # 提取50%利润
        
        # 交易记录
        self.trade_history = []
        self.daily_pnl = 0.0
        self.last_profit_extraction = datetime.now()
        
        # 波动监控
        self.volatility_threshold = 0.08  # 8%异常波动阈值
        self.last_volatility_alert = None
        
        print(f"🔒 风险管理系统初始化 - 初始资金: ${initial_balance:.2f}")
        print(f"   - 单日止损: {self.daily_loss_limit*100:.0f}% (${initial_balance*self.daily_loss_limit:.2f})")
        print(f"   - 硬性止损: {self.hard_stop_loss*100:.0f}% (${initial_balance*self.hard_stop_loss:.2f})")
        print(f"   - 利润提取: {self.profit_take_ratio*100:.0f}%")
    
    def update_balance(self, new_balance):
        """更新账户余额并检查风险限制"""
        old_balance = self.current_balance
        self.current_balance = new_balance
        
        # 更新峰值余额
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        # 计算当日盈亏
        pnl_change = new_balance - old_balance
        self.daily_pnl += pnl_change
        
        return self.check_risk_limits(pnl_change)
    
    def check_risk_limits(self, pnl_change):
        """检查所有风险限制"""
        results = {
            'hard_stop_triggered': False,
            'daily_limit_triggered': False,
            'profit_extraction': False,
            'messages': []
        }
        
        # 1. 硬性止损检查
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        if drawdown >= self.hard_stop_loss:
            results['hard_stop_triggered'] = True
            results['messages'].append(f"🚨 硬性止损触发! 亏损: {drawdown*100:.1f}% (${self.initial_balance - self.current_balance:.2f})")
        
        # 2. 单日亏损限制
        daily_drawdown = -self.daily_pnl / self.initial_balance
        if daily_drawdown >= self.daily_loss_limit:
            results['daily_limit_triggered'] = True
            results['messages'].append(f"⚠️ 单日亏损限制! 今日亏损: {daily_drawdown*100:.1f}% (${-self.daily_pnl:.2f})")
        
        # 3. 利润提取检查
        if self.current_balance > self.peak_balance * 1.10:  # 盈利超过10%
            time_since_last_extraction = (datetime.now() - self.last_profit_extraction).total_seconds() / 3600
            if time_since_last_extraction >= 24:  # 至少24小时提取一次
                results['profit_extraction'] = True
                profit = self.current_balance - self.initial_balance
                extract_amount = profit * self.profit_take_ratio
                results['messages'].append(f"💰 利润提取: ${extract_amount:.2f} (总盈利: ${profit:.2f})")
                self.last_profit_extraction = datetime.now()
        
        # 记录交易
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'balance': self.current_balance,
            'pnl_change': pnl_change,
            'daily_pnl': self.daily_pnl,
            'drawdown': drawdown,
            'risk_results': results
        }
        self.trade_history.append(trade_record)
        
        # 保持最近100条记录
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
        
        return results
    
    def monitor_volatility(self, price_data, current_price):
        """监控市场异常波动"""
        if 'technical_data' not in price_data:
            return False
        
        volatility = abs(price_data.get('price_change', 0)) / 100
        
        # 异常波动检测
        if volatility >= self.volatility_threshold:
            current_time = datetime.now()
            
            # 避免频繁警报（至少间隔1小时）
            if self.last_volatility_alert is None or \
               (current_time - self.last_volatility_alert).total_seconds() >= 3600:
                
                self.last_volatility_alert = current_time
                
                alert_msg = f"🌪️ 异常波动警报! 波动率: {volatility*100:.1f}%"
                if 'high' in price_data and 'low' in price_data:
                    price_range = price_data['high'] - price_data['low']
                    alert_msg += f", 价格区间: ${price_range:.2f}"
                
                print(f"🚨 {alert_msg}")
                
                # 记录到日志
                log_data = {
                    'timestamp': current_time.isoformat(),
                    'volatility': volatility,
                    'current_price': current_price,
                    'message': alert_msg
                }
                logger.warning(json.dumps(log_data))
                
                return True
        
        return False
    
    def get_risk_status(self):
        """获取当前风险状态"""
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        daily_drawdown = -self.daily_pnl / self.initial_balance if self.daily_pnl < 0 else 0
        
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'total_profit': self.current_balance - self.initial_balance,
            'drawdown': drawdown,
            'daily_drawdown': daily_drawdown,
            'hard_stop_remaining': max(0, self.hard_stop_loss - drawdown),
            'daily_limit_remaining': max(0, self.daily_loss_limit - daily_drawdown),
            'trade_count': len(self.trade_history)
        }
    
    def reset_daily_pnl(self):
        """重置每日盈亏（在交易日开始时调用）"""
        self.daily_pnl = 0.0
        print("📊 每日盈亏计数器已重置")
    
    def export_report(self):
        """生成风险报告"""
        status = self.get_risk_status()
        
        report = f"""
📊 风险管理报告
====================
初始资金: ${self.initial_balance:.2f}
当前资金: ${status['current_balance']:.2f}
峰值资金: ${status['peak_balance']:.2f}
累计盈亏: ${status['total_profit']:+.2f}

🔒 风险限制:
- 硬性止损剩余: {status['hard_stop_remaining']*100:.1f}%
- 单日止损剩余: {status['daily_limit_remaining']*100:.1f}%
- 当前回撤: {status['drawdown']*100:.1f}%
- 当日回撤: {status['daily_drawdown']*100:.1f}%

📈 交易统计:
- 总交易次数: {status['trade_count']}
- 最后提取时间: {self.last_profit_extraction.strftime('%Y-%m-%d %H:%M')}
"""
        
        return report

# 全局风险管理器实例
risk_manager = None

def initialize_risk_manager(initial_balance=100.0):
    """初始化风险管理器"""
    global risk_manager
    risk_manager = RiskManager(initial_balance)
    return risk_manager

def get_risk_manager():
    """获取风险管理器实例"""
    global risk_manager
    if risk_manager is None:
        risk_manager = RiskManager(100.0)  # 默认100 USDT
    return risk_manager