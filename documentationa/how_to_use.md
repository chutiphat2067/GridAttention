# คู่มือการใช้งาน Grid Trading System with Progressive Attention

## สารบัญ
1. [ภาพรวมระบบ](#1-ภาพรวมระบบ)
2. [การติดตั้งและตั้งค่า](#2-การติดตั้งและตั้งค่า)
3. [การเริ่มต้นใช้งาน](#3-การเริ่มต้นใช้งาน)
4. [การใช้งานระบบ](#4-การใช้งานระบบ)
5. [การตรวจสอบและ Monitoring](#5-การตรวจสอบและ-monitoring)
6. [การแก้ไขปัญหา](#6-การแก้ไขปัญหา)
7. [ความปลอดภัยและ Risk Management](#7-ความปลอดภัยและ-risk-management)
8. [การบำรุงรักษา](#8-การบำรุงรักษา)
9. [คำเตือนและข้อควรระวัง](#9-คำเตือนและข้อควรระวัง)
10. [ภาคผนวก](#10-ภาคผนวก)

---

## 1. ภาพรวมระบบ

### 1.1 Grid Trading System คืออะไร?
Grid Trading System เป็นระบบเทรดอัตโนมัติที่วางคำสั่งซื้อ-ขายเป็นตารางกริด (Grid) รอบๆ ราคาปัจจุบัน โดยมีเป้าหมายทำกำไรจากความผันผวนของราคา

### 1.2 Progressive Attention Architecture
ระบบนี้มีความพิเศษที่ใช้ AI แบบ Progressive Learning:
- **Learning Phase** (1,000+ trades): สังเกตและเรียนรู้พฤติกรรมตลาด
- **Shadow Phase** (200+ trades): คำนวณแต่ยังไม่ใช้งานจริง  
- **Active Phase**: ใช้ AI เพื่อปรับปรุงการเทรด

### 1.3 คุณสมบัติหลัก
- ✅ เทรดอัตโนมัติ 24/7
- ✅ ปรับตัวตามสภาพตลาด (4 โหมด: Ranging, Trending, Volatile, Dormant)
- ✅ Risk Management แบบ Conservative
- ✅ Fee Optimization
- ✅ ประมวลผลเร็ว < 5ms

---

## 2. การติดตั้งและตั้งค่า

### 2.1 ความต้องการของระบบ

#### Hardware Requirements
```
- CPU: 4 cores ขึ้นไป
- RAM: 8GB ขึ้นไป (แนะนำ 16GB)
- Storage: 100GB SSD
- Network: Internet ความเร็วสูง (< 50ms latency)
```

#### Software Requirements
```
- Python 3.11+
- PostgreSQL 14+ (หรือ TimescaleDB)
- Redis 7+
- Docker (optional)
```

### 2.2 ขั้นตอนการติดตั้ง

#### Step 1: Clone Repository
```bash
git clone https://github.com/your-repo/grid-trading-system.git
cd grid-trading-system
```

#### Step 2: สร้าง Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# หรือ
venv\Scripts\activate     # Windows
```

#### Step 3: ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # สำหรับ development
```

#### Step 4: ตั้งค่า Database
```bash
# PostgreSQL
createdb grid_trading
psql grid_trading < schema/database.sql

# Redis
redis-server --daemonize yes
```

#### Step 5: ตั้งค่า Configuration
```bash
cp config/config.example.yaml config/config.yaml
# แก้ไขไฟล์ config.yaml ตามความต้องการ
```

### 2.3 การตั้งค่า Exchange API

#### Binance Example
```yaml
# config/config.yaml
exchanges:
  binance:
    api_key: "YOUR_API_KEY"
    secret: "YOUR_SECRET_KEY"
    testnet: true  # เริ่มด้วย testnet ก่อน!
```

⚠️ **คำเตือน**: ตั้งค่า API permissions แบบ "Trade" only ห้ามเปิด "Withdraw"

---

## 3. การเริ่มต้นใช้งาน

### 3.1 Paper Trading Mode (แนะนำ!)

#### สร้างไฟล์ main.py
```python
import asyncio
from grid_trading_system import GridTradingSystem

async def main():
    # เริ่มด้วย Paper Trading
    config = {
        'mode': 'paper_trading',
        'initial_capital': 10000,  # USD
        'max_position_size': 0.05,  # 5% ต่อ position
        'max_daily_loss': 0.01,     # 1% daily loss limit
    }
    
    system = GridTradingSystem('config/config.yaml')
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())
```

#### รันระบบ
```bash
python main.py
```

### 3.2 การตรวจสอบการทำงาน

#### Check System Status
```bash
# ดู logs
tail -f logs/grid_trading.log

# ดู metrics
curl http://localhost:9090/metrics

# เปิด dashboard
open http://localhost:8080
```

### 3.3 Checklist ก่อนใช้เงินจริง

- [ ] Paper trade อย่างน้อย 4 สัปดาห์
- [ ] ผ่านทุก market conditions
- [ ] Win rate > 50%
- [ ] Max drawdown < 5%
- [ ] ไม่มี critical errors
- [ ] Attention system เข้าสู่ Active phase
- [ ] ทดสอบ emergency shutdown
- [ ] มี backup และ recovery plan

---

## 4. การใช้งานระบบ

### 4.1 Grid Strategy Parameters

#### Conservative Settings (แนะนำสำหรับมือใหม่)
```python
BEGINNER_CONFIG = {
    'grid_type': 'symmetric',
    'grid_levels': 3,           # จำนวนระดับน้อย
    'spacing': 0.002,           # 0.2% ระยะห่าง
    'position_size': 0.03,      # 3% ของ capital
    'take_profit': 0.002,       # 0.2%
    'stop_loss': 0.004,         # 0.4%
}
```

#### Standard Settings
```python
STANDARD_CONFIG = {
    'grid_type': 'symmetric',
    'grid_levels': 5,
    'spacing': 0.0015,          # 0.15%
    'position_size': 0.05,      # 5%
    'take_profit': 0.0015,      # 0.15%
    'stop_loss': 0.003,         # 0.3%
}
```

### 4.2 Market Regime Settings

ระบบจะปรับ strategy อัตโนมัติตาม market regime:

| Regime | Characteristics | Strategy Adjustments |
|--------|----------------|---------------------|
| **RANGING** | ราคาเคลื่อนที่ในกรอบ | Grid แคบ, Levels มาก |
| **TRENDING** | ราคามีทิศทางชัดเจน | Grid กว้าง, Asymmetric |
| **VOLATILE** | ความผันผวนสูง | Grid กว้างมาก, Levels น้อย |
| **DORMANT** | ตลาดเงียบ | ปิดการเทรด หรือ Grid แคบมาก |

### 4.3 คำสั่งควบคุมระบบ

#### Start/Stop Commands
```python
# Start trading
await system.start()

# Pause trading (keep positions)
await system.pause()

# Stop trading (close all positions)
await system.stop()

# Emergency shutdown
await system.emergency_shutdown("Manual intervention")
```

#### Position Management
```python
# ดู positions ปัจจุบัน
positions = await system.get_open_positions()

# ปิด position specific
await system.close_position("BTCUSDT")

# ปิดทุก positions
await system.close_all_positions()
```

---

## 5. การตรวจสอบและ Monitoring

### 5.1 Dashboard Overview

Dashboard แสดงข้อมูลสำคัญ:
- **P&L Chart**: กำไร/ขาดทุนแบบ real-time
- **Win Rate**: อัตราการชนะ
- **Active Positions**: positions ที่เปิดอยู่
- **System Health**: CPU, Memory, Latency
- **Attention Progress**: สถานะการเรียนรู้ของ AI

### 5.2 Key Metrics ที่ต้องดู

#### Trading Metrics
```
- Win Rate: ควร > 50%
- Profit Factor: ควร > 1.2
- Sharpe Ratio: ควร > 1.5
- Max Drawdown: ไม่ควรเกิน 5%
- Grid Fill Rate: ควร > 60%
```

#### System Metrics
```
- Execution Latency: ต้อง < 5ms (p99)
- Error Rate: ต้อง < 1%
- Memory Usage: ไม่ควรเกิน 80%
- API Rate Limit: ใช้ไม่เกิน 80%
```

### 5.3 Alerts Configuration

```yaml
# config/alerts.yaml
alerts:
  critical:
    - max_drawdown: 0.03      # 3%
    - consecutive_losses: 5
    - error_rate: 0.05        # 5%
    
  warning:
    - daily_loss: 0.01        # 1%
    - high_latency: 10        # 10ms
    - low_liquidity: true
    
  notification_channels:
    - email: your@email.com
    - telegram: @your_bot
    - webhook: https://your-webhook.com
```

---

## 6. การแก้ไขปัญหา

### 6.1 ปัญหาที่พบบ่อย

#### 1. High Latency (> 5ms)
```bash
# ตรวจสอบ
- Network latency ไปยัง exchange
- CPU usage
- จำนวน concurrent orders

# แก้ไข
- ลด grid levels
- ใช้ VPS ใกล้ exchange servers
- Optimize feature calculations
```

#### 2. Low Win Rate (< 45%)
```bash
# ตรวจสอบ
- Market regime detection accuracy
- Grid spacing settings
- Slippage

# แก้ไข
- เพิ่ม grid spacing
- ลด position size
- Review regime detection
```

#### 3. Attention Not Activating
```bash
# ตรวจสอบ
- จำนวน trades ที่ผ่านมา
- Data quality
- Learning metrics

# แก้ไข
- ต้องรอให้ครบ 1000+ trades
- Check data validation errors
- Review attention thresholds
```

### 6.2 Debug Mode

เปิด debug mode เพื่อดูรายละเอียด:
```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Enable specific component debugging
system.enable_debug(['attention', 'execution', 'risk'])
```

### 6.3 Recovery Procedures

#### After System Crash
```bash
1. ตรวจสอบ open positions
2. Verify account balance
3. Check order sync
4. Review error logs
5. Start in reduced mode
6. Monitor closely for 1 hour
```

---

## 7. ความปลอดภัยและ Risk Management

### 7.1 Risk Limits (ไม่ควรเปลี่ยน!)

```python
RISK_LIMITS = {
    'max_position_size': 0.05,      # 5% per position
    'max_total_exposure': 0.30,     # 30% total
    'max_daily_loss': 0.01,         # 1% daily
    'max_drawdown': 0.03,           # 3% max DD
    'max_correlation': 0.7,         # Between positions
}
```

### 7.2 Circuit Breakers

ระบบจะหยุดอัตโนมัติเมื่อ:
- ขาดทุนติดต่อกัน 5 ครั้ง
- Daily loss > 1%
- Drawdown > 3%
- Error rate > 10%
- Latency > 50ms

### 7.3 Security Best Practices

1. **API Keys**
   - ใช้ IP whitelist
   - Trade permissions only
   - Rotate keys ทุก 3 เดือน

2. **Server Security**
   - ใช้ firewall
   - SSH key authentication
   - Regular security updates

3. **Monitoring**
   - 24/7 alerts
   - Anomaly detection
   - Regular audits

---

## 8. การบำรุงรักษา

### 8.1 Daily Tasks
```bash
# Morning checklist
1. Check overnight performance
2. Review error logs
3. Verify positions sync
4. Check system resources

# Evening checklist
1. Backup database
2. Archive logs
3. Review daily P&L
4. Plan next day
```

### 8.2 Weekly Tasks
```bash
1. Performance analysis
2. Strategy optimization review
3. Risk parameter review
4. System update check
5. Clean old logs/data
```

### 8.3 Monthly Tasks
```bash
1. Full system audit
2. Backtest with recent data
3. Update ML models
4. Security review
5. Cost analysis
6. Generate reports
```

### 8.4 Backup & Recovery

#### Automated Backups
```bash
# Database backup (daily)
pg_dump grid_trading > backup/db_$(date +%Y%m%d).sql

# Config backup (on change)
cp -r config/ backup/config_$(date +%Y%m%d)/

# State backup (hourly)
python scripts/backup_state.py
```

#### Recovery Process
```bash
# Restore database
psql grid_trading < backup/db_20240120.sql

# Restore config
cp -r backup/config_20240120/* config/

# Verify and restart
python scripts/verify_recovery.py
python main.py --recovery-mode
```

---

## 9. คำเตือนและข้อควรระวัง

### 🚨 คำเตือนสำคัญ

1. **ห้ามใช้เงินที่ไม่สามารถเสียได้**
2. **ต้อง Paper Trade อย่างน้อย 4 สัปดาห์**
3. **เริ่มด้วยเงินจำนวนน้อย (< $1,000)**
4. **ตรวจสอบระบบทุกวัน**
5. **มี Emergency Fund สำรอง**

### ⚠️ ข้อควรระวัง

1. **Market Conditions**
   - ระวัง Flash Crash
   - ระวัง Low Liquidity
   - ระวัง Major News Events

2. **Technical Issues**
   - Internet connection ต้องเสถียร
   - มี backup connection
   - Monitor 24/7

3. **Psychological Factors**
   - อย่าปรับ parameters ขณะขาดทุน
   - ยึด risk limits เสมอ
   - ไม่ใช้ leverage เกินไป

---

## 10. ภาคผนวก

### 10.1 Glossary

| Term | คำอธิบาย |
|------|----------|
| **Grid** | ตารางคำสั่งซื้อ-ขาย |
| **Spacing** | ระยะห่างระหว่างคำสั่ง |
| **Regime** | สภาวะตลาด |
| **Attention** | ระบบ AI ที่เรียนรู้ |
| **Circuit Breaker** | ระบบหยุดฉุกเฉิน |
| **Drawdown** | การลดลงจากจุดสูงสุด |

### 10.2 Useful Commands

```bash
# System control
python main.py --start
python main.py --stop
python main.py --status

# Monitoring
python scripts/show_metrics.py
python scripts/show_positions.py
python scripts/show_pnl.py

# Maintenance
python scripts/cleanup_logs.py
python scripts/optimize_db.py
python scripts/backup_all.py

# Emergency
python scripts/emergency_stop.py
python scripts/close_all_positions.py
python scripts/export_state.py
```

### 10.3 Support & Resources

- 📧 Email: support@gridtrading.com
- 💬 Discord: discord.gg/gridtrading
- 📚 Documentation: docs.gridtrading.com
- 🐛 Issues: github.com/gridtrading/issues

### 10.4 Legal Disclaimer

```
การซื้อขายสินทรัพย์ดิจิทัลมีความเสี่ยงสูง 
ผลการดำเนินงานในอดีตไม่ได้เป็นสิ่งยืนยันผลการดำเนินงานในอนาคต
ควรศึกษาและทำความเข้าใจก่อนตัดสินใจลงทุน
ผู้พัฒนาไม่รับผิดชอบต่อความสูญเสียใดๆ ที่เกิดขึ้น
```

---

## 🎯 Quick Start Guide

### สำหรับผู้เริ่มต้น (5 ขั้นตอน)

1. **ติดตั้งระบบ**
   ```bash
   git clone [repository]
   pip install -r requirements.txt
   ```

2. **ตั้งค่า Paper Trading**
   ```yaml
   mode: paper_trading
   capital: 10000
   ```

3. **รันระบบ**
   ```bash
   python main.py
   ```

4. **Monitor ผ่าน Dashboard**
   ```
   http://localhost:8080
   ```

5. **รอการเรียนรู้**
   - Learning: 1,000 trades
   - Shadow: 200 trades
   - Active: AI ช่วยปรับปรุง

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**License**: MIT