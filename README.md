# Hyperliquid Basis Trading Bot

An automated trading bot for executing basis trades on Hyperliquid exchange. The bot captures funding rate arbitrage opportunities by simultaneously trading spot and perpetual futures.

## 🎯 What It Does

The bot:
- **Scans** the market for positive funding rates
- **Executes** basis trades (long spot + short perp) when opportunities arise
- **Manages** positions by scaling up/down based on funding rates
- **Displays** real-time portfolio status and market pulse

## ⚠️ Important Safety Notice

**This bot trades with REAL MONEY when `DRY_RUN = False`**

- Always test with `DRY_RUN = True` first
- Start with small position sizes
- Monitor the bot closely, especially during initial runs
- The bot uses leverage (default: 3x) - understand the risks

## 📋 Prerequisites

- Python 3.8+
- Hyperliquid account with USDC deposited
- Private key for your trading account
- Basic understanding of basis trading and funding rates

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd hl-funding-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
PRIVATE_KEY=your_private_key_here
ACCOUNT_ADDRESS=your_account_address_here
```

**⚠️ Never commit your `.env` file to version control!**

### 3. Run the Bot

```bash
python basis_bot.py
```

The bot will:
- Initialize and display account information
- Show portfolio status every 30 seconds
- Display market pulse (top funding rates)
- Execute trades when opportunities are found

## ⚙️ Configuration

Edit the `Config` class in `basis_bot.py` to customize:

### Safety Settings
```python
DRY_RUN = True  # Set to False to trade REAL MONEY
```

### Trading Parameters
```python
LEVERAGE = 3                    # Leverage multiplier
LEVERAGE_MODE = "cross"         # "cross" or "isolated"
ENTRY_SIZE_PCT = 0.05          # 5% of equity per trade
MAX_POS_PCT = 0.20            # Max 20% of equity per coin
```

### Funding Thresholds
```python
MIN_ENTRY_FUNDING = 0.00002    # Minimum funding rate to enter
SCALE_UP_FUNDING = 0.00005     # Funding rate to scale up position
SCALE_DOWN_FUNDING = -0.00001  # Funding rate to scale down position
```

### Timing
```python
LOOP_INTERVAL_SEC = 30         # How often to check for opportunities
SCALE_COOLDOWN_SEC = 3600      # Cooldown between scaling operations
```

## 📊 Understanding the Output

### Portfolio Dashboard
```
======================================================================
 📊 PORTFOLIO | Equity: $1000.00 | Margin: 50.00
----------------------------------------------------------------------
 COIN   SIDE  SIZE($)    ENTRY      MARK       PnL($)   FUNDING
----------------------------------------------------------------------
 BTC    LONG  200.00     45000.00   45100.00   +2.00    +0.0123%
======================================================================
```

### Market Pulse
```
 🔍 MARKET PULSE (Top Funding Rates):
    1. BTC    | Rate: 0.0123% | BE: 11.4h
    2. ETH    | Rate: 0.0105% | BE: 13.3h
```

**BE** = Breakeven hours (time to recover trading fees)

## 🏗️ Code Structure

The code is organized into logical modules:

### 1. **Config** (Lines 39-76)
- All configuration settings
- Environment variable loading
- Validation

### 2. **Utils** (Lines 104-135)
- `round_sz()`: Formats size according to `szDecimals`
- `round_px()`: Formats price with proper significant figures
- Handles perp (6 decimals) vs spot (8 decimals) differences

### 3. **DataManager** (Lines 137-207)
- `refresh()`: Fetches latest market metadata
- `get_perp_details()`: Gets perp asset info and ID
- `get_spot_details()`: Gets spot asset info and ID (10000 + index)
- `calculate_slippage()`: Estimates slippage for orders

### 4. **TradeExecutor** (Lines 209-321)
- `place_order()`: Places orders with proper formatting
- `execute_basis_trade()`: Executes the two-legged trade:
  - Leg 1: Buy/Sell spot
  - Leg 2: Sell/Buy perp (opposite direction)
- Handles errors and reversals if perp leg fails

### 5. **PortfolioManager** (Lines 323-432)
- `print_status()`: Displays portfolio dashboard
- `rebalance()`: Scales positions based on funding rates

### 6. **MarketScanner** (Lines 434-520)
- `scan()`: Scans market for trading opportunities
- Filters by funding rate, breakeven time, and slippage

### 7. **BasisBot** (Lines 522-598)
- Main bot class that orchestrates everything
- `run()`: Main loop that runs continuously

## 🔧 How to Maintain & Extend

### Adding New Features

1. **New Trading Strategy**
   - Add logic in `MarketScanner.scan()`
   - Modify `execute_basis_trade()` if needed

2. **Different Order Types**
   - Modify `place_order()` to support GTC, ALO, etc.
   - Update TIF in order payload

3. **Additional Risk Controls**
   - Add checks in `execute_basis_trade()`
   - Implement position limits in `PortfolioManager`

### Debugging

Enable debug logging:
```python
logger.setLevel(logging.DEBUG)  # In setup_logger()
```

### Common Issues

**"No positive funding rates found"**
- Normal - means no profitable opportunities at the moment
- Market pulse will show when opportunities arise

**"Equity is $0"**
- Deposit USDC to your Hyperliquid account
- Bot will wait and check again

**Order failures**
- Check slippage settings (`SCANNER_SLIPPAGE`, `EXECUTION_SLIPPAGE`)
- Verify asset exists in both spot and perp markets
- Check API rate limits

## 📚 API Compliance

This bot follows Hyperliquid API specifications:

- ✅ **Asset IDs**: Perp = index, Spot = 10000 + index
- ✅ **Price Formatting**: Max 5 sig figs, proper decimals (6 for perps, 8 for spot)
- ✅ **Size Formatting**: Rounded to `szDecimals`, trailing zeros removed
- ✅ **Notation**: Uses Px, Sz, Szi as per API docs
- ✅ **Order Types**: IOC (Immediate or Cancel) limit orders

See: [Hyperliquid API Documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api)

## 🛡️ Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Use API wallets** for production (see Hyperliquid docs on nonces)
3. **Start with small sizes** to test
4. **Monitor closely** during initial runs
5. **Keep `DRY_RUN = True`** until you're confident

## 📝 Key Concepts

### Basis Trading
- Long spot + Short perp = Captures funding rate
- When funding is positive, you earn the funding rate
- Risk: Price movements can offset funding gains

### Funding Rates
- Positive: Longs pay shorts (good for short perp positions)
- Negative: Shorts pay longs
- Paid every 8 hours on Hyperliquid

### Breakeven Time
- Time needed to recover trading fees from funding
- Formula: `(TAKER_FEE * 4) / funding_rate` in hours
- Only enter if breakeven < `MAX_BREAKEVEN_H`

## 🤝 Contributing

1. Test changes with `DRY_RUN = True`
2. Follow existing code style
3. Add comments for complex logic
4. Update this README if adding features

## 📄 License

[Add your license here]

## ⚠️ Disclaimer

This software is for educational purposes. Trading cryptocurrencies involves substantial risk. Use at your own risk. The authors are not responsible for any losses.

---

**Happy Trading! 🚀**

