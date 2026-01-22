#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperliquid Basis Trading Bot
This script executes basis trades on Hyperliquid exchange.
"""

import os
import sys
import time
import math
import logging

# Force unbuffered output for real-time logging
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass  # Fallback for older Python versions

print("🚀 Script starting...", flush=True)

import json
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import colorlog

print("✓ Imports successful", flush=True)

# Load environment variables
load_dotenv()
print("✓ Environment variables loaded", flush=True)

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Secrets
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")
    API_URL = constants.MAINNET_API_URL
    
    # --- SAFETY SWITCH ---
    DRY_RUN = True  # Set to False to trade REAL MONEY
    
    # --- STRATEGY SETTINGS ---
    LOOP_INTERVAL_SEC = 30        
    SCALE_COOLDOWN_SEC = 3600     
    
    # LEVERAGE
    LEVERAGE = 3 
    LEVERAGE_MODE = "cross" 

    # PORTFOLIO
    ENTRY_SIZE_PCT = 0.05         # 5% of Equity per trade
    MAX_POS_PCT = 0.20            # Max 20% of Equity per coin
    SCALE_STEP_PCT = 0.02         
    
    # FUNDING THRESHOLDS
    MIN_ENTRY_FUNDING = 0.00002   
    SCALE_UP_FUNDING = 0.00005    
    SCALE_DOWN_FUNDING = -0.00001 
    
    # EXECUTION
    TAKER_FEE = 0.00035
    SCANNER_SLIPPAGE = 0.005
    EXECUTION_SLIPPAGE = 0.001
    MAX_BREAKEVEN_H = 12.0
    MAX_EXECUTION_RETRIES = 3

    @staticmethod
    def validate():
        if not Config.PRIVATE_KEY or not Config.ACCOUNT_ADDRESS:
            raise ValueError("❌ Missing PRIVATE_KEY or ACCOUNT_ADDRESS in .env file")

# ==========================================
# 2. LOGGING SETUP
# ==========================================
def setup_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# ==========================================
# 3. UTILITIES
# ==========================================
class Utils:
    @staticmethod
    def round_sz(size, decimals):
        """
        Format size according to Hyperliquid API specs:
        - Sizes are rounded to szDecimals of that asset
        - Round down to avoid exceeding balance
        - Trailing zeros should be removed (per signing docs)
        
        Per API docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size
        """
        if size == 0: return "0"
        factor = 10 ** decimals
        # Round down to avoid exceeding balance
        rounded = math.floor(size * factor) / factor
        # Format with exact decimals, then remove trailing zeros (per signing docs)
        formatted = f"{rounded:.{decimals}f}".rstrip('0').rstrip('.')
        return formatted

    @staticmethod
    def round_px(price, sz_decimals, is_perp=True):
        """
        Format price according to Hyperliquid API specs:
        - Max 5 significant figures
        - Max decimals: 6 - szDecimals for perps, 8 - szDecimals for spot
        - Integer prices always allowed regardless of sig figs
        - Trailing zeros should be removed (per signing docs)
        
        Per API docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size
        """
        if price == 0: return "0"
        
        # Determine max decimals based on asset type
        # Perps: MAX_DECIMALS = 6, Spot: MAX_DECIMALS = 8
        max_decimals = (6 if is_perp else 8) - sz_decimals
        
        # Check if price is an integer (always allowed)
        if price == int(price):
            return str(int(price))
        
        # Format with max 5 significant figures
        sig_fig_str = f"{price:.5g}"
        final_price = float(sig_fig_str)
        
        # Round to max_decimals, but don't exceed it
        final_price = round(final_price, max_decimals)
        
        # Format and remove trailing zeros (per signing docs)
        formatted = f"{final_price:.{max_decimals}f}".rstrip('0').rstrip('.')
        return formatted

# ==========================================
# 4. DATA MANAGER
# ==========================================
class DataManager:
    def __init__(self, info_api):
        self.info = info_api
        self.perp_meta = None
        self.spot_meta = None
        self.perp_map = {} 
        self.spot_map = {} 

    def refresh(self):
        try:
            # 1. PERP META
            # Per API docs: Perp asset ID = index in meta() universe array
            # E.g., BTC = 0 on mainnet
            self.perp_meta = self.info.meta()
            self.perp_map = {a['name']: i for i, a in enumerate(self.perp_meta['universe'])}
            
            # 2. SPOT META
            # Per API docs: Spot asset ID = 10000 + spotInfo["index"]
            # where spotInfo is from spotMeta() universe
            # Note: Spot coin names use @{index} format (e.g., @1, @107) except PURR/USDC
            # The name field in spotMeta already contains the correct format
            self.spot_meta = self.info.spot_meta()
            self.spot_map = {a['name']: 10000 + a['index'] for a in self.spot_meta['universe']}
            
        except Exception as e:
            logger.error(f"Failed to refresh metadata: {e}")

    def get_perp_details(self, coin):
        """
        Get perp asset details and ID.
        Per API docs: Perp asset ID = index in meta() universe array.
        Returns: (asset_info_dict, asset_id)
        """
        idx = self.perp_map.get(coin)
        if idx is None: return None, None
        return self.perp_meta['universe'][idx], idx

    def get_spot_details(self, coin):
        """
        Get spot asset details and ID.
        Per API docs: Spot asset ID = 10000 + spotInfo["index"].
        Returns: (asset_info_dict, asset_id)
        """
        asset_id = self.spot_map.get(coin)
        if asset_id is None: return None, None
        # Find asset config by reversing the ID calculation
        raw_index = asset_id - 10000
        asset = next((a for a in self.spot_meta['universe'] if a['index'] == raw_index), None)
        return asset, asset_id

    def calculate_slippage(self, coin, usd_size, is_buy):
        """
        Calculate slippage for a given order size.
        Uses L2 snapshot: levels[0] = asks (sell side), levels[1] = bids (buy side)
        """
        try:
            l2 = self.info.l2_snapshot(coin)
            # L2 levels: 0 = asks (sell side), 1 = bids (buy side)
            side = 1 if is_buy else 0
            levels = l2['levels'][side]
            filled_qty, total_cost, remaining_usd = 0, 0, usd_size
            
            for level in levels:
                p, s = float(level['px']), float(level['sz'])
                lvl_usd = p * s
                if lvl_usd >= remaining_usd:
                    qty = remaining_usd / p
                    total_cost += qty * p
                    filled_qty += qty
                    remaining_usd = 0
                    break
                else:
                    total_cost += lvl_usd
                    filled_qty += s
                    remaining_usd -= lvl_usd
            
            if remaining_usd > 1.0: return 1.0
            avg_px = total_cost / filled_qty
            best_px = float(levels[0]['px'])
            return abs(avg_px - best_px) / best_px
        except Exception:
            return 1.0

# ==========================================
# 5. TRADE EXECUTOR (FIXED FOR SPOT)
# ==========================================
class TradeExecutor:
    def __init__(self, account, exchange, data_mgr):
        self.account = account
        self.exchange = exchange
        self.mgr = data_mgr
        self.leverage_set_cache = set()

    def set_leverage(self, coin, asset_id):
        if coin in self.leverage_set_cache: return
        if Config.DRY_RUN:
            self.leverage_set_cache.add(coin)
            return
        try:
            is_cross = True if Config.LEVERAGE_MODE == "cross" else False
            self.exchange.update_leverage(asset_id, is_cross, Config.LEVERAGE)
            self.leverage_set_cache.add(coin)
        except Exception:
            pass

    def place_order(self, coin, is_buy, usd_size, price, asset_id, is_perp):
        # 1. Formatting
        if is_perp:
            asset_info, _ = self.mgr.get_perp_details(coin)
        else:
            asset_info, _ = self.mgr.get_spot_details(coin)
            
        sz_decimals = asset_info['szDecimals']
        raw_sz = usd_size / price
        sz_str = Utils.round_sz(raw_sz, sz_decimals)
        
        buffer = Config.EXECUTION_SLIPPAGE
        limit_px = price * (1 + buffer) if is_buy else price * (1 - buffer)
        # Pass is_perp flag to round_px for correct decimal handling
        px_str = Utils.round_px(limit_px, sz_decimals, is_perp=is_perp)

        if Config.DRY_RUN:
            type_str = "PERP" if is_perp else "SPOT"
            side = "BUY" if is_buy else "SELL"
            logger.info(f"   [DRY RUN] {side} {type_str} {coin} | Sz: {sz_str} @ {px_str}")
            return {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": sz_str}}]}}}

        # 2. Payload Construction
        # According to Hyperliquid API docs:
        # - "a": Asset ID (integer) - Perp: index, Spot: 10000 + index
        # - "b": Side (boolean) - True = Buy/Bid, False = Sell/Ask
        #   Note: Docs mention "B"/"A" notation, but SDK accepts boolean
        # - "p": Price (Px) - formatted string (max 5 sig figs, trailing zeros removed)
        # - "s": Size (Sz) - in units of coin (base currency), rounded to szDecimals
        # - "r": Reduce only flag
        # - "t": Order type with TIF - "Ioc" = Immediate or Cancel
        # 
        # Note: Nonces are handled automatically by the SDK/Exchange class.
        # Per API docs, nonces must be within (T - 2 days, T + 1 day) where T is unix ms timestamp.
        # The SDK manages nonce tracking per signer address.
        #
        # We use 'bulk_orders' because it lets us pass the raw 'asset_id' explicitly.
        # This works for both Spot (10000+) and Perps (index).
        order_payload = {
            "a": asset_id,           # Asset ID per API spec
            "b": is_buy,             # Side: True=Buy/Bid, False=Sell/Ask
            "p": px_str,             # Price (Px) - formatted string with proper sig figs
            "s": sz_str,             # Size (Sz) - in units of coin, rounded to szDecimals
            "r": False,              # Reduce only flag
            "t": {"limit": {"tif": "Ioc"}}  # Limit order, Immediate or Cancel
        }

        try:
            # We send a "bulk" of 1 order to ensure explicit asset mapping
            # The SDK handles nonce management automatically
            return self.exchange.bulk_orders([order_payload])
        except Exception as e:
            logger.error(f"   ❌ API Error: {e}")
            return {"status": "error"}

    def execute_basis_trade(self, coin, price, size_usd, is_entry=True):
        action = "OPEN" if is_entry else "CLOSE"
        logger.info(f"⚡ {action} {coin} | Size: ${size_usd:.2f}")

        _, spot_id = self.mgr.get_spot_details(coin)
        _, perp_id = self.mgr.get_perp_details(coin)

        if is_entry: self.set_leverage(coin, perp_id)

        # LEG 1: SPOT
        spot_is_buy = True if is_entry else False
        spot_res = self.place_order(coin, spot_is_buy, size_usd, price, spot_id, is_perp=False)
        
        spot_filled_sz = 0.0
        if spot_res['status'] == 'ok':
            # Bulk order response format check
            status = spot_res['response']['data']['statuses'][0]
            if 'filled' in status:
                spot_filled_sz = float(status['filled']['totalSz'])
            else:
                logger.error(f"   ❌ Spot not filled. Status: {status}")
                return 
        else:
            logger.error(f"   ❌ Spot API failed: {spot_res}")
            return

        # LEG 2: PERP
        perp_is_buy = not spot_is_buy
        if spot_filled_sz > 0:
            filled_usd_val = spot_filled_sz * price
            perp_success = False
            for i in range(Config.MAX_EXECUTION_RETRIES):
                perp_res = self.place_order(coin, perp_is_buy, filled_usd_val, price, perp_id, is_perp=True)
                
                # Check success inside bulk response
                if perp_res['status'] == 'ok':
                    p_stat = perp_res['response']['data']['statuses'][0]
                    if 'filled' in p_stat:
                        logger.info(f"   ✅ Trade Complete.")
                        perp_success = True
                        break
                
                time.sleep(1)

            if not perp_success:
                logger.critical(f"   🚨 PERP FAILED. Reversing Spot.")
                reverse_is_buy = not spot_is_buy
                self.place_order(coin, reverse_is_buy, filled_usd_val, price, spot_id, is_perp=False)

# ==========================================
# 6. PORTFOLIO MANAGER (With Dashboard)
# ==========================================
class PortfolioManager:
    def __init__(self, info, mgr, executor):
        self.info = info
        self.mgr = mgr
        self.exec = executor
        self.last_scale_time = {}

    def print_status(self, account_addr):
        try:
            # Refresh meta to ensure we have latest pricing
            logger.debug("Refreshing metadata...")
            self.mgr.refresh()
            
            logger.debug(f"Fetching user state for {account_addr}...")
            user_state = self.info.user_state(account_addr)
            margin = user_state['marginSummary']
            positions = user_state['assetPositions']
            
            equity = float(margin['accountValue'])
            used_margin = float(margin['totalMarginUsed'])
            
            # --- HEADER ---
            print("\n" + "="*70)
            print(f" 📊 PORTFOLIO | Equity: ${equity:.2f} | Margin: {used_margin:.2f}")
            print("-" * 70)
            print(f" {'COIN':<6} {'SIDE':<5} {'SIZE($)':<10} {'ENTRY':<10} {'MARK':<10} {'PnL($)':<8} {'FUNDING'}")
            print("-" * 70)
            
            has_pos = False
            for p in positions:
                pos = p['position']
                coin = pos['coin']
                size = float(pos['szi'])
                if abs(size) == 0: continue
                
                has_pos = True
                entry_px = float(pos['entryPx']) if pos['entryPx'] else 0
                
                # Get Context
                _, asset_idx = self.mgr.get_perp_details(coin)
                if asset_idx is None: 
                    logger.warning(f"  ⚠️  Skipping {coin} - not found in perp metadata")
                    continue # Skip if metadata drift

                # Get latest price/funding from the meta refresh
                try:
                    ctx = self.info.meta_and_asset_ctxs()[1][asset_idx]
                    curr_px = float(ctx['midPx'])
                    funding = float(ctx['funding']) * 100 
                    
                    val_usd = abs(size * curr_px)
                    side = "LONG" if size > 0 else "SHORT"
                    pnl = (curr_px - entry_px) * size 
                    
                    print(f" {coin:<6} {side:<5} {val_usd:<10.2f} {entry_px:<10.4f} {curr_px:<10.4f} {pnl:+.2f}    {funding:+.4f}%")
                except (IndexError, KeyError) as e:
                    logger.warning(f"  ⚠️  Error getting context for {coin} (idx {asset_idx}): {e}")
                    continue
            
            if not has_pos:
                print(" (No open positions)")
            print("="*70 + "\n")
            
            return user_state, positions, equity
            
        except Exception as e:
            logger.error(f"Dashboard Error: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, [], 0

    def rebalance(self, user_state, positions, equity):
        if not user_state: return

        current_time = time.time()
        max_pos_usd = equity * Config.MAX_POS_PCT
        scale_step_usd = equity * Config.SCALE_STEP_PCT
        
        for p in positions:
            pos = p['position']
            coin = pos['coin']
            size = float(pos['szi'])
            if abs(size) == 0: continue

            if current_time - self.last_scale_time.get(coin, 0) < Config.SCALE_COOLDOWN_SEC:
                continue

            _, asset_idx = self.mgr.get_perp_details(coin)
            if asset_idx is None: continue
            
            ctx = self.info.meta_and_asset_ctxs()[1][asset_idx]
            price = float(ctx['midPx'])
            funding = float(ctx['funding'])
            curr_val_usd = abs(size * price)

            # SCALING LOGIC
            if funding > Config.SCALE_UP_FUNDING and curr_val_usd < max_pos_usd:
                amt = min(scale_step_usd, max_pos_usd - curr_val_usd)
                logger.info(f"📈 Scaling UP {coin} (Funding High)")
                self.exec.execute_basis_trade(coin, price, amt, is_entry=True)
                self.last_scale_time[coin] = current_time

            elif funding < Config.SCALE_DOWN_FUNDING:
                amt = min(scale_step_usd, curr_val_usd)
                logger.warning(f"📉 Scaling DOWN {coin} (Funding Neg)")
                self.exec.execute_basis_trade(coin, price, amt, is_entry=False)
                self.last_scale_time[coin] = current_time

# ==========================================
# 7. MARKET SCANNER
# ==========================================
class MarketScanner:
    def __init__(self, info, mgr):
        self.info = info
        self.mgr = mgr

    def scan(self, equity):
        try:
            # Ensure metadata is refreshed (in case print_status failed)
            if not self.mgr.spot_map:
                logger.debug("Refreshing metadata for market scan...")
                self.mgr.refresh()
            
            # Log debug info about spot_map size
            logger.debug(f"Spot map has {len(self.mgr.spot_map)} coins")
            
            # Data is already refreshed by Dashboard call in main loop
            meta_ctx = self.info.meta_and_asset_ctxs()
            universe = meta_ctx[0]['universe']
            ctxs = meta_ctx[1]

            candidates = []
            candidates_for_trading = []  # Only coins that exist in both spot and perp
            entry_size_usd = equity * Config.ENTRY_SIZE_PCT

            for i, asset in enumerate(universe):
                coin = asset['name']
                
                try:
                    # Get funding and price, handling None values
                    funding_raw = ctxs[i].get('funding')
                    price_raw = ctxs[i].get('midPx')
                    
                    # Skip if either value is None or empty
                    if funding_raw is None or price_raw is None:
                        continue
                    
                    funding = float(funding_raw)
                    price = float(price_raw)
                    
                    # Skip if price is invalid (zero or negative)
                    if price <= 0:
                        continue
                    
                    # Include ALL coins with positive funding for market pulse display
                    # (Don't filter by spot_map here - we want to show all funding rates)
                    if funding > 1e-10:  # Very small threshold to catch tiny positive values
                        candidates.append((coin, funding, price))
                        
                        # Only add to trading candidates if it also exists in spot
                        if coin in self.mgr.spot_map:
                            candidates_for_trading.append((coin, funding, price))
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    logger.debug(f"Skipping {coin} due to data error: {e}")
                    continue
            
            logger.debug(f"Found {len(candidates)} candidates with positive funding (all perps)")
            logger.debug(f"Found {len(candidates_for_trading)} candidates available for trading (spot+perp)")

            # Sort top 5 for display
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Always show market pulse, even if no candidates
            # Force flush to ensure output is visible immediately
            print(f" 🔍 MARKET PULSE (Top Funding Rates):", flush=True)
            if len(candidates) > 0:
                for i in range(min(5, len(candidates))):
                    c = candidates[i]
                    be_h = (Config.TAKER_FEE * 4) / c[1] if c[1] > 0 else 999
                    print(f"    {i+1}. {c[0]:<5} | Rate: {c[1]*100:.4f}% | BE: {be_h:.1f}h", flush=True)
            else:
                print("    (No positive funding rates found)", flush=True)
            print("", flush=True)

            # Select Best Viable (only when equity > 0)
            # Use candidates_for_trading (coins that exist in both spot and perp) for actual trades
            best_opp = None
            if equity > 0 and entry_size_usd > 0:
                for c in candidates_for_trading:
                    coin, fund, px = c
                    if fund <= Config.MIN_ENTRY_FUNDING: continue
                    
                    be_h = (Config.TAKER_FEE * 4) / fund
                    if be_h > Config.MAX_BREAKEVEN_H: continue
                    
                    slip = self.mgr.calculate_slippage(coin, entry_size_usd, is_buy=True)
                    if slip > Config.SCANNER_SLIPPAGE: continue
                    
                    best_opp = {"coin": coin, "price": px, "size": entry_size_usd}
                    break 

            return best_opp

        except Exception as e:
            logger.error(f"Error in market scan: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f" 🔍 MARKET PULSE (Top Funding Rates):", flush=True)
            print("    (Error loading market data)", flush=True)
            print("", flush=True)
            return None  # Return None on error

# ==========================================
# 8. MAIN BOT
# ==========================================
class BasisBot:
    def __init__(self):
        Config.validate()
        self.account = Account.from_key(Config.PRIVATE_KEY)
        self.info = Info(Config.API_URL, skip_ws=True)
        self.exchange = Exchange(self.account, Config.API_URL)
        
        self.mgr = DataManager(self.info)
        self.exec = TradeExecutor(self.account, self.exchange, self.mgr)
        self.scanner = MarketScanner(self.info, self.mgr)
        self.portfolio = PortfolioManager(self.info, self.mgr, self.exec)

    def run(self):
        print(f"HYPERLIQUID BOT INITIALIZED | Mode: {'DRY RUN' if Config.DRY_RUN else 'LIVE'}")
        print(f"Account: {self.account.address}")
        print(f"Loop interval: {Config.LOOP_INTERVAL_SEC} seconds")
        print("Starting main loop...\n")
        
        while True:
            try:
                # Debug: Log that we're starting an iteration
                logger.debug("Starting loop iteration...")
                
                state, positions, equity = self.portfolio.print_status(self.account.address)
                
                # Always show market pulse, regardless of equity
                opp = self.scanner.scan(equity)
                
                # Check for Equity > 0 to avoid crash
                if equity > 0:
                    if state: self.portfolio.rebalance(state, positions, equity)
                    
                    # Only execute trades if we have equity and found an opportunity
                    if opp:
                        # Only enter if we don't hold it
                        already_held = any(p['position']['coin'] == opp['coin'] for p in positions)
                        if not already_held:
                            logger.info(f"💎 OPPORTUNITY: {opp['coin']}")
                            self.exec.execute_basis_trade(opp['coin'], opp['price'], opp['size'], is_entry=True)
                else:
                    logger.warning("Equity is $0. Please deposit USDC.")
                    # Sleep longer if no funds
                    time.sleep(300)

                time.sleep(Config.LOOP_INTERVAL_SEC)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(10)

if __name__ == "__main__":
    bot = BasisBot()
    bot.run()