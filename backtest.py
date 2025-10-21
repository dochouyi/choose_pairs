import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
from ta.volatility import BollingerBands
# 使用你已有的选择器，不做修改
from methods.distance import DistanceSelector
from methods.sdr import SDRSelector
warnings.filterwarnings("ignore")


# =========================
# 交易记录的数据结构
# =========================
@dataclass
class TradeRecord:
    method: str
    pair: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    beta: float
    w_a: float
    w_b: float
    qty_a: float
    qty_b: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    fee: float
    pnl: float
    pnl_pct: float
    capital_per_leg: float
    bars_held: int

# =========================
# 配置
# =========================
@dataclass
class Config:
    freq: str = "5min"
    form_period: int = 200
    capital_per_trade: float = 1000.0
    fee_rate: float = 0.0003
    bb_window: int = 30
    bb_k: float = 3.0
    std_clip: float = 1e-6
    z_exit_to_sma: bool = True
    z_stop: float = 6.0
    vol_weight: bool = True
    debug: bool = True
    save_dir: Optional[str] = "out"
    recompute_candidates_every: int = 20

    # 执行与风险控制
    use_next_bar_price: bool = True
    slippage_bps: float = 1.0
    max_holding_bars: Optional[int] = 100
    cooldown_bars: int = 1

    # 候选过滤（在回测端可选附加过滤）
    min_corr: float = 0.2
    max_adf_p: float = 0.2
    use_adf_filter: bool = True
    min_form_bars: int = 60

    # 数量分配
    beta_neutral_qty: bool = True
    min_qty_clip: float = 1e-9

    # 利润门槛与缓冲
    min_take_profit_pct: float = 0.2
    enforce_profit_on_cross: bool = True
    takeprofit_exit_buffer_z: float = 0.2

    def __post_init__(self):
        if self.bb_k <= 0:
            raise ValueError("cfg.bb_k must be > 0")

# =========================
# 工具函数
# =========================
def load_freqtrade_file(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)
    arr = np.array(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Unexpected data shape in {path}: {arr.shape}")
    df = pd.DataFrame(arr, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce").astype("Int64"), unit="ms", utc=True)
    df = df.dropna(subset=["ts"])
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open","high","low","close","volume"]]

def parse_symbol_from_filename(fn: str) -> str:
    base = os.path.basename(fn)
    if base.endswith(".json"):
        base = base[:-5]
    if "-5m-futures" in base:
        return base.split("-5m-futures")[0]
    return base

def load_freqtrade_dir(dir_path: str, max_symbols: int = None) -> Dict[str, pd.DataFrame]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory not found: {dir_path}")
    files = [fn for fn in os.listdir(dir_path) if fn.endswith("-5m-futures.json")]
    files.sort()
    if max_symbols:
        files = files[:max_symbols]
    data = {}
    for fn in files:
        full = os.path.join(dir_path, fn)
        sym = parse_symbol_from_filename(full)
        try:
            df = load_freqtrade_file(full)
            data[sym] = df
        except Exception as e:
            print(f"Skip {fn}: {e}")
    if not data:
        raise FileNotFoundError("No -5m-futures.json found.")
    return data

def pair_vol(a: pd.Series, b: pd.Series, window: int) -> Tuple[float, float]:
    ra = a.pct_change().rolling(window, min_periods=window).std().iloc[-1]
    rb = b.pct_change().rolling(window, min_periods=window).std().iloc[-1]
    ra = float(ra) if np.isfinite(ra) and ra > 0 else 1e-4
    rb = float(rb) if np.isfinite(rb) and rb > 0 else 1e-4
    return ra, rb

def compute_bb(spread: pd.Series, cfg: Config):
    bb = BollingerBands(close=spread, window=cfg.bb_window, window_dev=cfg.bb_k, fillna=False)
    sma = bb.bollinger_mavg()
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    sd = ((upper - sma) / cfg.bb_k).clip(lower=cfg.std_clip)
    return sma, upper, lower, sd

def get_exec_price(series: pd.Series, t: int, use_next: bool, slippage_bps: float, min_clip: float=1e-12) -> float:
    if use_next and (t+1) < len(series):
        px = float(series.iloc[t+1])
    else:
        px = float(series.iloc[t])
    px = max(px, min_clip)
    return px * (1.0 + slippage_bps*1e-4)

def compute_position_sizes(ea: float, eb: float, beta: float, cfg: Config,
                           va: Optional[float]=None, vb: Optional[float]=None) -> Tuple[float,float,float,float]:
    cap = cfg.capital_per_trade
    if cfg.vol_weight and (va is not None) and (vb is not None):
        w_a = 1.0/max(va, cfg.min_qty_clip)
        w_b = 1.0/max(vb, cfg.min_qty_clip)
    else:
        w_a = w_b = 0.5
    s = w_a + w_b
    w_a /= s; w_b /= s
    if cfg.beta_neutral_qty:
        denom = (ea**2 + (beta*eb)**2)
        if denom <= 0:
            qty_a = (w_a*cap)/max(ea, cfg.min_qty_clip)
            qty_b = (w_b*cap)/max(eb, cfg.min_qty_clip)
        else:
            N_target = (w_a*cap + beta*w_b*cap) / (1.0 + beta)
            qty_a = N_target / max(ea, cfg.min_qty_clip)
            qty_b = (N_target / max(beta, cfg.min_qty_clip)) / max(eb, cfg.min_qty_clip)
    else:
        qty_a = (w_a*cap)/max(ea, cfg.min_qty_clip)
        qty_b = (w_b*cap)/max(eb, cfg.min_qty_clip)
    return float(w_a), float(w_b), float(qty_a), float(qty_b)

# =========================
# 回测类：调用 DistanceSelector
# =========================
class Backtester:
    def __init__(self, cfg: Config, selector: DistanceSelector):
        self.cfg = cfg
        self.selector = selector
        self.method_name = "DistanceSelector"

    def _try_open(self, A: pd.Series, B: pd.Series, beta: float, t_idx: int, label: str):
        spread = A - beta*B
        sma, upper, lower, sd = compute_bb(spread, self.cfg)
        if t_idx < len(spread):
            sp = spread.iloc[t_idx]
            ma = sma.iloc[t_idx]; up = upper.iloc[t_idx]; lo = lower.iloc[t_idx]
            if np.isnan(ma) or np.isnan(up) or np.isnan(lo):
                return None
            if sp >= up:
                side = "shortA_longB"
            elif sp <= lo:
                side = "longA_shortB"
            else:
                return None

            pa = get_exec_price(A, t_idx, self.cfg.use_next_bar_price, self.cfg.slippage_bps)
            pb = get_exec_price(B, t_idx, self.cfg.use_next_bar_price, self.cfg.slippage_bps)
            ts = spread.index[t_idx]
            va, vb = pair_vol(A.iloc[:t_idx+1], B.iloc[:t_idx+1], self.cfg.bb_window)
            w_a, w_b, qty_a, qty_b = compute_position_sizes(pa, pb, beta, self.cfg, va, vb)

            state = {
                "pair_label": label,
                "side": side,
                "entry_price_a": float(pa),
                "entry_price_b": float(pb),
                "entry_time": ts,
                "w_a": float(w_a),
                "w_b": float(w_b),
                "qty_a": float(qty_a),
                "qty_b": float(qty_b),
                "beta": float(beta),
                "entry_bar_index": int(t_idx)
            }
            return state
        return None

    def _step_manage(self, A: pd.Series, B: pd.Series, beta: float, state: Dict,
                     start_bar: int, end_bar: int,
                     window_start: pd.Timestamp, window_end: pd.Timestamp):
        cfg = self.cfg
        spread = A - beta*B
        sma, upper, lower, sd = compute_bb(spread, cfg)
        in_pos = True
        side = state["side"]
        ea = float(state["entry_price_a"])
        eb = float(state["entry_price_b"])
        et = pd.Timestamp(state["entry_time"])
        w_a = float(state["w_a"])
        w_b = float(state["w_b"])
        qty_a = float(state["qty_a"])
        qty_b = float(state["qty_b"])
        entry_bar_index = int(state.get("entry_bar_index", start_bar))

        closed_records = []
        bars_consumed = start_bar
        for t in range(start_bar, min(end_bar+1, len(spread))):
            sp = spread.iloc[t]; ma = sma.iloc[t]; sdd = sd.iloc[t]
            if np.isnan(ma) or np.isnan(sdd) or sdd <= 0:
                bars_consumed = t
                continue

            z = (sp - ma) / sdd

            # 止损
            stop_flag = abs(z) >= cfg.z_stop

            # 中轨缓冲判定
            cross_ok = False
            if cfg.z_exit_to_sma:
                if side == "shortA_longB":
                    if cfg.takeprofit_exit_buffer_z > 0:
                        cross_ok = sp <= ma and abs(z) <= cfg.takeprofit_exit_buffer_z
                    else:
                        cross_ok = sp <= ma
                elif side == "longA_shortB":
                    if cfg.takeprofit_exit_buffer_z > 0:
                        cross_ok = sp >= ma and abs(z) <= cfg.takeprofit_exit_buffer_z
                    else:
                        cross_ok = sp >= ma

            # 最长持仓
            hold_bars = t - entry_bar_index
            timeout_flag = (cfg.max_holding_bars is not None) and (hold_bars >= cfg.max_holding_bars)

            # 跨中轨时利润门槛
            want_close = False
            if cross_ok:
                if cfg.enforce_profit_on_cross and (cfg.min_take_profit_pct is not None) and (cfg.min_take_profit_pct > 0.0):
                    pa_tmp = get_exec_price(A, t, cfg.use_next_bar_price, cfg.slippage_bps)
                    pb_tmp = get_exec_price(B, t, cfg.use_next_bar_price, cfg.slippage_bps)
                    if side == "shortA_longB":
                        pnl_tmp = qty_a*(ea-pa_tmp) + qty_b*(pb_tmp-eb)
                    else:
                        pnl_tmp = qty_a*(pa_tmp-ea) + qty_b*(eb-pb_tmp)
                    fee_tmp = cfg.fee_rate*(ea*qty_a+eb*qty_b+pa_tmp*qty_a+pb_tmp*qty_b)
                    pnl_tmp -= fee_tmp
                    pnl_pct_tmp = pnl_tmp/(cfg.capital_per_trade)*100.0
                    want_close = pnl_pct_tmp >= cfg.min_take_profit_pct
                else:
                    want_close = True

            if stop_flag or timeout_flag or want_close:
                pa = get_exec_price(A, t, cfg.use_next_bar_price, cfg.slippage_bps)
                pb = get_exec_price(B, t, cfg.use_next_bar_price, cfg.slippage_bps)
                ts = spread.index[t]

                if side == "shortA_longB":
                    pnl = qty_a*(ea-pa) + qty_b*(pb-eb)
                else:
                    pnl = qty_a*(pa-ea) + qty_b*(eb-pb)
                fee = cfg.fee_rate*(ea*qty_a+eb*qty_b+pa*qty_a+pb*qty_b)
                pnl -= fee
                pnl_pct = pnl/(cfg.capital_per_trade)*100.0

                rec = TradeRecord(
                    method=self.method_name,
                    pair=state.get("pair_label","A-B"),
                    window_start=window_start,
                    window_end=window_end,
                    entry_time=et,
                    exit_time=ts,
                    side=side,
                    beta=float(beta),
                    w_a=float(w_a),
                    w_b=float(w_b),
                    qty_a=float(qty_a),
                    qty_b=float(qty_b),
                    entry_price_a=float(ea),
                    entry_price_b=float(eb),
                    exit_price_a=float(pa),
                    exit_price_b=float(pb),
                    fee=float(fee),
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    capital_per_leg=float(cfg.capital_per_trade),
                    bars_held=int(hold_bars if hold_bars>=0 else 0)
                )
                closed_records.append(rec)
                in_pos = False
                bars_consumed = t
                return closed_records, None, bars_consumed

            bars_consumed = t

        if in_pos:
            new_state = {
                "pair_label": state["pair_label"],
                "side": side,
                "entry_price_a": ea,
                "entry_price_b": eb,
                "entry_time": et,
                "w_a": w_a,
                "w_b": w_b,
                "qty_a": qty_a,
                "qty_b": qty_b,
                "beta": float(beta),
                "entry_bar_index": entry_bar_index
            }
            return closed_records, new_state, bars_consumed

    def backtest(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cfg = self.cfg
        symbols = list(data.keys())
        closes = {s: data[s]["close"].copy() for s in symbols}
        all_index = closes[symbols[0]].index

        all_trade_records: List[Dict] = []
        pos_state: Optional[Dict] = None

        cached_pairs: List[Tuple[str,str,float]] = []
        last_recompute_bar = -1
        cooldown_until_bar = -1

        def recompute_candidates(upto_bar: int):
            start = upto_bar - cfg.form_period
            if start < 0:
                return []
            form_index = all_index[start:upto_bar]
            form_prices = {s: closes[s].loc[form_index] for s in symbols}
            # DistanceSelector 内部有自己的 min_corr 与 beta 估算，这里仅做基本有效性过滤
            form_prices = {k:v.dropna() for k,v in form_prices.items() if v.dropna().shape[0] >= max(cfg.min_form_bars, cfg.bb_window)}
            if len(form_prices) < 2:
                return []
            # 直接调用你新写的选择器
            pairs = self.selector.select_pairs(form_prices)
            # 返回 [(a,b,beta), ...]
            return pairs

        i = cfg.form_period
        while i < len(all_index):
            in_cooldown = (i <= cooldown_until_bar)

            # 仅在空仓且需重算时更新候选
            if pos_state is None and (not in_cooldown) and (last_recompute_bar < 0 or (i - last_recompute_bar) >= cfg.recompute_candidates_every):
                cached_pairs = recompute_candidates(i)
                last_recompute_bar = i
                if cfg.debug:
                    print(f"[{all_index[i]}] Recomputed candidates: {len(cached_pairs)}")

            # 尝试开仓（单持仓）
            if pos_state is None and (not in_cooldown):
                for (a,b,beta) in cached_pairs:
                    A = closes[a]; B = closes[b]
                    label = f"{a}-{b}"
                    state = self._try_open(A, B, beta, i, label)
                    if state is not None:
                        pos_state = state
                        if cfg.debug:
                            print(f"[{all_index[i]}] Open {label} {state['side']} at A={state['entry_price_a']:.6f}, B={state['entry_price_b']:.6f}")
                        break
                i += 1
                continue

            # 管理持仓
            if pos_state is not None:
                a, b = pos_state["pair_label"].split("-")
                beta = float(pos_state["beta"])
                A = closes[a]; B = closes[b]
                closed_records, new_state, consumed = self._step_manage(
                    A, B, beta, pos_state, start_bar=i, end_bar=i,
                    window_start=all_index[max(0, i - cfg.form_period)],
                    window_end=all_index[i]
                )
                for rec in closed_records:
                    all_trade_records.append(asdict(rec))
                    if cfg.debug:
                        print(f"[{rec.exit_time}] Close {rec.pair} {rec.side} pnl={rec.pnl_pct:.4f}% (bars={rec.bars_held})")
                    cooldown_until_bar = max(cooldown_until_bar, i + cfg.cooldown_bars)
                pos_state = new_state
                i += 1
                continue

            # 冷却期推进
            if pos_state is None and in_cooldown:
                i += 1
                continue

        if pos_state is not None and cfg.debug:
            print(f"End with open position: {pos_state['pair_label']} since {pos_state['entry_time']}")

        trades_df = pd.DataFrame(all_trade_records)
        if trades_df.empty:
            print("[INFO] No completed trades.")
            return trades_df

        # 保存
        save_base = cfg.save_dir or "out"
        os.makedirs(save_base, exist_ok=True)
        ts_str = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(save_base, f"trades_seq_{self.method_name}_{ts_str}.csv")
        json_path = os.path.join(save_base, f"trades_seq_{self.method_name}_{ts_str}.json")

        sort_cols = [c for c in ["window_start", "pair", "entry_time"] if c in trades_df.columns]
        if sort_cols:
            trades_df = trades_df.sort_values(sort_cols).reset_index(drop=True)
        trades_df.to_csv(csv_path, index=False)

        def _json_default(o):
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, pd.Timestamp):
                if pd.isna(o):
                    return None
                return o.isoformat()
            try:
                if o is pd.NaT:
                    return None
            except Exception:
                pass
            return str(o)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(trades_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=_json_default)

        print(f"[{self.method_name}] Trades saved to: {csv_path} and {json_path}")

        # 绩效统计
        trades_df["cum_pnl_pct"] = trades_df["pnl_pct"].cumsum()
        trades_df["ret"] = 1.0 + trades_df["pnl_pct"]/100.0
        trades_df["equity"] = trades_df["ret"].cumprod()
        trades_df["peak"] = trades_df["equity"].cummax()
        trades_df["dd"] = trades_df["equity"]/trades_df["peak"] - 1.0
        max_dd = trades_df["dd"].min() if not trades_df.empty else 0.0

        avg_bars = trades_df["bars_held"].mean() if "bars_held" in trades_df.columns else np.nan
        med_bars = trades_df["bars_held"].median() if "bars_held" in trades_df.columns else np.nan

        wins = (trades_df["pnl_pct"] > 0).sum()
        total_trades = len(trades_df)
        global_winrate = wins / total_trades if total_trades > 0 else float("nan")
        total_return_pct = trades_df["pnl_pct"].sum() if total_trades > 0 else 0.0
        avg_return_pct = trades_df["pnl_pct"].mean() if total_trades > 0 else 0.0
        std_return_pct = trades_df["pnl_pct"].std(ddof=1) if total_trades > 1 else 0.0

        print("\n=== Global trade-level stats (Completed trades only) ===")
        print(f"Method: {self.method_name}")
        print(f"Completed trades: {total_trades}")
        print(f"Winrate: {global_winrate:.2%}" if total_trades > 0 else "Winrate: N/A")
        print(f"Total return (sum of pnl_pct): {total_return_pct:.2f}%")
        print(f"Avg return per trade: {avg_return_pct:.4f}%  |  Std: {std_return_pct:.4f}%")
        print(f"Equity (final): {trades_df['equity'].iloc[-1]:.4f}  |  Max Drawdown: {max_dd:.2%}")
        if np.isfinite(avg_bars):
            print(f"Bars held: avg={avg_bars:.2f}, median={med_bars:.0f}")

        return trades_df

# =========================
# 示例运行入口
# =========================
if __name__ == "__main__":
    data_dir = "/home/houyi/crypto/pair_trading/data/"
    print(f"Loading data from: {data_dir}")
    data = load_freqtrade_dir(data_dir, max_symbols=33)

    cfg = Config(
        freq="5min",
        form_period=100,
        capital_per_trade=1000.0,
        fee_rate=0.0003,
        bb_window=30,
        bb_k=3,
        std_clip=1e-6,
        z_exit_to_sma=True,
        z_stop=6.0,
        vol_weight=True,
        debug=True,
        save_dir="out",
        recompute_candidates_every=20,

        use_next_bar_price=True,
        slippage_bps=1.0,
        max_holding_bars=100,
        cooldown_bars=1,

        min_corr=0.2,
        max_adf_p=0.2,
        use_adf_filter=True,
        min_form_bars=60,

        beta_neutral_qty=True,

        min_take_profit_pct=0.2,
        enforce_profit_on_cross=True,
        takeprofit_exit_buffer_z=0.2
    )

    selector = SDRSelector()  # 使用你新写的选择器
    bt = Backtester(cfg, selector)
    trades_df = bt.backtest(data)
    if not trades_df.empty:
        print(trades_df.tail(5))