from metrics import engle_granger_beta, adf_test_simple
from typing import Dict, List, Tuple, Set
import pandas as pd
from base import Pair
from metrics import corr_returns
import numpy as np


class CointegrationSelector:
    def __init__(self, **kwargs):
        self.select_pairs_per_window: int = 5
        self.use_log_price: bool = False
        self.adf_lags: int = 1
        self.adf_crit: float = -3.3
        self.min_corr: float = 0.2
        self.bb_window_for_beta: int = 30

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:

        keys = list(prices.keys())
        candidates: List[Tuple[str, str, float, float]] = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):

                a, b = keys[i], keys[j]
                a_s, b_s = prices[a], prices[b]

                beta_coint, resid = engle_granger_beta(a_s, b_s, use_log_price=self.use_log_price)
                if resid.empty or resid.isna().all():
                    continue
                t_stat = adf_test_simple(resid.values, lags=self.adf_lags)
                if np.isfinite(t_stat) and t_stat < self.adf_crit:
                    c = corr_returns(a_s, b_s)
                    if np.isfinite(c) and c >= self.min_corr:
                        # 使用协整beta
                        candidates.append((a, b, float(beta_coint), float(c)))

        candidates.sort(key=lambda x: -x[3])
        out: List[Pair] = []
        for a, b, beta, _ in candidates:
            out.append((a, b, beta))
            if len(out) >= self.select_pairs_per_window:
                break
        return out