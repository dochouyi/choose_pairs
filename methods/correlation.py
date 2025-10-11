from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import pandas as pd
from base import Pair
import numpy as np
from metrics import corr_returns, estimate_beta_ols


class CorrelationSelector:
    def __init__(self, **kwargs):
        self.select_pairs_per_window: int = 1
        self.max_candidates: int = 20
        self.min_form_bars: int = 60
        self.min_corr: float = 0.2
        self.use_log_price: bool = False
        self.bb_window_for_beta: int = 30

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:

        keys = list(prices.keys())
        scores: List[Tuple[str, str, float]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                c = corr_returns(prices[a], prices[b])
                if np.isfinite(c):
                    scores.append((a, b, c))
        scores.sort(key=lambda x: -x[2])

        out: List[Pair] = []
        for a, b, c in scores[: self.max_candidates]:
            a_s, b_s = prices[a], prices[b]
            if len(a_s) < max(self.min_form_bars, 30):
                continue
            if not np.isfinite(c) or c < self.min_corr:
                continue
            beta = estimate_beta_ols(a_s, b_s, use_log_price=self.use_log_price)

            out.append((a, b, beta))
            if len(out) >= self.select_pairs_per_window:
                break
        return out