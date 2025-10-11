from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from base import Pair
from metrics import unified_scaled_distance, corr_returns, estimate_beta_ols


class DistanceSelector:
    def __init__(self,):

        self.select_pairs_per_window: int = 3
        self.max_candidates: int = 5
        self.min_corr: float = 0.2
        self.use_log_price: bool = True
        self.bb_window_for_beta: int = 30  # 仅用于 beta 估计的最小窗约束

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:

        keys = list(prices.keys())
        scores: List[Tuple[str, str, float]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                d = unified_scaled_distance(prices[a], prices[b])
                scores.append((a, b, d))

        scores.sort(key=lambda x: x[2])
        out: List[Pair] = []
        for a, b, _ in scores[: self.max_candidates]:
            a_s, b_s = prices[a], prices[b]

            c = corr_returns(a_s, b_s)
            if not np.isfinite(c) or c < self.min_corr:
                continue
            beta = estimate_beta_ols(a_s, b_s, use_log_price=self.use_log_price)
            out.append((a, b, beta))

            if len(out) >= self.select_pairs_per_window:
                break
        return out