from typing import Dict
from data import load_prices_from_csv
from methods.distance import DistanceSelector
from methods.correlation import CorrelationSelector
from methods.cointegration import CointegrationSelector
from methods.sdr import SDRSelector
from methods.ga import GASelector
from methods.nsga2 import NSGA2Selector


def demo(csv_path: str):
    prices = load_prices_from_csv(csv_path)

    distance = DistanceSelector()
    corr = CorrelationSelector()
    coint = CointegrationSelector()
    sdr = SDRSelector()
    ga = GASelector(pairs_per_chrom=5, pop=80, gen=60, candidate_source="distance", use_log_price=False)
    nsga = NSGA2Selector(pairs_per_chrom=5, pop=80, gen=60, use_log_price=False)

    methods = {
        # "Distance": distance,
        # "Correlation": corr,
        # "Cointegration": coint,
        # "SDR": sdr,
        "GA": ga,
        "NSGA-II": nsga,
    }

    results: Dict[str, list] = {}
    for name, method in methods.items():
        pairs = method.select_pairs(prices)
        results[name] = pairs

    for k, v in results.items():
        print(f"[{k}]")
        for a, b, beta in v:
            print(f"  ({a}, {b}) beta={beta:.3f}")
        print()

if __name__ == "__main__":
    demo("data.csv")