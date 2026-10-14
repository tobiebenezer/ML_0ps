"""Sessions 4-5: KS drift test + latency percentiles + cost formula."""
import numpy as np
from scipy.stats import ks_2samp

from app.core.redis_client import get_redis
from app.core.config import settings

_mem_times: list[float] = []  # fallback when Redis is down


def ks_drift(baseline: np.ndarray, new: np.ndarray, p_threshold: float = 0.05) -> dict:
    stat, p_value = ks_2samp(np.asarray(baseline, dtype=float), np.asarray(new, dtype=float))
    drifted = bool(p_value < p_threshold)
    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "drifted": drifted,
        # Recommend retrain only on real drift (Session 4 framework: investigate first)
        "recommend_retrain": drifted,
    }


def record_latency_ms(ms: float) -> None:
    r = get_redis()
    if r is None:
        _mem_times.append(ms)
        del _mem_times[: settings.latency_window]
        return
    try:
        r.rpush("latency:window", ms)
        r.ltrim("latency:window", -settings.latency_window, -1)
    except Exception:
        _mem_times.append(ms)


def get_latencies() -> list[float]:
    r = get_redis()
    if r is not None:
        try:
            return [float(v) for v in r.lrange("latency:window", 0, -1)]
        except Exception:
            pass
    return list(_mem_times)


def latency_stats() -> dict:
    times = get_latencies()
    if not times:
        return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.array(times, dtype=float)
    return {
        "count": len(times),
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def cost_per_1000(hourly_cost_usd: float) -> dict:
    """Session 4-5 formula: cost/pred = hourly_cost / preds_per_hour."""
    s = latency_stats()
    avg_s = (s["avg_ms"] / 1000.0) or 0.05  # sane default before any traffic
    pph = 3600.0 / avg_s
    return {
        "avg_latency_s": avg_s,
        "predictions_per_hour": pph,
        "cost_per_1000_usd": (hourly_cost_usd / pph) * 1000,
    }
