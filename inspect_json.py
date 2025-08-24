import json
from collections import Counter
from statistics import mean, median, pstdev
from math import isfinite

# helper loader
def _load(path_or_list):
    if isinstance(path_or_list, list):
        return path_or_list
    with open(path_or_list, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("JSON root is not a list")
    return obj

def analyze_fields(path_or_list):
    """
    Return dict:
      - total_entries
      - unique_keysets_count
      - by_keyset_counts (mapping of tuple(keys) -> count)
      - field_count_distribution (mapping field_count -> count)
      - fields_uniform (bool)
      - sample_keys (one example key list)
    """
    data = _load(path_or_list)
    keysets = Counter()
    count_by_fieldcount = Counter()
    sample = None
    for e in data:
        if not isinstance(e, dict):
            continue
        ks = tuple(sorted(e.keys()))
        keysets[ks] += 1
        count_by_fieldcount[len(ks)] += 1
        if sample is None:
            sample = ks
    return {
        "total_entries": len(data),
        "unique_keysets_count": len(keysets),
        "by_keyset_counts": {k: v for k, v in keysets.items()},
        "field_count_distribution": dict(count_by_fieldcount),
        "fields_uniform": len(keysets) == 1,
        "sample_keys": list(sample) if sample is not None else [],
    }

def relevance_stats(path_or_list, compute_correlations=True):
    """
    Return dict with relevance distribution:
      - count, min, max, mean, median, stddev
    If compute_correlations=True also returns simple Pearson-like correlations
    (relevance vs totalArticleCount, relevance vs summary length) when possible.
    """
    data = _load(path_or_list)
    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return None

    relev = []
    totarts = []
    summ_lens = []
    for e in data:
        if not isinstance(e, dict):
            continue
        r = safe_int(e.get("relevance"))
        if r is not None:
            relev.append(r)
        tac = safe_int(e.get("totalArticleCount"))
        if tac is not None:
            totarts.append(tac)
        s = e.get("summary")
        stext = ""
        if isinstance(s, dict):
            stext = " ".join(v for v in s.values() if isinstance(v, str))
        elif isinstance(s, str):
            stext = s
        summ_lens.append(len(stext))

    out = {}
    if relev:
        out["count"] = len(relev)
        out["min"] = min(relev)
        out["max"] = max(relev)
        out["mean"] = mean(relev)
        out["median"] = median(relev)
        out["stddev"] = pstdev(relev) if len(relev) > 1 else 0.0
    else:
        out["count"] = 0

    def pearson(x, y):
        if not x or not y or len(x) != len(y):
            return None
        mx = mean(x); my = mean(y)
        num = sum((a-mx)*(b-my) for a,b in zip(x,y))
        denx = sum((a-mx)**2 for a in x)
        deny = sum((b-my)**2 for b in y)
        if denx == 0 or deny == 0:
            return None
        return num / ((denx**0.5)*(deny**0.5))

    if compute_correlations and relev:
        # align to min lengths where needed
        m = min(len(relev), len(totarts))
        if m>0:
            out["corr_relevance_totalArticleCount"] = pearson(relev[:m], totarts[:m])
        else:
            out["corr_relevance_totalArticleCount"] = None
        m2 = min(len(relev), len(summ_lens))
        if m2>0:
            out["corr_relevance_summary_length"] = pearson(relev[:m2], summ_lens[:m2])
        else:
            out["corr_relevance_summary_length"] = None

    return out

def unique_categories(path_or_list):
    """
    Return tuple (n_unique, Counter of category -> frequency, list of unique categories).
    Expects each entry's 'categories' to be a list of strings.
    """
    data = _load(path_or_list)
    ctr = Counter()
    sum_ctr = Counter()
    for i, e in enumerate(data):
        if not isinstance(e, dict):
            continue
        cats = e.get("categories") or []
        if isinstance(cats, list):
            for c in cats:
                if isinstance(c, str):
                    ctr[c] += 1
        
        sum = e.get("summary")["eng"]
        sum_ctr[str(i)] = len(sum)

    # calculate summary length distribution
    summary_length_distribution = Counter(sum_ctr.values())
    return len(ctr), ctr, list(ctr.keys()), len(sum_ctr), summary_length_distribution

# print(analyze_fields("newsapi.json"))
'''
{'total_entries': 6326, 
'unique_keysets_count': 1, 
'by_keyset_counts': {('categories', 'eventDate', 'id', 'location', 'relevance', 'summary', 'title', 'totalArticleCount'): 6326}, 
'sample_keys': ['categories', 'eventDate', 'id', 'location', 'relevance', 'summary', 'title', 'totalArticleCount']}
'''
# print(relevance_stats("newsapi.json"))
print(unique_categories("newsapi.json"))

''''
calculate summary length distribution

Counter({501: 5856, 500: 180, 499: 143, 497: 48, 498: 30, 495: 24, 
493: 13, 496: 12, 468: 10, 467: 1, 298: 1, 487: 1, 480: 1, 443: 1, 
492: 1, 491: 1, 494: 1, 476: 1, 341: 1}))
'''
