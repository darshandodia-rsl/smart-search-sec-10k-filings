from _corpus_env import user_agent  # loads .env, cd into corpus folder
import requests, time, json, os, sys, pandas as pd
from datetime import date, timedelta
UA = user_agent()
H = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
CACHE = "scripts/cache"; os.makedirs(CACHE, exist_ok=True)
SIC_KEEP = {7370, 7371, 7372, 7373, 7374}
_last = [0.0]
def get(url):
    wait = 0.125 - (time.time() - _last[0])  # 8 req/s max
    if wait > 0: time.sleep(wait)
    for a in range(4):
        _last[0] = time.time()
        r = requests.get(url, headers=H, timeout=30)
        if r.status_code == 200: return r.json()
        if r.status_code == 404: return None
        time.sleep(2 * (a + 1))
    raise RuntimeError(url + " " + str(r.status_code))
def cached(name, url):
    p = f"{CACHE}/{name}.json"
    if os.path.exists(p):
        return json.load(open(p))
    d = get(url)
    json.dump(d, open(p, "w"))
    return d
def fy_of(rd):
    d = date.fromisoformat(rd) - timedelta(days=60)
    return d.year

tk = cached("company_tickers", "https://www.sec.gov/files/company_tickers.json")
by_cik = {}
for v in tk.values():
    by_cik.setdefault(v["cik_str"], []).append(v)
print("unique CIKs", len(by_cik), flush=True)

cands = []
for i, (cik, lst) in enumerate(by_cik.items()):
    s = cached(f"sub_{cik:010d}", f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if i % 500 == 0: print(i, flush=True)
    if not s: continue
    try: sic = int(s.get("sic") or 0)
    except: sic = 0
    if sic in SIC_KEEP:
        cands.append((cik, lst[0]["ticker"], s["name"], sic, s))
print("SIC-matching", len(cands), flush=True)

rows = []
for cik, tkr, name, sic, s in cands:
    recs = []
    def add(r):
        for f, rd, ad, fd in zip(r["form"], r["reportDate"], r["accessionNumber"], r["filingDate"]):
            if f == "10-K" and rd: recs.append((fy_of(rd), rd, fd, ad))
    add(s["filings"]["recent"])
    for fl in s["filings"].get("files", []):
        if fl["filingTo"] >= "2015-01-01":
            add(cached(f"subfile_{cik:010d}_{fl['name']}", f"https://data.sec.gov/submissions/{fl['name']}"))
    fys = {r[0] for r in recs}
    if 2024 not in fys: continue
    n = len([y for y in fys if 2015 <= y <= 2024])
    rows.append(dict(cik=cik, ticker=tkr, name=name, sic=sic, n_10k_fy2015_2024=n))
df = pd.DataFrame(rows)
print("with FY2024 10-K", len(df), flush=True)

fl = {}
for per in ["CY2023Q4I", "CY2024Q1I", "CY2024Q2I", "CY2024Q3I", "CY2024Q4I"]:
    d = cached(f"frame_{per}", f"https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/{per}.json")
    for x in d["data"]:
        fl.setdefault(x["cik"], []).append((x["end"], x["val"], per))
def pick(c):
    v = fl.get(c)
    return max(v)[1] if v else None
df["public_float_usd"] = df.cik.map(pick)
print("float coverage", df.public_float_usd.notna().sum(), "/", len(df))
df.to_csv("scripts/step1_raw.csv", index=False)
