from _corpus_env import user_agent  # loads .env, cd into corpus folder
import requests, time, json, os, pandas as pd
from datetime import date, timedelta
H={"User-Agent":user_agent()}
def get(u):
    time.sleep(0.15); r=requests.get(u,headers=H,timeout=30); r.raise_for_status(); return r.json()
def fy_of(rd): return (date.fromisoformat(rd)-timedelta(days=60)).year
# HubSpot float entries across frames
for per in ["CY2023Q4I","CY2024Q1I","CY2024Q2I","CY2024Q3I","CY2024Q4I"]:
    d=json.load(open(f"scripts/cache/frame_{per}.json"))
    for x in d["data"]:
        if x["cik"] in (1404655,): print("HUBS",per,x["end"],x["val"],x["accn"])
# extra companies
tk=json.load(open("scripts/cache/company_tickers.json"))
tick={v["ticker"]:v for v in tk.values()}
fl={}
for per in ["CY2023Q4I","CY2024Q1I","CY2024Q2I","CY2024Q3I","CY2024Q4I"]:
    for x in json.load(open(f"scripts/cache/frame_{per}.json"))["data"]: fl.setdefault(x["cik"],[]).append((x["end"],x["val"]))
rows=[]
for t in ["PANW","FTNT","AAPL"]:
    cik=tick[t]["cik_str"]; s=get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    fys=set()
    def add(r):
        for f,rd in zip(r["form"],r["reportDate"]):
            if f=="10-K" and rd: fys.add(fy_of(rd))
    add(s["filings"]["recent"])
    for f in s["filings"].get("files",[]):
        if f["filingTo"]>="2015-01-01": add(get("https://data.sec.gov/submissions/"+f["name"]))
    rows.append(dict(cik=cik,ticker=t,name=s["name"],sic=int(s["sic"]),n_10k_fy2015_2024=len([y for y in fys if 2015<=y<=2024]),public_float_usd=max(fl[cik])[1] if cik in fl else None,fy2024=2024 in fys))
print(pd.DataFrame(rows).to_string())
pd.DataFrame(rows).to_csv("scripts/extra.csv",index=False)
