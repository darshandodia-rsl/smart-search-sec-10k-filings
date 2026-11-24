import _corpus_env  # loads .env, cd into corpus folder
import pandas as pd, json, re, random
from bs4 import BeautifulSoup
mf=pd.read_csv("manifest.csv",dtype={"cik":str}); random.seed(7)
for _,r in mf.iloc[random.sample(range(len(mf)),5)].iterrows():
    d=json.load(open(r.json_path)); t=d["item_7"]
    raw=open(f"data/raw/10-K/{d['filename']}",errors="backslashreplace").read()
    txt=re.sub(r"\s+"," ",BeautifulSoup(raw,"lxml").get_text(" "))
    norm=lambda s:re.sub(r"\s+"," ",s)
    head=norm(t)[:80]; tail=norm(t)[-80:]
    print("="*100); print(r.ticker,r.fiscal_year,r.accession_number,"words",r.words_item_7)
    print("EXTRACTED START:",norm(t)[:200]); print("EXTRACTED END  :",norm(t)[-200:])
    i=[m.start() for m in re.finditer(re.escape(head[:50]),txt)]; j=[m.start() for m in re.finditer(re.escape(tail[-50:]),txt)]
    print("ORIGINAL before start (last occurrence):",txt[max(0,i[-1]-80):i[-1]+60] if i else "NOT FOUND")
    print("ORIGINAL after end (last occurrence)   :",txt[j[-1]+50:j[-1]+200] if j else "NOT FOUND (tables removed/whitespace)")
