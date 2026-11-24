import _corpus_env  # loads .env, cd into corpus folder
import pandas as pd, json, os, re
from datetime import date, timedelta
m=pd.read_csv("data/FILINGS_METADATA.csv",dtype=str)
c=pd.read_csv("companies.csv",dtype={"cik":str})
m["cik"]=m.CIK.astype(str)
m=m.merge(c[["cik","ticker","name"]],on="cik")
m["fiscal_year"]=m["Period of Report"].map(lambda r:(date.fromisoformat(r)-timedelta(days=60)).year)
m["accession_number"]=m.html_index.str.extract(r"/(\d{10}-\d{2}-\d{6})-index")[0]
print("rows",len(m),"missing period",m["Period of Report"].isna().sum())
out=m[(m.fiscal_year>=2015)&(m.fiscal_year<=2024)].copy()
print("outside FY range:",len(m)-len(out))
out=out.sort_values(["cik","fiscal_year","Filing Date"])
dups=out[out.duplicated(["cik","fiscal_year"],keep=False)]
print("duplicate cik-FY rows:",len(dups)); print(dups[["ticker","fiscal_year","Period of Report","Filing Date","accession_number"]].to_string())
out=out.drop_duplicates(["cik","fiscal_year"],keep="first")
rows=[]
for _,r in out.iterrows():
    p=f"data/extracted/10-K/{r.filename.split('.')[0]}.json"
    d=json.load(open(p)) if os.path.exists(p) else {}
    wc={k:len(str(d.get(k) or "").split()) for k in ("item_1","item_1A","item_7","item_7A")}
    tot=sum(wc.values())
    rows.append(dict(cik=r.cik,ticker=r.ticker,name=r["name"],fiscal_year=r.fiscal_year,period_of_report=r["Period of Report"],filing_date=r["Filing Date"],accession_number=r.accession_number,json_path=p if d else "",
      words_item_1=wc["item_1"],words_item_1A=wc["item_1A"],words_item_7=wc["item_7"],words_item_7A=wc["item_7A"],words_total=tot,
      est_tokens_1_1A_7=int((wc["item_1"]+wc["item_1A"]+wc["item_7"])*1.33),est_tokens_all=int(tot*1.33)))
mf=pd.DataFrame(rows); mf.to_csv("manifest.csv",index=False)
print("manifest",len(mf)); print("tokens 1+1A+7:",mf.est_tokens_1_1A_7.sum(),"all:",mf.est_tokens_all.sum())
g=mf.pivot_table(index="ticker",columns="fiscal_year",values="accession_number",aggfunc="count").reindex(c.ticker).fillna(0).astype(int)
g.to_csv("scripts/grid.csv"); miss=(g==0).sum().sum()
print("missing cells",miss,"of",g.size); print("missing per year",(g==0).sum().to_dict())
print("companies with missing:"); print({t:[int(y) for y in r.index[r==0]] for t,r in g.iterrows() if (r==0).any()})
bad=mf[(mf.words_item_1<500)|(mf.words_item_1A<500)|(mf.words_item_7<500)]
bad.to_csv("scripts/extraction_failures.csv",index=False)
print("failures",len(bad)); print(bad[["ticker","fiscal_year","words_item_1","words_item_1A","words_item_7"]].to_string(index=False))
