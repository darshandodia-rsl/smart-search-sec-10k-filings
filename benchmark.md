# Benchmark questions and answers

The same questions are asked of every retrieval method, so later methods (graph retrieval,
Phase 6) can be compared against earlier ones. Each run records the exact answer, the
sources it was given, the token cost, and a hand check of the answer against its sources.

## Questions
| # | Question | Type |
|---|---|---|
| Q1 | Who does Microsoft consider its main competitors? | single company, relationship |
| Q2 | Which companies depend on a small number of semiconductor suppliers? | cross-company, specific fact |
| Q3 | How do companies describe cybersecurity breach risks? | cross-company, theme |

---

## Run 1: Phase 2 plain vector baseline (2026-11-25)

**Setup:** `sec10k.retrieval.qa`, k = 8 passages; bge-base-en-v1.5 (fp32) vector search over
170,187 chunks, 5× over-fetch, same-company year-over-year copies merged at cosine ≥ 0.95;
answer by `openai/gpt-oss-120b` on Groq (reasoning_effort=low, max 1,500 answer tokens).
No company/year filters.

| # | Tokens in / out | Verdict |
|---|---|---|
| Q1 | 2,184 / 353 | ✅ correct, fully cited |
| Q2 | 2,924 / 377 | ⚠️ 2 of 4 companies not supported by their sources; 1 relevant source missed |
| Q3 | 3,414 / 340 | ✅ each company's summary matches its source |

### Q1: Who does Microsoft consider its main competitors?

**Answer:**

> Microsoft's 10‑K filings list a broad set of rivals across its many businesses. The company repeatedly points to **Apple and Google** as its primary competitors for Windows, Office, Azure‑related services, and its consumer devices [1][5][6].
>
> In the cloud‑computing arena, Microsoft Azure is said to face "diverse competition" from **Amazon, Google, IBM, Oracle, Salesforce.com, VMware and open‑source offerings** [2].
>
> For its productivity and collaboration suite, Microsoft notes competition from a long list of software vendors that includes **Apple, Cisco Systems, Meta, Google, IBM, Okta, Proofpoint, Slack, Symantec, Zoom and others** [7].
>
> In the gaming space, Xbox and cloud‑gaming services compete with **Amazon, Apple, Meta, Tencent, Nintendo and Sony** [1].
>
> Its advertising business is described as competing chiefly with **Google** and a variety of web‑sites and social platforms such as **Meta (Facebook)** [1][3].
>
> Overall, Microsoft's filings portray **Apple, Google, Amazon, and a range of enterprise‑software firms (e.g., Salesforce, Oracle, SAP, IBM) as its principal competitors across its core product lines** [1][2][6][7].

**Sources:**
```
[1] 0.701  MSFT FY2016,2017,2018,2019,2020,2021,2022,2023,2024, Item 1 Business  | Competition
[2] 0.699  MSFT FY2015, Item 1 Business  | Competition
[3] 0.686  MSFT FY2015, Item 1 Business  | Competition
[4] 0.677  OKTA FY2019,2020,2021,2022,2023,2024, Item 1 Business  | Our Competitors
[5] 0.676  MSFT FY2015, Item 1 Business  | Competition
[6] 0.671  MSFT FY2015,2016,2017,2018, Item 1 Business  | Competition
[7] 0.659  MSFT FY2019,2021,2022,2023,2024, Item 1 Business  | Competition
[8] 0.654  MSFT FY2015, Item 1 Business  | Competition
```

**Check:** Claims match the sources. Notes: [4] is OKTA (no company filter) and was not used;
source [1] shows the FY2023 text (which mentions Xbox) but is cited as FY2016–2024, so the
answer implies the Xbox wording appears in all those years; no fiscal years in the answer.

### Q2: Which companies depend on a small number of semiconductor suppliers?

**Answer:**

> - **Apple Inc.** – Apple notes that many of its components, including some custom‑made parts, are obtained from a single or limited number of sources, making it vulnerable to supply disruptions of those semiconductor components【2】.
> - **Cadence Design Systems, Inc.** – Cadence states that it depends on sole suppliers for certain hardware components, which include semiconductor parts used in its emulation and prototyping hardware products【3】.
> - **Mobileye Global Inc.** – Mobileye explicitly relies on a single supplier, STMicroelectronics, for its EyeQ™ System‑on‑Chips, and notes that this limited‑source relationship creates a material supply‑chain risk【6】【8】.
> - **Fortinet, Inc.** – Fortinet's filing lists several key semiconductor components (CPUs, network and wireless chips, memory devices, etc.) that are available only from limited or sole sources, indicating dependence on a small number of semiconductor suppliers【7】.
>
> These companies all disclose in their SEC 10‑K risk‑factor or business sections that they depend on a small number of semiconductor suppliers for critical components.

**Sources:**
```
[1] 0.667  AGYS FY2017,2020,2022, Item 1A Risk Factors  | Our dependence on certain strategic partners makes us vulnerable to the extent we rely on them.
[2] 0.650  AAPL FY2019,2020,2021,2022,2023,2024, Item 1 Business  | Supply of Components
[3] 0.649  CDNS FY2015,...,2024, Item 1A Risk Factors  | We depend on sole suppliers for certain hardware components and contract manufacturers ...
[4] 0.648  TDC FY2020,2021,2022,2023,2024, Item 1A Risk Factors  | Our future results depend in part on our relationships with key suppliers, ...
[5] 0.645  TDC FY2015,2016,2017,2018, Item 1A Risk Factors  | Reliance on Third Parties-Our future results depend in part on ...
[6] 0.644  MBLY FY2022,2023,2024, Item 1A Risk Factors  | We face additional supply chain risks ... reliance on a single supplier or limited suppliers ...
[7] 0.638  FTNT FY2023,2024, Item 1 Business  | Manufacturing and Suppliers
[8] 0.635  MBLY FY2022,2023,2024, Item 7 MD&A  | Supply and manufacturing capacity.
```

**Check:** Searched each source for semiconductor/chip/CPU/Intel/TSMC/STMicro.
Mentions found only in [5] TDC (chips, Intel), [6] MBLY, [7] FTNT, [8] MBLY.
- ❌ AAPL [2] and CDNS [3] say only "components"; "semiconductor" was added by the model.
- ✅ MBLY and FTNT are supported.
- ❌ TDC [5] (Intel chips) was missed.
- 4 of 8 sources are generic sole-supplier text: vector search matched the *topic*, not the fact.

### Q3: How do companies describe cybersecurity breach risks?

**Answer:**

> Companies frame cybersecurity breach risks as threats that could damage reputation, disrupt operations, and impair financial performance. For example, DXC Technology warns that security breaches or cyber‑attacks could cause "significant financial loss" and service interruptions that harm its reputation and business [1]. Teradata notes that a cyber incident or failure of its own or third‑party systems could "adversely impact our reputation, business, and financial results," potentially causing data loss and system outages [2]. Certara says a breach could make its solutions "perceived as not being secure," leading customers to reduce or stop use and exposing the company to "significant liabilities" [3]. Five9 emphasizes that security breaches or cyber‑attacks could lead to litigation, regulatory risk, reputational damage and "significant liability" that affect results of operations [4]. SAIC states that cyber or other security threats could "negatively affect" its business and financial results, including unauthorized access to sensitive information and liability for damages [5]. Aurora describes that any material cyber‑event could prevent it from "effectively operating its business" and cause loss of intellectual property or data [6]. Leidos warns that breaches could "negatively impact…business and financial results," impair service delivery and harm reputation or competitive position [7]. Kyndryl similarly notes that cybersecurity and privacy incidents could "adversely impact our business," citing risks from hackers, nation‑states, and social‑engineering attacks [8].

**Sources:**
```
[1] 0.742  DXC FY2018, Item 1A Risk Factors
[2] 0.734  TDC FY2023,2024, Item 1A Risk Factors
[3] 0.734  CERT FY2024, Item 1A Risk Factors
[4] 0.729  FIVN FY2024, Item 1A Risk Factors
[5] 0.728  SAIC FY2023,2024, Item 1A Risk Factors
[6] 0.727  AUR FY2021,2023, Item 1A Risk Factors
[7] 0.727  LDOS FY2020, Item 1A Risk Factors
[8] 0.725  KD FY2021,2023,2024, Item 1A Risk Factors
```

**Check:** Each summary matches its source. Year-over-year merging gave 8 different companies.
Limitation: 8 of 130 companies, picked by wording similarity; nothing says how typical they are
or how the wording changed over time.
