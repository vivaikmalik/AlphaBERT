# Literature gap check: "event geometry" (FinBERT-embedding atypicality/novelty of 8-Ks and returns)

Scope: does anyone already (a) embed 8-Ks with a contextual/transformer model, (b) cluster into latent event
types, (c) score cross-sectional atypicality (distance to cluster centroid) and/or temporal novelty (distance
from the firm's own past filings), (d) detect "event waves" (spikes in cluster filing counts), signed by tone,
and (e) link this to next-month returns? Findings below, verified where a source could be fetched directly;
anything resting on a search-engine summary only is marked UNVERIFIED.

## Closest prior work, by ingredient

**Temporal novelty vs. a firm's own past disclosures (well-established, mostly with bag-of-words/cosine, not
deep embeddings):**

- Tetlock (2011), "All the News That's Fit to Reprint: Do Investors React to Stale Information?", *Review of
  Financial Studies* 24(5), 1481-1512. Staleness = cosine similarity of a news story to the prior 10 stories
  about the same firm; stale-news reactions partially reverse over the next week, informative-news reactions
  do not. Verified (RFS, Oxford Academic).
- Brown & Tucker (2011), "Large-Sample Evidence on Firms' Year-over-Year MD&A Modifications", *Journal of
  Accounting Research* 49(2), 309-346. Introduces a cosine-based year-over-year MD&A change score; larger
  economic change → larger modification; predictive power has weakened over time. Verified (JAR/Wiley).
- Cohen, Malloy & Nguyen (2020), "Lazy Prices", *Journal of Finance* 75(3), 1371-1415. Cosine/Jaccard
  similarity between a firm's current and immediately-prior 10-K/10-Q; "changers" minus "non-changers" earns
  up to ~188 bps/month; effect concentrated in CEO/CFO language, litigation, and risk-factor sections. This is
  the closest existing analogue to our *temporal* novelty axis, but on 10-K/10-Q, not 8-K, and uses classic
  vector-space (BoW) similarity rather than transformer embeddings. Verified (JoF/Wiley, NBER WP 25084).
- Nakatsuka & Suimon (2026, UNVERIFIED — could not confirm authors from a primary source; article itself
  verified to exist), "How does the market respond to textual novelty: Joint analysis of similarity and
  sentiment on Japanese corporate disclosures", *Economics Letters* (forthcoming/2026). Sentence-BERT
  cosine-similarity novelty vs. a firm's own past earnings-announcement text ("tanshin"), joined with tone;
  finds novel+toned sentences explain post-announcement CARs ([+5,+30] days) better than sentiment alone.
  This is the single closest paper to our sign(tone) x novelty hypothesis, but on Japanese earnings
  announcements, sentence-level (not filing-level), and with no cross-sectional clustering component.

**Cross-sectional atypicality vs. a latent, data-driven event-type cluster (the part we could not find):**

- Dyer, Lang & Stice-Lawrence (2017), "The Evolution of 10-K Textual Disclosure", *Journal of Accounting and
  Economics* 64(2), 221-245. LDA topics over 10-Ks 1996-2013; documents rising length, boilerplate, and
  "stickiness," with 3 of 150 LDA topics (fair value, internal controls, risk factors) driving most of the
  increase. Topic modeling is used descriptively, not as an atypicality/return-predictor score. Verified.
- Dolphin, Dursun, Blankenship, Adams & Pike (2026), "Grounded Event Extraction from SEC 8-K Filings with a
  Fine-Grained Taxonomy", arXiv:2607.08346 (unreviewed preprint, submitted July 2026). LLM-tags 292,984
  8-Ks (2022-2026) into a 119-type taxonomy finer than SEC item codes; an event study on unsigned abnormal
  returns confirms the finer taxonomy separates economically distinct events sharing one item code. Closest
  thing to "latent event types" for 8-Ks specifically, but the taxonomy is supervised/LLM-labeled (not
  k-means on embeddings), produces categorical tags rather than a continuous atypicality score, and does not
  test signed-tone atypicality against next-month returns. Verified to exist via arXiv; not peer-reviewed.

**Raw FinBERT/LLM embeddings as direct return predictors (no clustering/atypicality step):**

- Yılkı (2026), "Supply Chain Propagation of Textual Signals: LLM Embeddings and Cross-Sectional Return
  Predictability", arXiv:2606.29290 (unreviewed preprint, June 2026, single author). FinBERT embeddings of
  10-K MD&A for 255 S&P 500 firms, 2011-2025; network-propagated embedding factor predicts returns
  (Fama-MacBeth t = -2.64; FF5 alpha 7.27%/yr). Uses embeddings directly as a factor, not centroid-distance
  atypicality. Verified to exist via arXiv; not peer-reviewed, small/curated sample, no independent
  replication found.
- Chen, Kelly & Xiu (2022/2023), "Expected Returns and Large Language Models", SSRN 4416687 / Wharton
  working paper. LLM (ChatGPT/LLaMA-family) embeddings of financial news used directly as return predictors
  in a cross-sectional asset-pricing framework. Verified (SSRN, Wharton Jacobs Levy Center).
- Ke, Kelly & Xiu (2019), "Predicting Returns with Text Data" (SESTM), NBER WP 26186. Supervised
  sentiment-screening + topic model turns news text into a return-predictive sentiment score; trained
  out-of-sample (train/test split), the closest methodological precedent for our pre-2019 fit-then-apply
  design. Verified (NBER).
- Huang, Wang & Yang (2023), "FinBERT: A Large Language Model for Extracting Information from Financial
  Text", *Contemporary Accounting Research* 40(2). Domain-adapted BERT sentiment classifier, outperforms
  Loughran-McDonald dictionary and classical ML baselines. This is the embedding backbone our design and
  several of the above papers rely on. Verified (CAR/Wiley).
- Kim, Muhn & Nikolaev (2023), "Bloated Disclosures: Can ChatGPT Help Investors Process Financial
  Information?", NBER/SSRN/arXiv 2306.10224. GPT-generated MD&A/call-transcript summaries are shorter and
  their sentiment explains market reactions better than sentiment from the original (bloated) text —
  relevant precedent for "obfuscation via excess/atypical language dilutes signal." Verified.

**Other supporting/background citations, all verified:**
- Li (2008), *Journal of Accounting and Economics* 45, 221-247: low-readability (high Fog, long) annual
  reports associate with lower, less persistent earnings — mechanism for "bad news written in bespoke,
  obfuscating language."
- Bloomfield (2002), *Accounting Horizons* 16(3), 233-243: Incomplete Revelation Hypothesis — costlier-to-
  extract information is less completely reflected in price; theoretical basis for our under-reaction claim.
- Hanley & Hoberg (2010), *Review of Financial Studies* 23(7), 2821-2864: decomposes IPO prospectus text
  into "standard" vs. "informative" content; informative content lowers underpricing. Closest precedent for
  a standard-vs-idiosyncratic text decomposition, applied to IPOs not 8-Ks.
- Lerman & Livnat (2010), *Review of Accounting Studies* 15, 752-778: post-2004 8-K item-level market
  reactions; establishes item-level heterogeneity in 8-K informativeness that any 8-K clustering approach
  must account for.
- Cao, Jiang, Wang & Yang (2024), "From Man vs. Machine to Man + Machine", *Journal of Financial Economics*
  160: AI analyst built on disclosure/macro text beats human analysts more when information is
  high-dimensional and voluminous — supportive context for embedding-based approaches to disclosure text.
- Wang, Johnson, Hybinette & Balch (2025), "Is All the Information in the Price? LLM Embeddings versus the
  EMH in Stock Clustering", NeurIPS 2025 workshop / arXiv:2509.01590. Price-based clustering of stocks beats
  both GICS and LLM-news-embedding clustering for factor-model fit — a negative/cautionary result on how
  much genuinely new information text embeddings add beyond price. Verified (NeurIPS listing + arXiv).

## Answers

**(1) Has anyone measured embedding-space atypicality of 8-Ks vs. latent event-type clusters and/or own
history, linked to returns?**
Own-history (temporal) novelty: yes, repeatedly — Tetlock (2011) on news, Brown & Tucker (2011) and Cohen,
Malloy & Nguyen (2020) on 10-K/10-Q, and Nakatsuka & Suimon (2026, unverified) on Japanese earnings texts —
but always with bag-of-words/cosine or sentence-embedding similarity, not a k-means/PCA-whitened FinBERT
pipeline, and never on 8-Ks specifically. Cross-sectional atypicality vs. a *latent, unsupervised* event-type
cluster centroid: not found. The nearest thing (Dolphin et al. 2026, preprint) builds a fine-grained
*supervised* 8-K taxonomy and shows it separates return outcomes within an item code, but produces category
labels, not a continuous distance-based atypicality score, and does not combine it with tone or novelty.
"Event waves" (spikes in a cluster's monthly filing count) as a return signal: not found anywhere in the
literature searched.

**(2) The gap, precisely.** No study combines, on SEC 8-Ks: (i) frozen contextual (FinBERT) embeddings, (ii)
unsupervised latent event-type clusters fit out-of-sample (e.g., k-means on pre-period PCA-whitened
embeddings), (iii) a continuous cross-sectional atypicality score (distance to the firm's assigned cluster
centroid), (iv) a temporal novelty score (distance from the firm's own filing history) in the *same*
embedding space, (v) a cluster-level "event wave" abnormal-filing-count signal, and (vi) FinBERT-tone signing
of all three, tested jointly against next-month returns. Each ingredient individually has partial precedent
(temporal novelty: strong precedent, mostly non-neural; raw embeddings as predictors: emerging 2026
preprints; fine-grained supervised 8-K event types: one 2026 preprint), but the *cross-sectional atypicality
relative to an unsupervised latent cluster* and the *event-wave* ingredients appear to be the actually novel
parts of the idea; the temporal-novelty ingredient is the least novel and should be framed as an extension
(deep embeddings, 8-Ks, monthly horizon) of Cohen-Malloy-Nguyen/Brown-Tucker rather than as new territory.

**(3) Design choices the closest papers suggest.**
- Similarity/distance: cosine similarity is the field standard (Tetlock; Cohen-Malloy-Nguyen; Brown-Tucker);
  in whitened PCA space, Euclidean/Mahalanobis distance to centroid is the natural analogue — consistent
  with the plan.
- Horizons: short-horizon reaction-plus-reversal at about a week (Tetlock); drift over roughly a quarter
  (Cohen-Malloy-Nguyen, Lerman-Livnat); [+5,+30] trading days (Nakatsuka-Suimon). Testing both an immediate
  window and a ~1-3 month drift, not just next-month, is standard practice and would strengthen results.
- Controls: firm size, book-to-market, momentum/volatility, filing length/boilerplate share, industry, SEC
  item code, and earnings surprise (SUE) — so the atypicality score isn't just proxying for filing length or
  the magnitude of the underlying news. Cohen-Malloy-Nguyen also show effects concentrate in specific
  sections (risk factors, litigation, CEO/CFO language), suggesting per-cluster/per-item breakdowns of our
  atypicality effect, not just a pooled average.
- Out-of-sample discipline: Ke-Kelly-Xiu (SESTM) train on an early sample and test out-of-sample — directly
  analogous to, and supportive of, fitting k-means/PCA whitening on pre-2019 filings only.

**(4) Risks/confounds.**
- Length & boilerplate mechanics: Dyer-Lang-Stice-Lawrence show 10-K length/boilerplate rose mechanically
  1996-2013 from regulatory change, not firm news — atypicality/novelty scores need year fixed effects or
  detrending so they don't just pick up secular disclosure inflation or new FASB/SEC rules.
- Industry re-discovery: clusters may just recover GICS industry rather than "event type"; Wang et al. (2025)
  found LLM-embedding clusters add little over price-based/GICS clustering for factor-model fit — a caution
  that embedding-space structure may already be priced, working against the under-reaction hypothesis.
- Filing frequency / firm-size confound: unusual disclosures may cluster in small, illiquid, infrequent
  filers, conflating "atypicality" with known size/illiquidity premia; needs firm fixed effects or frequency
  controls (Lerman-Livnat show item usage varies greatly across firms).
- Tone-atypicality collinearity: if unusual events (restatements, going-concern, bankruptcy) are
  mechanically also the most negative, an "atypical x negative" effect could just be a sentiment effect in
  disguise; Nakatsuka-Suimon's finding that novelty adds power *beyond* sentiment is encouraging precedent
  but needs direct replication on 8-Ks/US data, not an assumption.
- Legal/template mechanics: many 8-K items (e.g., 5.02 executive changes, 8.01 catch-all) are drafted from
  law-firm boilerplate; "atypicality" could track which template/vendor was used rather than economic
  novelty — a purely mechanical confound distinct from information content.
- Cluster drift / concept drift: event types evolve (COVID, SPACs, crypto disclosures post-2020); clusters
  fit on pre-2019 data may misclassify later regimes, requiring drift diagnostics or periodic refitting.
- Calendar clustering: monthly "event waves" could just reflect earnings-season calendar effects (many firms
  file Item 2.02 in the same week) rather than a genuine surge in one latent event type — needs
  day-of-week/earnings-season controls before interpreting a wave as informative.
