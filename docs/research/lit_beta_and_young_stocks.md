# Beta Estimation for Young/Speculative Names, and Why Our Book Isn't Market-Neutral

Context: ex-ante beta is constrained to ~0, but realized beta is -0.3 to -0.5, likely
because the model shorts young IPO/de-SPAC/speculative names whose historical beta is
missing (filled with 1.0) or understated by naive OLS, when true beta for this cohort is
closer to 1.25-1.7.

## 1. Best-practice beta estimation

**Blume (1971, *Journal of Finance* 26(1), "On the Assessment of Risk")** showed betas
regress toward 1 across periods and proposed adjusting raw OLS beta toward the
cross-sectional mean (source of the classic `0.67*raw + 0.33*1.0` adjustment); **Blume
(1975, *JF* 30(3), "Betas and Their Regression Tendencies")** formalized this. **Vasicek
(1973, *JF* 28(5), "A Note on Using Cross-Sectional Information in Bayesian Estimation of
Security Betas")** generalized it into precision-weighted Bayesian shrinkage: noisy
(short-history) betas shrink hard toward the population mean, well-estimated ones barely
move — the correct starting point for young names.

**Levi & Welch (2017, *JFQA* 52(2), "Best Practice for Cost-of-Capital Estimates")**
recommend Vasicek-shrunk betas from 1-4 years of daily data, shrunk a *second* time (more
for smaller/younger firms); if own history is short, use market-cap peer betas — industry
averages "should never be used," they have almost no predictive power. **Welch (2022,
*Critical Finance Review* 11, "Simply Better Market Betas")** shows winsorizing daily
returns at roughly -2x/+4x the market return before regressing ("slope-winsorized beta")
plus return-age decay (~3-5mo half-life) beats plain OLS/EWMA — directly targets the
extreme-single-day distortion typical of illiquid young stocks.

**Frazzini & Pedersen (2014, *Journal of Financial Economics* 111(1), "Betting Against
Beta")**: `beta = corr(long window, ~5yr weekly) * (sigma_i / sigma_m, short window,
~1yr)`. Decoupling a slow correlation estimate from a fast volatility ratio suits exactly
the case of rising idiosyncratic vol with a still-noisy correlation.

**Cosemans, Frehen, Schotman & Bauer (2016, *Review of Financial Studies* 29(4),
"Estimating Security Betas Using Prior Information Based on Firm Fundamentals")**: a
hybrid estimator shrinking rolling-window beta toward a cross-sectionally *fitted* prior
(size, book-to-market, momentum, industry) rather than a flat mean; reduces tail
measurement error and the resulting betas carry significant priced risk that flat-Vasicek
beta doesn't. Closest paper to our proposed characteristics-based prior.

**Hollstein, Prokopczuk & Wese Simen** ("Estimating Beta," *Journal of Financial Markets*
2019; "The Conditional CAPM Revisited: Evidence from High-Frequency Betas," *Management
Science* 2020): forecast shrinkage should condition on firm characteristics, and
high-frequency/intraday realized betas forecast future beta better than daily OLS —
useful where listing history is too short for reliable daily OLS.

**Time-varying beta**: **Adrian & Franzoni (2009, *Journal of Empirical Finance* 16(4),
"Learning about Beta")** model beta as a Kalman-filtered latent state with uncertain
long-run mean — naturally wide uncertainty for young names with short histories. **Ang &
Kristensen (2012, published in *JFE*; earlier NBER WP, "Testing Conditional Factor
Models")** use nonparametric kernel-weighted regressions to let beta vary smoothly.
Both favor a state-space/kernel estimator over fixed-window OLS around structural breaks
(lockup expiry, de-SPAC redemption), where a jump is more likely than a drift.

## 2. Beta and returns of young/IPO/SPAC/lottery stocks

**Ritter (1991, *JF* 46(1), "The Long-Run Performance of IPOs")** and **Loughran & Ritter
(1995, *JF* 50(1), "The New Issues Puzzle")**: IPOs/SEOs underperform matched non-issuers
by a wide margin over 3-5 years — the foundational "new issues puzzle." This motivates
shorting young names on *alpha* grounds but says nothing about their *beta*; conflating
underperformance with low systematic risk is likely the source of our bug.

**Gahng, Ritter & Zhang (2023, *Review of Financial Studies* 36(9), "SPACs")**: equal-
weighted de-SPAC returns average -11.3% in year one vs. +19.4% for the market
(2010-2020, n=152) — a pool of small, speculative names that, having only recently begun
trading as the merged entity, almost certainly have their true beta understated by any
"missing history -> 1.0" fill rule.

**Bali, Cakici & Whitelaw (2011, *JFE* 99(3), "Maxing Out: Stocks as Lotteries...")**:
high-MAX (lottery-like) stocks earn >1%/month lower future returns than low-MAX stocks,
and MAX resolves the idio-vol discount puzzle. A real, non-beta alpha source concentrated
in this same cohort — argues for fixing beta *measurement*, not avoiding the cohort.

**Short-selling frictions**: **D'Avolio (2002, *JFE* 66(2-3), "The Market for Borrowing
Stock")**: institutional ownership explains ~55% of loan supply; "special" (expensive-to-
borrow) stocks concentrate among small, low-ownership, high-disagreement names — the same
profile. **Drechsler & Drechsler (2014, NBER WP 20282, "The Shorting Premium and Asset
Pricing Anomalies")**: short fees interact with nearly every major anomaly, which is
concentrated in the ~20% of stocks with high borrow costs — the shorting premium
compensates concentrated short risk. Our shorts here likely carry unmodeled borrow/squeeze
risk on top of the beta error.

## 3. Why market-neutral books drift, and robust construction

**Daniel & Moskowitz (2016, *JFE* 122(2), "Momentum Crashes")**: a factor book built
neutral ex ante can develop large, state-dependent market exposure (momentum's short leg
turns sharply negative-beta in panic/rebound states) — a static ex-ante constraint alone
is not sufficient; exposure needs dynamic monitoring.

**Robust optimization under parameter uncertainty**: **Goldfarb & Iyengar (2003,
*Mathematics of Operations Research* 28(1), "Robust Portfolio Selection Problems")**
formalize ellipsoidal uncertainty sets around estimated betas/covariances, solvable as a
second-order cone program. **Ceria & Stubbs (2006, *Journal of Asset Management* 7(2),
"Incorporating Estimation Errors into Portfolio Selection")** show mean-variance weights
are highly sensitive to small input errors and that penalizing worst-case exposure within
a confidence ellipsoid improves realized behavior. Direct support for a robust
(uncertainty-margin) beta-neutral constraint, with margin width tied to name-level
estimation uncertainty.

## Recommendations

- **Replace the flat 1.0 fill with Vasicek shrinkage toward a Cosemans-style
  characteristics prior** (size, sector, realized vol, leverage, listing age) — never
  toward a flat industry average, which Levi & Welch show has almost no predictive power.

- **Bias the prior beta for young/IPO/de-SPAC names above 1.0 (~1.2-1.5+), not toward the
  ~1.0 population mean.** Small size, high idio-vol, high leverage, high MAX, and recent
  listing all predict high beta in this literature; add an explicit age term that decays
  toward the population mean only after ~2-3 years of history (Levi & Welch's window
  guidance).

- **Fuse Frazzini-Pedersen (long-window corr x short-window vol ratio), slope-winsorized
  OLS (Welch 2022), and, if available, high-frequency realized beta (Hollstein et al.) in
  the precision-weighted combination** rather than relying on one estimator.

- **Use a Kalman-filtered or kernel-weighted time-varying beta (Adrian & Franzoni 2009;
  Ang & Kristensen 2012) instead of fixed-window OLS** around known structural breaks
  (lockup expiry, de-SPAC redemption), since beta likely jumps rather than drifts there.

- **Move from a point-estimate to an uncertainty-margin (robust) beta-neutral constraint**
  (Goldfarb & Iyengar 2003; Ceria & Stubbs 2006): constrain worst-case beta within a
  confidence band, penalizing reliance on high-uncertainty young-name betas to hit
  neutrality.

- **Add a portfolio-level realized-beta hedge as a backstop, not a substitute.** Standard
  practice in defensive-equity portfolios and justified by Daniel & Moskowitz's finding
  that neutral-by-design books still drift; it reacts after the fact and doesn't fix
  mis-sized individual shorts or their borrow-cost risk (D'Avolio 2002; Drechsler &
  Drechsler 2014), so bottom-up beta fixes remain primary.
