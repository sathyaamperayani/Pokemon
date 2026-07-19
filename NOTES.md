# Task 2 Notes: EDA and Cleaning Decisions

## First look
Raw data: 1,351 rows, 14 columns. No exact duplicate rows. Missing values
found in `base_experience` (49 rows, ~3.6%) and `secondary_type` (586 rows,
~43%). No obvious garbage values in numeric columns, but two things stood
out on closer inspection (see below).

## Cleaning decisions

**1. Removed alternate forms (id > 10000).** PokeAPI assigns IDs above
10000 to alternate forms of existing species (mega evolutions, gigantamax
forms, regional variants, etc.) — e.g. `charizard-gmax` and `deoxys-attack`
are separate rows from their base species. I removed these 326 rows
because: (a) they represent the same species being counted multiple times,
which would bias the model toward species with many known variants, and
(b) all 34 "gmax" forms shared an identical weight value of exactly 10000,
which is PokeAPI's placeholder for "unknown," not real data.

**2. Filled missing `base_experience` with the median.** Only ~3.6% of
rows were affected, so dropping them felt wasteful. Median (not mean) was
used because the distribution is right-skewed by a small number of very
high-experience Pokemon (legendaries), which would pull a mean estimate
upward.

**3. Filled missing `secondary_type` with the label "none".** This
column being blank isn't missing data in the usual sense — it means the
Pokemon genuinely has only one type. Treating it as its own category
(rather than imputing a statistic, which doesn't make sense for a
category) preserves that information for the model.

**4. Dropped exact duplicate rows defensively.** None were found in this
run, but the step is kept in the pipeline in case re-fetching ever
introduces any.

**5. Checked for rare classes.** After removing alternate forms, the
smallest primary-type class is Flying with 9 examples. This is small but
not critical (a 1–2 example class would be); I'm keeping it as its own
class for now, but flagging it for Task 3 — with a small test split,
Flying may end up with very few or zero test examples, which will affect
how meaningfully we can evaluate that class specifically.

**6. One-hot encoded `secondary_type`.** This is a feature, not the
target, so it's encoded here. `primary_type` (the target) is left as
text — it gets its own `LabelEncoder` in Task 3, per the assignment.

## Result
Clean dataset: 1,025 rows (one row per base-species Pokemon), 32 columns
(after one-hot encoding secondary_type), saved to `data/clean_data.csv`.
