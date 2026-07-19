# Task 3 Notes: Model Selection

## Setup
- **Features**: height, weight, base_experience, hp, attack, defense,
  special_attack, special_defense, speed (stats only). `secondary_type`
  and `n_types` were deliberately excluded even though they exist in
  `clean_data.csv` -- they describe a Pokemon's typing directly, which
  is what we're predicting, so including them would leak target
  information rather than genuinely test whether stats alone predict type.
- **Target**: primary_type (18 classes), label-encoded.
- **Split**: 80/20 train/test, stratified on the encoded target so
  every class (including the rare ones like Flying, 9 total) appears
  in both splits.
- **Scaling**: StandardScaler, fit on train only, applied to both.

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.215 | 0.133 | 0.148 | 0.135 |
| Random Forest | 0.215 | 0.140 | 0.152 | 0.138 |

Macro-averaged metrics were used as the primary comparison (not
accuracy alone), since the classes are imbalanced (Water: 160,
Flying: 9) and accuracy alone would overstate performance by rewarding
correct guesses on the common classes only.

## Why these numbers are low, and that's expected
With 18 classes, random guessing would score ~5.5% accuracy; both
models land around 21.5%, roughly 4x better than chance. Pokemon
typing is a design/lore decision, not a deterministic function of base
stats -- a Fire-type and a Fighting-type can have near-identical stat
profiles. The goal of this task is a working, honestly-evaluated
pipeline, not a high score.

## Which model I'd deploy
**Random Forest**, though the two are close. It beats Logistic
Regression on every metric (accuracy tied, macro-F1 slightly higher),
and it also handles the non-linear, overlapping relationships between
stats and type more naturally than a linear model -- e.g. it can
represent "high attack AND low defense AND high speed" as a combined
signal rather than only linear weighted sums of each stat. Feature
importances were fairly evenly spread (weight: 0.137, special_attack:
0.120, base_experience: 0.114 were the top three), suggesting no single
stat dominates the (limited) predictive signal, which is consistent
with typing not being reducible to one or two stats.

## Artifacts saved
`model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `model_metrics.json`
(the last containing both models' metrics, the confusion matrix, class
labels, feature importances, and the feature list) -- all needed by
the Streamlit dashboard in Task 4.
