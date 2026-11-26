# Language ID and scope gate: val

Rates with Wilson 95% intervals. Validation = data/synthetic/val.jsonl (in scope) + data/synthetic/off_topic.jsonl; both were used to choose the gate's threshold and hyper-parameters, so these are tuning numbers.

Records: 786 (286 in scope, 500 off topic).

## Language ID

fastText `lid.176.ftz`, unsupported if another language has p >= 0.85.

Accuracy (supported-language messages): 99.7% [99.1, 99.9] (784/786)

Wrongly flagged as unsupported: 0.0% [0.0, 0.5] (0/786)

**Accuracy per language**

| group | rate [95% CI] (hits/n) |
|---|---|
| ar | 100.0% [98.6, 100.0] (263/263) |
| en | 100.0% [98.5, 100.0] (261/261) |
| fr | 99.2% [97.3, 99.8] (260/262) |

**Accuracy per style**

| group | rate [95% CI] (hits/n) |
|---|---|
| clean | 100.0% [99.3, 100.0] (551/551) |
| noisy | 100.0% [98.3, 100.0] (220/220) |
| telegraphic | 86.7% [62.1, 96.3] (13/15) |

**Accuracy per decision rule**

| rule | rate [95% CI] (hits/n) |
|---|---|
| best_supported | 100.0% [72.2, 100.0] (10/10) |
| fallback | 66.7% [20.8, 93.9] (2/3) |
| top | 99.9% [99.3, 100.0] (772/773) |

Other-language probe (18 messages in es/de/it/pt/tr/nl/ru/fa/ur/zh/ja): detected as unsupported 100.0% [82.4, 100.0] (18/18)

Errors:

| gold | predicted | fastText top | text |
|---|---|---|---|
| fr | en | en 0.36 | visa tokyo |
| fr | en | sv 0.00 | conesiller svp |

## Scope gate

One-Class SVM (nu=0.01, gamma=8.0) on `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings of 1646 in-scope training queries; threshold -0.0148 (target in-scope acceptance 0.97).

In-scope accepted: 97.2% [94.6, 98.6] (278/286)

Off-topic rejected: 89.4% [86.4, 91.8] (447/500)

**In-scope accepted per language**

| group | rate [95% CI] (hits/n) |
|---|---|
| ar | 100.0% [96.2, 100.0] (96/96) |
| en | 94.7% [88.3, 97.7] (90/95) |
| fr | 96.8% [91.1, 98.9] (92/95) |

**In-scope accepted per request count**

| requests | rate [95% CI] (hits/n) |
|---|---|
| 1 | 97.8% [94.9, 99.0] (220/225) |
| 2 | 95.7% [85.8, 98.8] (45/47) |
| 3 | 92.9% [68.5, 98.7] (13/14) |

**In-scope accepted per style**

| group | rate [95% CI] (hits/n) |
|---|---|
| clean | 100.0% [98.2, 100.0] (206/206) |
| noisy | 92.3% [83.2, 96.7] (60/65) |
| telegraphic | 80.0% [54.8, 93.0] (12/15) |

**Off-topic rejected per language**

| group | rate [95% CI] (hits/n) |
|---|---|
| ar | 85.6% [79.5, 90.1] (143/167) |
| en | 93.4% [88.5, 96.3] (155/166) |
| fr | 89.2% [83.6, 93.1] (149/167) |

**Off-topic rejected per kind**

| kind | rate [95% CI] (hits/n) |
|---|---|
| general | 95.8% [93.5, 97.3] (430/449) |
| near_miss | 33.3% [22.0, 47.0] (17/51) |

In-scope rejected (lowest scores):

| score | lang | text |
|---|---|---|
| -0.0151 | en | traditional diishes in djerba i shouldnt miss and which mseums r worth it in djerba |
| -0.0150 | fr | conesiller svp |
| -0.0150 | en | rate baht dirhams |
| -0.0149 | en | what should i eat in lisbon? also, which museums r worth it in lisbon? |
| -0.0149 | fr | il me faut un aller simple de munich a new york vendredi pour 5 personnes + la mete la-bas a cette date + que fair a new york avec des enfants ? |
| -0.0148 | en | carry on weight |
| -0.0148 | en | im going to phuket any must see places thx in advance |
| -0.0148 | fr | quels monuments visiter a londres |

Off-topic accepted (highest scores):

| score | lang | text |
|---|---|---|
| -0.0092 | ar | مساء الخير، هل هناك رحلات بحرية من مدريد؟ شكرا |
| -0.0116 | ar | لو سمحت، هل هناك رحلات بحرية من تونس؟ |
| -0.0122 | ar | هل هناك رحلات بحرية من مدريد |
| -0.0127 | fr | Un taxi pour l'aéroport de Lisbonne, c'est combien ? |
| -0.0129 | en | Hi, any cruises leaving from Tokyo? |
| -0.0130 | en | Hi there! How much is a taxi to Berlin airport? |
| -0.0131 | ar | من فضلك، أريد استئجار سيارة في طوكيو شكرا |
| -0.0132 | en | any cruises leaving from madrid |
| -0.0133 | ar | أريد استئجار سيارة في برلين |
| -0.0134 | en | im looknig for a bus to istanbul |
| -0.0134 | ar | من فضلك، أبحث عن حافلة إلى دبي |
| -0.0134 | ar | مواعيد القطارات إلى مدريد؟ |
| -0.0134 | ar | كم سعر التاكسي إلى مطار إسطنبول؟ |
| -0.0136 | en | I'm looking for a bus to Tunis |
| -0.0137 | fr | je cherche un bus pour berlin |

## Latency (CPU, one message at a time)

| stage | p50 ms | p95 ms |
|---|---|---|
| language ID | 0.05 | 0.10 |
| scope gate (embedding + SVM) | 103.6 | 176.2 |
