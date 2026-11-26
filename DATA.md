# Data

All data concerns the fictional **Demo Travel Agency** and a fixed catalog of 32 cities in 16
countries (`configs/destinations.yaml`).

## Training and validation data (synthetic)

The training and validation sets are **synthetic**. They are not committed; they are shared
privately (as a private Kaggle dataset for training).

| File | fr | en | ar | Total |
|---|---|---|---|---|
| `data/synthetic/train.jsonl` | 549 | 549 | 548 | 1646 |
| `data/synthetic/val.jsonl` | 95 | 95 | 96 | 286 |
| `data/synthetic/off_topic.jsonl` (scope-gate threshold tuning only) | 167 | 166 | 167 | 500 |

- Split: 85 / 15, stratified by subcategory × language (single-request) and by language × number
  of requests (multi-request), seeded.
- In-scope records: 1530 single-request, 309 with 2 requests, 93 with 3 requests.
- Style: 30% noisy messages (no accents or punctuation, lowercase,
  abbreviations, telegraphic fragments, casual Arabic spelling), 12% tricky
  ones (a city that is not the destination, corrections); some mixed-language messages; relative and
  absolute dates; missing information.
- All 32 catalog cities appear; no other city is labeled.
- Off-topic set: general non-travel requests, plus 51 travel-adjacent requests outside the
  15 subcategories (car rental, trains, taxis...) that the assistant also declines.

### Requests per subcategory × language (train + val)

| Category | Subcategory | fr | en | ar | Total |
|---|---|---|---|---|---|
| flights | `flight_search` | 103 | 90 | 98 | 291 |
| flights | `flight_baggage` | 49 | 46 | 43 | 138 |
| flights | `flight_change_cancel` | 40 | 43 | 42 | 125 |
| accommodation | `hotel_search` | 86 | 87 | 89 | 262 |
| accommodation | `hotel_services` | 40 | 41 | 43 | 124 |
| accommodation | `hotel_cancellation` | 40 | 39 | 34 | 113 |
| destination | `weather` | 81 | 76 | 77 | 234 |
| destination | `food` | 56 | 57 | 66 | 179 |
| destination | `attractions` | 64 | 58 | 60 | 182 |
| practical | `currency_exchange` | 39 | 39 | 48 | 126 |
| practical | `visa_documents` | 46 | 51 | 48 | 145 |
| practical | `travel_insurance` | 38 | 41 | 45 | 124 |
| agency | `payment_info` | 42 | 47 | 42 | 131 |
| agency | `booking_process` | 41 | 48 | 41 | 130 |
| agency | `contact_human` | 39 | 45 | 39 | 123 |
| | **Total** | 804 | 808 | 815 | 2427 |

## Test set

Not committed. Three sources, always reported separately:

| Source | File | Records | Status |
|---|---|---|---|
| `generated_validated` | `data/test/generated_validated.jsonl` | 0 | every label checked and corrected by the author (in progress) |
| `massive` | `data/test/massive_off_topic.jsonl` | 180 | off-topic queries from MASSIVE, scope gate only |
| `human` | `data/test/human.jsonl` | 0 | optional, written by the author |

Targets for `generated_validated`: about 12 single-request queries per subcategory, about 40 multi-request
queries and a few out-of-catalog cities, balanced across fr / en / ar.

### MASSIVE

Off-topic test queries come from **MASSIVE** (FitzGerald et al., 2022, *MASSIVE: A 1M-Example
Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*),
released by Amazon under the **CC BY 4.0** license (`AmazonScience/massive`, version 1.1).

- Official **test** split, locales fr-FR, en-US and ar-SA.
- Scenarios: alarm, audio, calendar, cooking, email, iot, lists, music, news, play, social, takeaway (12 scenarios).
- 5 utterances per scenario and locale, seeded sampling (`scripts/fetch_massive.py`).
- Used for nothing else (no training, no tuning). The ar-SA utterances are largely Saudi dialect,
  not Modern Standard Arabic.

## Quality checks (`scripts/validate_data.py`)

Schema, known subcategories, slots, catalog places (typos matched by fuzzy matching), currency
words, every gold date span normalizes, exact and near duplicates, and leakage: any synthetic
example with embedding cosine > 0.9 (`paraphrase-multilingual-MiniLM-L12-v2`) to any test example
is removed. Current status: all checks pass.

## Other data (committed)

- `data/policies/`: policy documents of the fictional agency (baggage, changes, hotel rules,
  insurance, payment, booking, general visa guidance pointing to official embassy websites).
- `data/mocks/`: fictional airlines and hotels with illustrative prices (`scripts/build_mocks.py`).
- `data/lookups/`: city and currency lookup tables derived from `configs/destinations.yaml` and
  `configs/currencies.yaml` (`scripts/build_lookups.py`).
