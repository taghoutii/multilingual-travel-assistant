# Annotation guide

How test queries are written, labeled and checked. The same rules apply to the training data, so
that labels mean the same thing everywhere.

## Files

| File | Source value | Content |
|---|---|---|
| `data/test/generated_validated.jsonl` | `generated_validated` | main test set; every label checked and corrected by hand |
| `data/test/human.jsonl` (optional) | `human` | queries written by hand |
| `data/test/massive_off_topic.jsonl` | `massive` | off-topic queries from MASSIVE, built by `scripts/fetch_massive.py`; never edited |

Test files are not committed. Never edit `massive_off_topic.jsonl` by hand.

## Record format

One JSON object per line:

```json
{"id": "gv-B1-01", "text": "Bonjour, je cherche un vol Paris → Lisbonne pour le 12 janvier, on sera deux.", "lang": "fr", "in_scope": true, "label": {"requests": [{"subcategory": "flight_search", "entities": {"origin": "Paris", "destination": "Lisbonne", "date_start": "12 janvier", "travelers": 2}}]}, "source": "generated_validated"}
```

- `id`: unique across all files (`gv-B<batch>-<nn>` for `generated_validated`, `h-<nnn>` for `human`).
- `lang`: `fr`, `en` or `ar` (the main language of the message).
- `in_scope`: `true` for travel requests covered by the 15 subcategories, `false` otherwise.
- `label`: `{"requests": [...]}` with 1 to 3 requests in the order they appear in the text, or
  `null` when `in_scope` is `false`.
- `meta` (optional): `{"out_of_catalog": true}` on lines that mention a city outside the catalog.

Empty template for a new line:

```json
{"id": "", "text": "", "lang": "", "in_scope": true, "label": {"requests": [{"subcategory": "", "entities": {}}]}, "source": "human"}
```

## Subcategories and slots

Defined in `configs/taxonomy.yaml` (15 subcategories, fixed). An entity field is allowed in a
request only if it is a required or optional slot of that request's subcategory.

| Subcategory | Required slots | Optional slots |
|---|---|---|
| `flight_search` | origin, destination, date_start | date_end, travelers |
| `flight_baggage` | — | — |
| `flight_change_cancel` | — | — |
| `hotel_search` | destination, date_start, date_end | travelers, budget_amount, budget_currency |
| `hotel_services` | — | — |
| `hotel_cancellation` | — | — |
| `weather` | destination | date_start |
| `food` | destination | — |
| `attractions` | destination | — |
| `currency_exchange` | currency_from, currency_to | amount |
| `visa_documents` | — | destination (city or country) |
| `travel_insurance` | — | — |
| `payment_info` | — | — |
| `booking_process` | — | — |
| `contact_human` | — | — |

A request that misses a required slot is still labeled with what is there (for example a flight
with no date); the assistant asks for the rest.

## Labeling rules

1. **Copy as written.** Places, dates and currency words are copied from the message in the same
   language, script and spelling: "Lisbonne", "باريس", "Nice" inside an Arabic sentence.
2. **Typos stay.** "edinburg" is labeled `"edinburg"`; normalization matches it to Edinburgh later.
3. **Only slots, no empty fields.** Leave out any field the message does not give, and any field
   that is not a slot of the subcategory ("insurance for a trip to Jordan" → `travel_insurance`
   with `{}`).
4. **No leading prepositions or articles.** "le 12 janvier" → `12 janvier`, "يوم 20 ديسمبر" →
   `20 ديسمبر`, "au Caire" → `Caire`, "to the United States" → `United States`.
5. **Shared information is copied.** "A flight to Rome and a hotel there" → both requests get
   `"destination": "Rome"`. A hotel booked "for 2 nights" after a flight "on 3 March" gets
   `"date_start": "3 March"` and `"date_end": "2 nights"`.
6. **Ranges sharing a month: full date at each end.** "du 20 au 25 décembre" → `20 décembre` and
   `25 décembre`; "March 3 to 7" → `March 3` and `March 7`.
7. **Numbers become numbers.** "deux personnes" → `"travelers": 2`, "cinq cents dinars" →
   `"amount": 500`, "ألف درهم" → `"amount": 1000`.
8. **Durations go into `date_end`** as written, without the preposition: "pour une semaine" →
   `une semaine`, "لمدة ثلاث ليال" → `ثلاث ليال`, "for 5 nights" → `5 nights`.
9. **Currency words keep their qualifier**: "dinars algériens", "درهم مغربي", "Canadian dollars".
   Without a qualifier, ambiguous words get a default currency during normalization
   (see below).
10. **Corrections: label the final value only.** "a flight to Madrid... sorry, Barcelona" →
    `Barcelona`; "not this weekend, next weekend" → `next weekend`. When the correction is
    elliptical ("not this weekend, the next one"), write the shortest complete phrase from the
    message: `next weekend`.
11. **Other places are not labeled.** "I live in Paris but I want the weather in Rome" → only
    `"destination": "Rome"`.
12. **Out-of-catalog cities** are labeled as written, and the line gets
    `"meta": {"out_of_catalog": true}`.

## How dates and currencies are normalized

Gold labels keep the raw text; code turns them into values (reference date: the eval reference
date in `configs/settings.yaml`). `scripts/validate_data.py` reports every gold date span that
does not normalize, so it can be rephrased or checked.

- A weekday, "this <weekday>" or "next <weekday>" is the first such day after the reference date.
- "this weekend" is the coming Saturday; "next weekend" is one week later.
- "next week" is +7 days; "next month" is the first day of next month.
- "early / mid / end of <month>" is the 1st / 15th / last day; a month alone is the 1st.
- Numeric dates are day-first in French and Arabic (12/01 = 12 January), month-first in English.
- Durations: N nights or N days → end = start + N days; N weeks → 7N days.
- Ambiguous currency words without a qualifier: dinar → TND, dollar → USD, pound / livre → GBP,
  جنيه → EGP, dirham → MAD, riyal → SAR, franc → CHF, ليرة → TRY (`configs/currencies.yaml`).

## Checking a batch

For every line:

1. Read the text. Is it a realistic message? Is `lang` right?
2. Are the subcategories right, one request per distinct ask, in order? Nothing outside the 15?
3. Is every entity present in (or copied from) the text, following rules 1–12? Anything missing?
4. Numbers as numbers? No extra fields?
5. Lines with a city outside the catalog carry `"meta": {"out_of_catalog": true}`.

Then run:

```bash
uv run --group ml python scripts/validate_data.py
```

and fix what it reports in the test files: schema errors, unknown subcategories, places not in the
catalog, currency words not recognized, date spans that do not normalize, and duplicates.
