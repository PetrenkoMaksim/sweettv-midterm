# Hand-check sheet (raw inputs; Maksim computes the results himself)

## (i) AP@5 of the hybrid (fold A, out-of-fold) for user_id 5024631139882254261

Airings watched >= 80 % per show in weeks 21–30 (the truth rule counts these): {12002955: 229, 12002896: 95, 12002805: 33, 12002172: 25, 12002806: 24, 12001682: 16, 240081: 12, 200469: 8}

Truth (top-5 by count, tie-break by smaller id): [12002955, 12002896, 12002805, 12002172, 12002806]
Prediction (rank 1..5): [12002955, 12002896, 12001682, 12002856, 12002172]

Rule (tools/evaluate.py apk): walk the list; at each hit at rank i add (hits so far)/i; divide by min(|truth|, 5).

- rank 1: 12002955 hit -> +1/1
- rank 2: 12002896 hit -> +2/2
- rank 3: 12001682 miss
- rank 4: 12002856 miss
- rank 5: 12002172 hit -> +3/5

AP@5 = 2.6000 / 5 = 0.5200 (code: 0.5200)

## (ii) One view record and the labelled airings it matched (from src/prep.py)

View record (row 3416555 of the views file): user_id=1298154713044215330, vsetv_id=1495, start_time=2020-05-04 20:59:15, stop_time=2020-05-04 22:04:50, duraton=3935

| tv_show_id | title | airing start | airing end | duration s | overlap = min(ends) − max(starts) |
|---|---|---|---|---|---|
| 12001682 | М/с 'Губка Боб Квадратные Штаны', 136 с. 'День без слез / Летняя работа'. | 2020-05-04 21:00:00 | 2020-05-04 21:20:00 | 1200 | 1200 s (frac of airing 1.000) |
| 12001682 | М/с 'Губка Боб Квадратные Штаны', 137 с. 'Обед из одного блюда / Гэри влюбился'. | 2020-05-04 21:20:00 | 2020-05-04 21:45:00 | 1500 | 1500 s (frac of airing 1.000) |
| 12001697 | М/с 'Аватар: Легенда об Аанге', 39 с. 'Гуру'. | 2020-05-04 21:45:00 | 2020-05-04 22:05:00 | 1200 | 1190 s (frac of airing 0.992) |


Arithmetic: overlap = min(view stop, airing end) − max(view start, airing start); frac = overlap / airing duration.

## (iii) Re-run the fold-A table

```
python src/prep.py      # ~1–2 min, builds cache/
python src/run.py validate   # appends a new block of rows to results/attempts.csv; compare with the existing rows
```
