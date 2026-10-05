# V5-equal: причины редких сигналов на MEXC

Повторная диагностика той же зафиксированной модели только на session confirmation closes train и validation BTC/ETH. Префиксы, history, session clock и hashes взяты из зарегистрированного плана. Сделки, PnL, relaxed rules и новые варианты не рассчитывались; signed код не менялся; final этим helper не читается.

Это гистограмма одного конечного `reason`, возвращённого анализатором на каждом закрытии. Для ранних контекстных checks это первая причина выхода; внутри перебора raid последний сохранённый failure может описывать последнего проверенного кандидата. Причины зависят от порядка и совместного выполнения условий. Их нельзя трактовать как независимый вклад каждого фильтра либо оценку эффекта его удаления.

| Режим / окно | Session closes, BTC+ETH | Accepted outputs | Уникальные symbol+ID |
|---|---:|---:|---:|
| intraday / train | 12600 | 1 | 1 |
| intraday / validation | 7560 | 0 | 0 |

## intraday / train

| Возвращённая причина | Закрытий | Доля |
|---|---:|---:|
| no_v1_daily_origin | 7182 | 57.00% |
| no_v1_context_trend | 2898 | 23.00% |
| h1_context_misaligned | 1542 | 12.24% |
| no_valid_liquidity_raid | 506 | 4.02% |
| no_unhit_daily_target | 420 | 3.33% |
| insufficient_whole_manipulation_recovery | 49 | 0.39% |
| no_direct_engulfment_or_inversion | 2 | 0.02% |
| accepted | 1 | 0.01% |

## intraday / validation

| Возвращённая причина | Закрытий | Доля |
|---|---:|---:|
| no_v1_daily_origin | 3696 | 48.89% |
| no_v1_context_trend | 2352 | 31.11% |
| h1_context_misaligned | 767 | 10.15% |
| no_valid_liquidity_raid | 471 | 6.23% |
| no_unhit_daily_target | 259 | 3.43% |
| insufficient_whole_manipulation_recovery | 9 | 0.12% |
| no_direct_engulfment_or_inversion | 6 | 0.08% |

Числа не подтверждают прибыль и не означают, что редкое условие следует ослабить. Они описывают текущую формулу: V1 численный контекст и historical origin, совпадение H1, свежую B/TP, causal liquidity event и OR-модель с третью полного диапазона. Авторский ручной выбор ноги и начала коррекции остаётся прокси.

Accepted outputs могут повторяться на нескольких закрытиях одного raid; уникальные symbol+ID показывают число идей до admission. Это не число заполненных/закрытых позиций общего счёта.

План SHA256: 07053a68410e3ded5cdd33f021fd37c0a0b4d6e00c89068522f2dc89c9b464bf. Подробные причины по каждому символу, границы префиксов и hashes находятся в strategy_v5_signal_funnel.json.
