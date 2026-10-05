# V6 — завершённая проверка текущего A→B

**Устойчивое преимущество не подтверждено.** V6 на MEXC дала одну закрытую убыточную сделку (−4.94 USDT); в validation/final сделок нет. На известном Binance-дне есть один TP, на внешнем 2021 закрытых сделок нет. Ни одна из двух моделей не прошла заранее заданные критерии.

Выполнены все 28 заранее фиксированных сравнений V6/V5 equal. Правила после просмотра результатов не менялись. Каждый случай начинает с отдельного общего для его пар счёта 1000 USDT; таблица показывает результат счёта после комиссии, проскальзывания и funding. Плечо не умножает эти цифры повторно.

V6 заменяет pre-A классический тренд на снятие/реакцию и текущую защищённую A→B. H1 направлен к D1 B, начало манипуляции определяется на M5 отдельно от H1 TP. Сохранены entry engulfment OR FVG, gross2R, структурный SL и финансовый движок V5. Численные допущения перечислены в strategy_v6.md.

| Блок | Этап | Модель | Baseline: закрытых / TP | Baseline PnL USDT | Stress: закрытых / TP | Stress PnL USDT |
|---|---|---|---:|---:|---:|---:|
| MEXC 5 пар / фактическая M5 | train | v6-current-leg | 1 / 0 | -4.9439 | 0 / 0 | +0.0000 |
| MEXC 5 пар / фактическая M5 | train | v5-equal-control | 1 / 0 | -4.9041 | 0 / 0 | +0.0000 |
| MEXC 5 пар / фактическая M5 | validation | v6-current-leg | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| MEXC 5 пар / фактическая M5 | validation | v5-equal-control | 2 / 1 | +2.1094 | 0 / 0 | +0.0000 |
| MEXC 5 пар / фактическая M5 | final | v6-current-leg | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| MEXC 5 пар / фактическая M5 | final | v5-equal-control | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| Binance BTC/ETH 2024–2025 / M1 | train | v6-current-leg | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| Binance BTC/ETH 2024–2025 / M1 | train | v5-equal-control | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| Binance BTC/ETH 2024–2025 / M1 | validation | v6-current-leg | 1 / 1 | +7.9884 | 1 / 1 | +6.2622 |
| Binance BTC/ETH 2024–2025 / M1 | validation | v5-equal-control | 2 / 1 | +2.9355 | 1 / 1 | +6.2467 |
| Binance BTC/ETH 2024–2025 / M1 | final | v6-current-leg | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| Binance BTC/ETH 2024–2025 / M1 | final | v5-equal-control | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| Binance BTC/ETH внешний 2021 / M1 | external | v6-current-leg | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |
| Binance BTC/ETH внешний 2021 / M1 | external | v5-equal-control | 0 / 0 | +0.0000 | 0 / 0 | +0.0000 |

## Причины отклонения и число сигналов

Ниже сессионные подтверждения V6 до финансового допуска. Несколько подтверждений одного raid могут соответствовать одной идее; это не независимые сделки.

| Блок/этап | Проверок | Нет D1 A→B | H1 против D1 | Принятых подтверждений | Уникальных идей | Baseline допущено | Stress отказов по издержкам/контракту |
|---|---:|---:|---:|---:|---:|---:|---:|
| mexc_five/train | 18900 | 13482 | 3394 | 3 | 2 | 2 | 3 |
| binance_known/train | 30744 | 23520 | 4702 | 4 | 2 | 1 | 4 |
| mexc_five/validation | 9450 | 4074 | 3675 | 0 | 0 | 0 | 0 |
| binance_known/validation | 15204 | 9744 | 3705 | 2 | 1 | 1 | 0 |
| mexc_five/final | 8820 | 6804 | 1547 | 0 | 0 | 0 | 0 |
| binance_known/final | 15456 | 11172 | 2777 | 4 | 1 | 0 | 4 |
| binance_2021/external | 30660 | 23268 | 4551 | 0 | 0 | 0 | 0 |

Частота отбора и экономический допуск — разные ограничения. Уменьшение стопа повышает отношение комиссии к R; большее плечо при фиксированном cash риске не устраняет это. В полном JSON сохранена каждая причина отказа.

## Статистическая проверка и win rate

- MEXC 5 пар / фактическая M5, V6 baseline: закрытых сделок 1, TP 0, нетто-побед 0; win rate 0.00%; 95% интервал Wilson 0.00–79.35%. Известные folds объединены только для описания числа исходов; это отдельные счета.
- Binance BTC/ETH 2024–2025 / M1, V6 baseline: закрытых сделок 1, TP 1, нетто-побед 1; win rate 100.00%; 95% интервал Wilson 20.65–100.00%. Известные folds объединены только для описания числа исходов; это отдельные счета.
- Binance BTC/ETH внешний 2021 / M1, V6 baseline: закрытых сделок 0, TP 0, нетто-побед 0; win rate не определён (нет закрытых сделок); 95% интервал Wilson не определён. Известные folds объединены только для описания числа исходов; это отдельные счета.

## Фиксированные критерии

- mexc_five/v6-current-leg: критерии не пройдены; baseline_total_fewer_than_30_closed, train_baseline_nonpositive, validation_baseline_nonpositive, validation_fewer_than_15_closed, final_baseline_nonpositive, final_fewer_than_15_closed, validation_stress_nonpositive, final_stress_nonpositive
- mexc_five/v5-equal-control: критерии не пройдены; baseline_total_fewer_than_30_closed, train_baseline_nonpositive, validation_fewer_than_15_closed, final_baseline_nonpositive, final_fewer_than_15_closed, validation_stress_nonpositive, final_stress_nonpositive
- binance_known/v6-current-leg: критерии не пройдены; baseline_total_fewer_than_30_closed, train_baseline_nonpositive, validation_fewer_than_15_closed, final_baseline_nonpositive, final_fewer_than_15_closed, final_stress_nonpositive
- binance_known/v5-equal-control: критерии не пройдены; baseline_total_fewer_than_30_closed, train_baseline_nonpositive, validation_fewer_than_15_closed, final_baseline_nonpositive, final_fewer_than_15_closed, final_stress_nonpositive
- binance_2021/v6-current-leg: критерии не пройдены; baseline_nonpositive, stress_nonpositive, baseline_fewer_than_30_closed
- binance_2021/v5-equal-control: критерии не пройдены; baseline_nonpositive, stress_nonpositive, baseline_fewer_than_30_closed

Положительная сумма по отдельному окну не означает устойчивого преимущества. Требуются baseline train/validation/final >0, stress validation/final >0, ≥30 закрытых всего и ≥15 в validation и final, flat inventory. Внешний 2021 требует положительных обеих издержек и ≥30 baseline закрытых. На нуле сделок 70% не измеряется; повторяющиеся подтверждения одного raid не являются независимыми сделками. Все 28 исходов, включая нули/убытки, сохранены.

## Разметка и соответствие источнику

Визуально проверены четыре графика до новой прибыли: два известных реальных контекста и два синтетических полных LONG/SHORT контракта. BTC 24.06.2025: D1 LONG A98115.4 и B106486.2, H1 LONG к D1 B без обязательного локального H1 B. Старый V5 пример 12.01.2026: D1 SHORT/H1 LONG; V6 обязан отклонять, независимо от известного прибыльного результата V5. Графики и записи source review включены в архив.
После результатов дополнительно проверены первые хронологические закрытые V6 baseline идеи каждого известного блока: DOGE MEXC 15.04.2026 и ETH Binance 24.06.2025. Их eventID соответствует исполненному сигналу; уровень/начало M5/цель были известны до первого снятия, восстановление≥1/3, добавление будущих экстремальных свечей не изменяет ни признаки, ни сигнал. Эти две иллюстрации выбраны после результатов, включая проигравшую, и являются описательной проверкой; правила не менялись. Графики используют только предсигнальные свечи.
Прибыльная V6 validation идея ETH также приходится на 24 июня 2025 — день публичного BTC примера, использованного для калибровки контекста. Коррелированная сделка другой пары не превращает этот известный день в независимый holdout.
Это ограниченная калибровка, не независимая ручная выборка десятков авторских входов. Точное начало визуальной манипуляции и критерий HTF реакции автор не задаёт исчерпывающе. Наш последний M5 pivot, применение engulfment/FVG к HTF и ожидание закрытой D1 для продолжения за B остаются прокси; совпадение всех ручных решений автора и заявленные 70% не доказаны. Синтетические TF не являются агрегацией одного реального ряда.

## Данные, издержки и история экспериментов

MEXC: пятипарное окно 10.04–04.10.2026, 90/45/42.0104 дней. Binance известный блок:2024/2025H1/2025H2. Эти цены и корзина уже исследовались; pristine holdout не утверждается. Отдельный календарный 2021 выбран до скачивания/просмотра прибыли:72 официальных ZIP проверены по CHECKSUM, на каждую пару 525600 реальных тестовых M1 и 1095 событий funding; M1 прогрев декабрь 2020, D1 январь–ноябрь 2020. Это другая площадка и прошлый режим, результаты не переносятся на MEXC. [Официальные архивы Binance](https://github.com/binance/binance-public-data).
Baseline: комиссия 5 bps + проскальзывание 2 bps на сторону; stress: 10 + 5 bps. Fee/slippage списываются при fill/exit по соответствующему номиналу. Funding фактический своей площадки, цена его расчёта — closed-price proxy. Плечо 10–50 выбирается по активу и дистанции до структурного стопа/ликвидационного proxy; плановый cash риск≤0.5%, SL≤1.5%, netRR≥1.25, cost/R≤1/3. Gap/funding могут превысить плановый риск. Современные публичные tiers MEXC — историческое допущение; BinanceMM1% — явный proxy. Реальный fair/mark price и ликвидации не реконструированы; M5 fills имеют грубое разрешение.
ДоV6 было 698 отдельных расчётов и отдельно 2 старых shared-wallet replay; теперь 726. Исходники/результатыV1–V5 сохранены без замены. Первая регистрация была отклонена guard до сигналов/PnL из-за декодирования текста в Windows китайских имён контрактов; исходный план сохранён, чтение исправлено на UTF-8 и выполнена новая регистрация с теми же правилами. Это не изменение стратегии после результатов.

## Воспроизводимость и поставка

Итоговый план SHA256 `fe0728ebf6b986f1b9a3d438abe9456d9b6e5ae1bd0309ab8414ed3df4bcefc3`. Полный JSON включает каждый trade/daily equity и signal funnel; CSV содержит все 28 строк. Независимая арифметика проверяет 28 случаев, их fees/funding/risk/margin и signed input hashes. Состояния final/external открытия привязаны к завершённым предыдущим этапам.
В репозитории strategy_v6_evidence.json — компактный индекс с SHA полного JSON; полный strategy_v6_evidence.json, причинные графики, планы, проверки и открытые маркеры находятся в strategy_v6_results.zip. Внешние CSV и официальные checksum — research_v6_external_data.zip, известные данные — прежние research_binance_data.zip и research_v5_robustness_data.zip. Исходники — mexc_volium_bot.zip; V5 отдельно research_v5_source.zip.
Запуск paper: установить requirements.txt, скопировать .env.example в .env, python -m app. При существующем .env явно поставить VOLIUM_STRATEGY_PROFILE=v6_current_leg и VOLIUM_MODE=intraday. Ключи для paper и backtest не нужны. Бэктест не включает live.
