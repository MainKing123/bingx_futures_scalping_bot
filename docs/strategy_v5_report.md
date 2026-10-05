# V5: фиксированная проверка стратегии и исполнения

**Прибыльное преимущество не подтверждено:** ни одна новая модель не выполнила заранее заданные условия по числу сделок и положительным проверочным периодам.

Выполнены все 57 заранее заданных trials и отдельно шесть описательных пяти парных проверок, объявленных после основных исходов. Прежние 635 trials и два отдельных общего портфеля сохранены; всего 63 новых и 698 накопленных stage trials. Повторённые модели и пересекающиеся окна не являются независимыми выборками.

Соседний `research_report.md` сохранён как исторический отчёт V2–V4. Указанные в нём 635 trials, runtime paper V1 и ещё не открытые final относятся к прежнему снимку работы; его описания «текущего» проекта не описывают новую поставку V5. Актуальный статус V5 приведён в этом отчёте, исторический файл не переписывается.

На MEXC за 330 дней обе новые версии дали одну TP-сделку в train: +7,78 USDT. В validation сделок нет; финальная сырая идея была отклонена фильтром net reward/risk после издержек. Известная 30-дневная проверка пяти пар дала в scalp две сделки, одну TP: +2,10 USDT; при повышенных издержках обе идеи отклонены. Эти редкие положительные эпизоды не подтверждают преимущество.

Две версии оставили старый численный D1/H1 контекст: поиск наличия origin-sweep в истории не задаёт однозначно последнюю авторскую ногу A→B. Восстановление на одну треть всей манипуляции привязано к заранее известному структурному TP — нашей причинной опорной точке; точный алгоритм начала авторской манипуляции не доказан. Юнит-тесты проверяют выбранные формулы, а не точное совпадение с ручной торговлей автора.

Каждый режим и fold отдельно начинает с общего счёта 1000 USDT. Прибыль разных рынков и folds не складывается в доходность одной работающей системы. Стресс: модельная комиссия 10 bps и slippage 5 bps на сторону; baseline 5 + 2 bps. Ставки и времена funding фактические опубликованные, денежная сумма рассчитывается по свечному ценовому proxy.

Повышенные издержки меняют admission и номинал позиции по заранее заданному фильтру. Поэтому stress иногда даёт меньший набор входов и даже больший PnL; это не сравнение только комиссии на неизменном наборе сделок и не подбор по исходу.

| Площадка | Окно | Режим / рынок | Модель | Baseline: сделки / TP / PnL | Stress: сделки / TP / PnL |
|---|---|---|---|---:|---:|
| mexc | train | intraday / crypto_core | v5-strict-cost | 1 / 1 / +7.784 | не объявлен |
| mexc | train | intraday / crypto_core | v5-equal-cost | 1 / 1 / +7.784 | не объявлен |
| mexc | train | intraday / crypto_core | v1-legacy3x | 4 / 2 / +0.092 | не объявлен |
| mexc | validation | intraday / crypto_core | v5-strict-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| mexc | validation | intraday / crypto_core | v5-equal-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| mexc | validation | intraday / crypto_core | v1-legacy3x | 4 / 1 / -10.713 | 4 / 1 / -17.120 |
| mexc | final | intraday / crypto_core | v5-strict-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| mexc | final | intraday / crypto_core | v5-equal-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| mexc | final | intraday / crypto_core | v1-legacy3x | 2 / 1 / +1.262 | 2 / 1 / -2.946 |
| mexc | robustness | intraday / core_plus_top3 | v5-strict-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| mexc | robustness | intraday / core_plus_top3 | v5-equal-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| mexc | robustness | intraday / core_plus_top3 | v1-legacy3x | 3 / 0 / -20.367 | 3 / 0 / -26.562 |
| mexc | robustness | scalp / core_plus_top3 | v5-strict-cost | 2 / 1 / +2.103 | 0 / 0 / +0.000 |
| mexc | robustness | scalp / core_plus_top3 | v5-equal-cost | 2 / 1 / +2.103 | 0 / 0 / +0.000 |
| mexc | robustness | scalp / core_plus_top3 | v1-legacy3x | 6 / 2 / -10.938 | 6 / 2 / -24.173 |
| binance_usdm | train | intraday / crypto_core | v5-strict-cost | 0 / 0 / +0.000 | не объявлен |
| binance_usdm | train | intraday / crypto_core | v5-equal-cost | 0 / 0 / +0.000 | не объявлен |
| binance_usdm | train | intraday / crypto_core | v1-legacy3x | 3 / 1 / -4.170 | не объявлен |
| binance_usdm | train | scalp / crypto_core | v5-strict-cost | 1 / 0 / -4.991 | не объявлен |
| binance_usdm | train | scalp / crypto_core | v5-equal-cost | 1 / 0 / -4.991 | не объявлен |
| binance_usdm | train | scalp / crypto_core | v1-legacy3x | 12 / 6 / -5.529 | не объявлен |
| binance_usdm | validation | intraday / crypto_core | v5-strict-cost | 2 / 1 / +2.936 | 1 / 1 / +6.247 |
| binance_usdm | validation | intraday / crypto_core | v5-equal-cost | 2 / 1 / +2.936 | 1 / 1 / +6.247 |
| binance_usdm | validation | intraday / crypto_core | v1-legacy3x | 4 / 0 / -27.653 | 4 / 0 / -37.556 |
| binance_usdm | validation | scalp / crypto_core | v5-strict-cost | 3 / 0 / -14.853 | 0 / 0 / +0.000 |
| binance_usdm | validation | scalp / crypto_core | v5-equal-cost | 3 / 0 / -14.853 | 0 / 0 / +0.000 |
| binance_usdm | validation | scalp / crypto_core | v1-legacy3x | 16 / 3 / -83.401 | 16 / 3 / -135.110 |
| binance_usdm | final | intraday / crypto_core | v5-strict-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| binance_usdm | final | intraday / crypto_core | v5-equal-cost | 0 / 0 / +0.000 | 0 / 0 / +0.000 |
| binance_usdm | final | intraday / crypto_core | v1-legacy3x | 2 / 1 / -1.369 | 2 / 1 / -8.472 |
| binance_usdm | final | scalp / crypto_core | v5-strict-cost | 1 / 1 / +6.582 | 0 / 0 / +0.000 |
| binance_usdm | final | scalp / crypto_core | v5-equal-cost | 1 / 1 / +6.582 | 0 / 0 / +0.000 |
| binance_usdm | final | scalp / crypto_core | v1-legacy3x | 8 / 3 / -14.129 | 8 / 3 / -37.972 |
| mexc | post_hoc_power_supplement | intraday / core_plus_top3 | v5-strict-cost | 3 / 1 / -2.795 | 0 / 0 / +0.000 |
| mexc | post_hoc_power_supplement | intraday / core_plus_top3 | v5-equal-cost | 3 / 1 / -2.795 | 0 / 0 / +0.000 |
| mexc | post_hoc_power_supplement | intraday / core_plus_top3 | v1-legacy3x | 11 / 3 / -24.827 | 11 / 3 / -41.318 |

Все планы, code/data/parent/protocol/source-архив SHA повторно проверены после расчёта. Проверены chronology signal ≤ entry ≤ exit ≤ конец fold, полный основной набор 57 заданий и отдельный набор шести. Final opened markers связаны с неизменёнными train/validation SHA.

Дополнительный post_hoc_power_supplement использует BTC/ETH/ZEC/SOL/DOGE на общих реальных M5 с 10 апреля 2026 15:05 UTC до 4 октября 2026 15:20 UTC: 177,0104 дня, 50 979 полных баров каждой пары, пропусков нет. Начало механически отодвинуто для неизменённого warmup 990 M5. Все H1/D1 и funding своей площадки сохранены. Он объявлен после основных результатов по запросу пяти пар, не имеет splits/selection/eligibility и не подтверждает преимущество; известные цены и ретроспективный текущий состав дают bias. Ни дата, ни состав не выбирались по supplement PnL. Бюджет основного плана остаётся 57.

Фиксированный критерий требует положительного baseline отдельно в train, validation и final; положительного stress отдельно в validation/final; минимум 30 закрытых сделок всего и 15 в каждом validation/final; отсутствия незавершённого инвентаря. Малая выборка не заменяет статистическую проверку. Финальный PnL не используется для выбора или продвижения модели.

Редкость сигналов связана прежде всего с неизменённым численным контекстом v1. На MEXC train причины отсутствия daily origin и строгого daily trend возвращены на 80% сессионных закрытий; ещё 12,24% — H1 несогласование. Это гистограмма последовательных диагностических проверок, а не 12 600 независимых торговых возможностей и не оценка эффекта удаления фильтра. Полная причинная воронка сохранена в strategy_v5_signal_funnel.md/json.

Полное совпадение с ручной торговлей автора не подтверждено: старый численный D1/H1 context proxy и алгоритм origin всей манипуляции остаются интерпретациями. Следующий осмысленный исследовательский шаг — заранее зафиксировать D1 A→B на датированных ручных примерах до новой оценки PnL. Этот шаг пока не выполнен; снятие фильтра само по себе не доказывает прибыльное преимущество.

Strict и equal — связанные фиксированные версии с одинаковой финансовой политикой. У legacy3x одновременно отличаются правила и старая финансовая модель; его результат не изолирует эффект одного изменения. Новые комиссии оплачиваются при входе и выходе по соответствующему номиналу, funding своей площадки учтён один раз.

Плечо 10–50 меняет маржу, а не денежный риск счёта. Planned loss до 0,5% капитала включает консервативные издержки, но разрыв цены и funding могут превысить его. Например, единственная новая train-сделка BTC дала +80,81% относительно первоначальной маржи, однако всего +0,778% относительно счёта 1000 USDT. Guard ликвидации является proxy; историческая mark price, очередь лимитов и partial fills не реконструированы.

2R относится к исходной gross structural geometry. После округлений и модельных издержек ожидаемый net reward/risk ниже; для новых моделей отдельно требуется минимум 1,25. Гарантированного net 2R нет.

Текущие контрактные правила MEXC и отдельные Binance MMR 1% / публичная liquidation fee — исторически неизвестные прокси. Binance не получает MEXC tick/lot/tiers. MEXC использует реальные M5 за 330 дней, поскольку старые M1 недоступны; Binance — реальные M1 2024–2025 и funding Binance. Результат Binance не считается доходностью MEXC.

Новые MEXC 150/90/90 folds объявлены до регистрации, но окна частично ранее изучены. Публичный пример BTC 24 июня 2025 внутри Binance validation раскрыт. Известная 30-дневная robustness (BTC/ETH/ZEC/SOL/DOGE) использует обязательный core и три volatile пары из текущего snapshot: selection/survivorship bias остаётся. Robustness исключена из eligibility.

Обе новые версии остаются экспериментальными paper-only. Для подтверждения нужна последующая перспективная проверка заранее неизменённых правил. Авторские 70% и прибыль не обещаются. Все 63 строки с отдельной маркировкой шести описательных cases, funding, комиссии, slippage, Wilson intervals, gross/net PnL, cash/marked DD, margin ROI/leverage, открытые и незаполненные идеи и причины отказов находятся в strategy_v5_evidence.json и strategy_v5_all_trials.csv.


## Полная поставка данных

Этот документ скопирован из итогового отчёта. В репозитории strategy_v5_evidence.json содержит компактный индекс всех63проверок с SHA полного JSON. Полные журналы сделок, денежные события, daily equity и причинные графики находятся в отдельном strategy_v5_results.zip; публичные CSV — в трёх research_*_data.zip. Архив mexc_volium_bot.zip содержит тот же код и документы.
