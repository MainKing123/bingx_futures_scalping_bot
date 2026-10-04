# Последняя объявленная контекстная попытка VOLIUM v4

Этот протокол объявляется **до любых v4 train/validation/final исходов**. Новый runner — `app.research_post_sweep`; новый provider — `app.strategy.volium_v4`. Старые v1/v2/v3 providers, портфельный engine, `research_suite.py`, `research_followup.py`, четыре зарегистрированных плана и их результаты сохраняются побайтно. Архив `research_v3_source.zip` и per-file manifest проверяются до новой регистрации. Расширять этот последний контекстный family после просмотра новых исходов в этой исследовательской сессии нельзя.

## Причина и ограничение новой гипотезы

Предыдущие модели требовали HH/HL или LL/LH до снятия A. Причинный разбор публичного авторского примера BTC24июня2025 показал расхождение с этой формализацией: дневное снятие June22 было известно, но тренд до A отсутствовал. Это дата внутри уже использованного Binance validation; причинные цены/признаки и публично известный исход автора просмотрены. Ни replay, ни forward/final для этого source-case не открывались. Новый вариант нельзя объявлять независимым подтверждением или полной репликой ручных правил автора.

Единственная новая интерпретация — `post_sweep_break`: A является последним наблюдаемым закрытым D1 sweep/reclaim ближайшего подтверждённого ещё неснятого уровня, без обязательного тренда до A и без right-pivot для самой A. C допускается **только как непосредственно следующий закрытый D1**: для LONG high>A.high, low>A.low и close>A.high; для SHORT — зеркально high<A.high, low<A.low и close<A.low. Любой другой C отвергается. Новейший A с неверным C не заменяется старым подходящим A. Известные lower-TF A/B-integrity, unique first-raid, фактический общий край события, причинный ATR до первого raid, engulf/V, структурный TP и 2R сохраняются.

Post-A break может соответствовать локальному развороту против глобального тренда автора. Это прямо раскрытый смысловой риск, а не доказанное правило видео. Дополнительных трендовых окон, флагов, монет, фильтров прибыли или новых strength-порогов нет. Source/API/causality проверяются video-agent и root до freeze; источник и точный код provider подписываются в плане.

## Конечная сетка и обе площадки

Только intraday D1→H1→M5. **24 основных варианта**: body/ATR0.8/1.2/1.6 × body/range0.6/0.7 × body/opposite-body1/1.5 × reaction bars2/3. ATR14, correction target `leg_origin`, context `post_sweep_break` фиксированы. Три фиксированных контроля: исходная v1, V3 `latest_leg_daily` default, V3 `latest_leg_local` default. Контроль не eligible для выбора; у него нет strength-сетки. V1 сохраняет раскрытое ограничение повторного raid, V3 сохраняется с его финальным event-fix.

Два новых плана регистрируются **оба до первого v4 train**; runner проверяет наличие и совпадение общего protocol/code/grid/parent ledger, не читая peer outcomes для выбора. MEXC: неизменные `crypto_core` BTC/ETH и `five_fx` BTC/ETH/EUR/GBP/JPY, M5 execution, прежние150дней outcome, train90/validation30/final30. Binance USD-M: только coreBTC/ETH, фактическое M1 execution, весь2024 train, January–June2025 validation, July–December2025 final. Площадка, часы, порядок пар, UTC folds и разрешение берутся из их родительского дизайна побайтно. Binance результат не является MEXC доходностью.

На каждую группу27train; всего **81 train**: MEXC54 и Binance27 (72 основных +9 контролей). Validation: shortlist≤3 плюс три контроля на группу, максимум **18**. Final: только замороженный validation winner плюс три контроля, baseline и stress по одному разу, максимум **24**. Без winner группа выполняет final command с completed0, **без `final_opened` marker**, включая отсутствие финальных контрольных запусков. Ни проигрыш другого рынка, ни outcome другой площадки не меняет второй план. Это последний объявленный source-family, не переход к другой выборке после final.

## Данные, история попыток и экономика

До v4 завершены v2 MEXC62train+2validation, v2 Binance62+2, v3 MEXC102+6, v3 Binance51+3; у всех final0 и final не открывался. Это **290 research-stage trials плюс255 v1 =545 уже просмотренных trials**. Два прежних30-дневных общих v1-портфеля раскрываются отдельно и не складываются как независимые выборки. Заранее известный авторский пример дополнительно раскрывается как source inspection внутри validation. Повторные train/validation, похожие candidates и вложенные портфели зависимы; holdout не гарантирует отсутствие переобучения, PBO/CSCV здесь не вычисляется.

Только реальные cached OHLC и funding той же площадки, все CSV/manifest SHA проверены. MEXC M1 в старом окне отсутствует, поэтому M5 — явно раскрытый coarse execution proxy; Binance официальные USD-M M1 архивы и funding служат отдельным cross-venue experiment. Funding не переносится между площадками. D1/M5 history110; V4 и оба V3 контроля H1 history2030 при lookback80, v1 H1 history110 для сохранения контрольного фактора. Только закрытые свечи ≤ текущего signal close. Недостающая causal история отвергается без синтетики/fallback.

Все финансовые правила наследуются без настройки: общий счёт1000USDT, риск0.5%, leverage3, max3positions, daily loss2%, limitTTL30минут, фиксированные UTC+3 окна10–12 и16:30–18. Core fee/slip5+2bps per side baseline,10+5stress. Весь MEXC five_fx —5+6 и10+10. MEXC current contract proxy округляет tick/lot, соблюдает min/max и EUR850units; его ограничения не утверждаются исторически известными. На Binance MEXC constraints не применяются. Лимитная очередь/исторический spread не наблюдаются; внутрисвечные неоднозначности трактуются консервативно. Исходы не могут менять издержки.

## Неизменный выбор и окончательный вердикт

Train shortlist≤3: net>0 после fees/slippage/funding, ≥10 закрытых сделок, нет незавершённых заполненных позиций; порядок PF, net return, stable candidateID. Validation выбирает только shortlist: net>0, ≥3 сделки, flat inventory; порядок validationPF, return, trainPF, stableID. Порог10/3/3 не уменьшается после расчётов. Final открывается один раз только для подписанного winner; требуется положительный net во всех трёх folds и ≥10/3/3 закрытых сделок. При n<30 в любом fold результат обозначается малой выборкой, даже положительный. Stress final, Wilson TP/net-win интервалы, funding, marked equity, открытые позиции, unfilled signals и все нулевые/убыточные варианты показываются отдельно.

Stage recovery допускает только прежний plan/code/data SHA и тот же frozen selection. После просмотра final кандидат/сетка/окно не меняются; старые final не переименовываются в независимые данные. Отсутствие прошедшего кандидата — окончательный допустимый результат этого конечного исследования; будущая прибыль не обещается.

## CLI и воспроизводимость

`python -m app.research_post_sweep draft` — без сети/данных/provider анализа. Register требует cache, finite markets, точные folds, все четыре `--parent-dir`, source review, `--source-archive` с соседним `_manifest.json`, `--peer-out-dir`; MEXC также исходный `--contract-snapshot`. Затем для обоих plans выполняются train→validation→final. `ProviderSpec`, V4 candidateID, code/data/source archive/parent stage SHA, cumulative ledger, costs и constraints записаны в новом plan; старый runner не monkeypatch-ится. Batch извлекает все структурные кандидаты и фильтрует каждый strength вариант отдельно; positive signal schedule не меняет полный execution/funding clock.
