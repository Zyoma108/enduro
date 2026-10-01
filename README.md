# Enduro

Торговый агент для крипторынка: LLM-трейдер, который ищет волатильность в интрадее
и зарабатывает на ней. Рынок — бездорожье, агент — райдер, риск-менеджмент — тормоза.

- Анализ: несколько бирж, источник правды — Binance (наибольший объём).
- Исполнение: Bybit.
- Решения: LLM-модель, которой задан «дух» трейдинга промптом, работает через инструменты.
- Риск-менеджмент: детерминированный код с правом вето над любым решением агента.

Подробности — в [ARCHITECTURE.md](ARCHITECTURE.md).

## Быстрый старт

```bash
python3 -m venv .venv
.venv/bin/pip install -e . --group dev

# Живой поток данных с Binance и Bybit со сводкой каждые 5 секунд.
# Поток пишется в data/ (Parquet); --no-record — без записи.
.venv/bin/enduro collect
.venv/bin/enduro collect --symbols BTC/USDT:USDT --interval 2

# SQL по записанным данным (представления trades, books, candles)
.venv/bin/enduro sql "select exchange, symbol, count(*) from trades group by all"

# Радар: какие монеты сейчас необычно активны
.venv/bin/enduro universe          # список монет, которые сканируем
.venv/bin/enduro backfill          # загрузить/дополнить 28 дней минутных свечей
.venv/bin/enduro scan              # обновление раз в минуту (--once, --json, --top N)
.venv/bin/enduro focus ETH/USDT:USDT   # поток, стакан, подтверждение Binance (--json)

# Исполнение (по умолчанию Bybit Demo Trading; ключи — в .env)
.venv/bin/enduro account           # режим маржи, хедж, баланс, позиции, ордера
.venv/bin/enduro account --setup   # включить кросс-маржу и режим хеджирования
.venv/bin/enduro test-trade ETH/USDT:USDT --side long --with-stop 1   # только demo
.venv/bin/enduro risk              # лимиты и состояние риск-менеджера (--reset — снять kill-switch)

# Агент. По умолчанию модель запускается через Claude Code CLI (`claude -p`) под вашей
# учётной записью; промпт — prompts/trader.md, журнал — state/journal/*.jsonl
.venv/bin/enduro agent --dry-run --ticks 3   # решения без отправки ордеров
.venv/bin/enduro agent                       # торговля на demo
.venv/bin/enduro feedback                    # чего агенту не хватило (отзывы на инструменты)
.venv/bin/enduro journal -f                  # журнал по-человечески (--day, -a — всё подряд)
```

Настройки — в `config.toml`, секреты — только через переменные окружения / `.env`
(см. `.env.example`).

## Разработка

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
