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
```

Настройки — в `config.toml`, секреты — только через переменные окружения / `.env`
(см. `.env.example`).

## Разработка

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
