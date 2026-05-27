# ПЗ 7. LLM и VLM-классификация

## Назначение

ПЗ 7 добавляет семантическую классификацию текста и кадров. В проекте
реализованы три режима: локальная текстовая LLM, текстовая LLM через
OpenRouter и визуальная модель через OpenRouter.

## Реализация

Файлы:

```text
src/pz7_llm.py
src/pz7_openrouter.py
src/pz7_vlm_gemini.py
run_pz7.py
run_pz7_openrouter.py
run_pz7_vlm.py
```

Режимы:

| Режим | Назначение |
|---|---|
| `ollama` | локальная классификация текстовых фрагментов |
| `openrouter` | классификация текста через внешний API |
| `vlm` | классификация кадров через визуальную модель |

Для серверного полного теста использовалась модель:

```text
google/gemini-2.5-flash
```

## Вход

Для текстового режима:

```text
output/pz3/<video>/subtitles.json
output/pz4/<video>/transcript.json
```

Для VLM:

```text
output/pz2/<video>/frame_*.jpg
```

## Выход

```text
output/pz7/<video>/
├── classified.jsonl
└── summary.json
```

`classified.jsonl` содержит результат по каждому фрагменту или кадру.
`summary.json` содержит агрегированную статистику.

## Запуск

Локальный текстовый режим:

```bash
python run_pz7.py
```

OpenRouter для текста:

```bash
OPENROUTER_API_KEY=<key> python run_pz7_openrouter.py
```

VLM по кадрам:

```bash
OPENROUTER_API_KEY=<key> python src/pz7_vlm_gemini.py \
  output/pz2/example \
  --model google/gemini-2.5-flash \
  --out-name example \
  --every-n 10
```

## Что показывать

1. `src/pz7_vlm_gemini.py` как основной VLM-режим.
2. `summary.json` с количеством обработанных кадров и найденных признаков.
3. `classified.jsonl` с объяснениями по отдельным кадрам.
4. Связь VLM-признаков с `job_result.json` курсовой.

## Ограничения

- внешний API получает данные из кадров, поэтому нужен явный допуск на такой
  запуск;
- бесплатные модели OpenRouter могут быть нестабильны или возвращать неполный
  ответ;
- VLM не является детерминированной экспертизой;
- для длинных видео нужно прореживание кадров через `--every-n`.
