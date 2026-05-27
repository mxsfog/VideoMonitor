# Чек-лист перед публикацией

Эта папка подготовлена как чистый пакет для будущего репозитория.

## Входит в публикацию

- исходный код `src/`;
- реализации ПЗ `src/pz1_counters.py` ... `src/pz8_postprocess.py`;
- тесты `tests/`;
- документация `docs/`;
- описания ПЗ `docs/pz1.md` ... `docs/pz8.md`;
- `Dockerfile` и `docker-compose.yml`;
- `README.md`, `pyproject.toml`, `requirements.txt`, `requirements-server.txt`;
- CI-конфигурация `.github/`;
- безопасный пример окружения `.env.example`;
- безопасный скрипт сборки релиза `scripts/build_release.py`.

## Не входит в публикацию

- реальные данные `data/`;
- модели `models/`;
- результаты запусков `output/`;
- release-архивы `dist/`;
- виртуальное окружение `.venv/`;
- кэши Python, pytest и ruff;
- локальные `.env.*` файлы.

## Результаты ПЗ для показа

Код и описания всех ПЗ уже находятся в этом пакете. Сгенерированные результаты
не коммитятся: их нужно показывать из локального `output/` или с развёрнутого
сервера после запуска pipeline.

## Проверки

```bash
python3 -m ruff check src tests
python3 -m pytest -q
python3 scripts/verify_contract.py
python3 scripts/build_release.py --check
```

## Публикация в новый репозиторий

```bash
git init
git add .
git status --short
git commit -m "Подготовить сервис анализа видео"
git branch -M main
git remote add origin <repo-url>
git push -u origin main
```

Перед `git push` нужно ещё раз убедиться, что в индексе нет `data/`, `models/`,
`output/`, `.env` и других локальных файлов.
