# ПЗ 8. Постобработка результатов

## Назначение

ПЗ 8 приводит промежуточные результаты к более удобному виду: удаляет
повторяющиеся текстовые фрагменты и объединяет близкие объектные детекции в
треки.

## Реализация

Файл:

```text
src/pz8_postprocess.py
```

Скрипт выполняет две операции:

1. дедупликация OCR-субтитров через `rapidfuzz`;
2. склейка YOLO bounding boxes в треки по IoU и допустимому разрыву между
   кадрами.

## Вход

```text
output/pz3/<video>/subtitles.json
output/pz5/<video>/detections.jsonl
```

## Выход

```text
output/pz8/<video>/
├── subs_dedup.json
└── tracks.json
```

`subs_dedup.json` содержит очищенные текстовые интервалы. `tracks.json`
содержит объектные треки с начальным и конечным кадром.

## Запуск

```bash
python src/pz8_postprocess.py \
  --out-name example \
  --subs output/pz3/example/subtitles.json \
  --detections output/pz5/example/detections.jsonl
```

## Что показывать

1. Код `src/pz8_postprocess.py`.
2. `subs_dedup.json`.
3. `tracks.json`.
4. Как количество сырых YOLO-детекций уменьшается до треков.

## Ограничения

- простая IoU-склейка не заменяет полноценный multi-object tracking;
- при резких сменах сцены треки могут дробиться;
- при пересечении похожих объектов возможна неверная склейка.
