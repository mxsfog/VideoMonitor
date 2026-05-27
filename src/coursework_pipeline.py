"""Единый pipeline курсовой работы по анализу видео.

Сценарий последовательно запускает ПЗ 2-8, собирает их артефакты и формирует
отчёты по видео. ПЗ 1 остаётся отдельным заданием по изображениям счётчиков.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from common import DATA_DIR, OUTPUT_DIR, setup_logging
from content_taxonomy import keyword_matches
from contract import build_job_result, write_contract_artifacts
from dto import DetectionClass

logger = logging.getLogger(__name__)

OUT_ROOT = OUTPUT_DIR / "coursework"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SRC = Path(__file__).parent

# COCO-классы, которые в рамках учебной задачи считаются риск-сигналами.
SUSPICIOUS_OBJECTS = {"knife", "gun", "rifle", "weapon", "scissors"}
HIGH_RISK_SUBCLASSES = {
    "VIOLENCE",
    "EXTREMISM",
    "TERROR",
    "TERRORCONTENT",
    "DRUGS",
    "DRUGS2KIDS",
    "NUDE",
    "SEX",
    "KIDSPORN",
    "SUICIDE",
    "KIDSSUICIDE",
}


def run_step(cmd: list[str], desc: str) -> None:
    """Запустить шаг pipeline отдельным процессом."""
    logger.info(">>> %s\n    %s", desc, " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"шаг провалился ({res.returncode}): {desc}")


def ensure_artifact(path: Path, description: str) -> None:
    """Остановить pipeline, если обязательный артефакт отсутствует."""
    if not path.exists():
        raise FileNotFoundError(f"нет обязательного артефакта {description}: {path}")


def ensure_frames(frames_dir: Path) -> None:
    """Проверить наличие кадров после ПЗ 2."""
    if not frames_dir.exists() or not any(frames_dir.glob("frame_*.jpg")):
        raise FileNotFoundError(f"нет кадров ПЗ2: {frames_dir}")


def load_detection_classes(path: Path | None) -> list[DetectionClass] | None:
    """Загрузить фильтр классов из запроса REST API."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = payload.get("detectionClasses")
    if raw_items is None and "request" in payload:
        raw_items = payload["request"].get("detectionClasses")
    if not raw_items:
        return None
    return [DetectionClass.model_validate(item) for item in raw_items]


def write_pipeline_state(path: Path | None, payload: dict) -> None:
    """Записать состояние, которое читает REST worker."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_keywords(text: str, langs: tuple[str, ...]) -> list[str]:
    """Найти риск-термины в тексте."""
    del langs
    return [match for match, _ in keyword_matches(text)]


def set_verdict_from_score(findings: dict) -> None:
    """Назначить вердикт по итоговому score."""
    if findings["destructive_score"] >= 0.5:
        findings["verdict"] = "destructive"
    elif findings["destructive_score"] >= 0.2:
        findings["verdict"] = "suspicious"
    else:
        findings["verdict"] = "safe"


def apply_contract_result_to_findings(findings: dict, contract_result) -> None:
    """Синхронизировать Markdown-вывод с контрактным результатом."""
    detections = list(contract_result.detections)
    if not detections:
        return

    subclasses = [str(item.subclass) for item in detections]
    findings["contract_detected_subclasses"] = sorted(set(subclasses))
    if any(subclass in HIGH_RISK_SUBCLASSES for subclass in subclasses):
        findings["destructive_score"] = max(findings["destructive_score"], 0.5)
    else:
        findings["destructive_score"] = max(findings["destructive_score"], 0.2)
    findings["destructive_score"] = round(min(findings["destructive_score"], 1.0), 3)
    set_verdict_from_score(findings)


def aggregate_findings(case_dir: Path, video_stem: str, langs: tuple[str, ...]) -> dict:
    """Собрать признаки из артефактов ПЗ."""
    findings: dict = {
        "video": video_stem,
        "subs_keywords": [],
        "transcript_keywords": [],
        "llm_labels": {},
        "suspicious_objects": {},
        "destructive_score": 0.0,
        "verdict": "safe",
    }

    # Текст из кадра.
    subs_path = OUTPUT_DIR / "pz3" / video_stem / "subtitles.json"
    if subs_path.exists():
        subs = json.loads(subs_path.read_text(encoding="utf-8"))
        all_text = " ".join(s.get("text", "") for s in subs)
        findings["subs_keywords"] = extract_keywords(all_text, langs)

    # Распознанная речь.
    tr_path = OUTPUT_DIR / "pz4" / video_stem / "transcript.json"
    if tr_path.exists():
        tr = json.loads(tr_path.read_text(encoding="utf-8"))
        findings["transcript_keywords"] = extract_keywords(tr.get("full_text", ""), langs)

    # ПЗ 7 пишет один summary: либо по текстовой LLM, либо по VLM.
    llm_path = OUTPUT_DIR / "pz7" / video_stem / "summary.json"
    if llm_path.exists():
        llm_data = json.loads(llm_path.read_text(encoding="utf-8"))
        if "destructive_by_label" in llm_data:
            # Визуальный режим: классификация объектов на кадрах.
            findings["vlm_destructive"] = llm_data["destructive_by_label"]
            findings["vlm_destructive_by_subclass"] = llm_data.get("destructive_by_subclass", {})
            findings["vlm_frames_with_destructive"] = llm_data.get("frames_with_destructive", 0)
            findings["vlm_frames_total"] = llm_data.get("frames_processed", 0)
            findings["vlm_top_objects"] = dict(list(llm_data.get("by_label", {}).items())[:10])
        else:
            # Текстовый режим: классификация фрагментов OCR/Whisper.
            findings["llm_labels"] = llm_data.get("by_label", {})

    # Объектные детекции YOLO.
    yolo_path = OUTPUT_DIR / "pz5" / video_stem / "summary.json"
    if yolo_path.exists():
        by_class = json.loads(yolo_path.read_text(encoding="utf-8")).get("by_class", {})
        findings["suspicious_objects"] = {k: v for k, v in by_class.items() if k in SUSPICIOUS_OBJECTS}
        findings["all_classes"] = by_class

    # Итоговый score собирается из независимых слабых сигналов.
    score = 0.0
    score += min(len(findings["subs_keywords"]) * 0.1, 0.3)
    score += min(len(findings["transcript_keywords"]) * 0.1, 0.3)
    bad = sum(v for k, v in findings["llm_labels"].items()
              if k in {"toxic", "extremism", "harassment", "adult"})
    total = sum(findings["llm_labels"].values()) or 1
    score += min(0.4 * bad / total, 0.4)
    score += min(sum(findings["suspicious_objects"].values()) * 0.05, 0.3)
    # VLM учитывается и по доле кадров, и по числу найденных объектов.
    if findings.get("vlm_frames_total"):
        frac = findings["vlm_frames_with_destructive"] / findings["vlm_frames_total"]
        score += min(0.2 * frac, 0.2)
        score += min(sum(findings.get("vlm_destructive", {}).values()) * 0.2, 0.6)
    findings["destructive_score"] = round(min(score, 1.0), 3)

    set_verdict_from_score(findings)
    return findings


def write_report(case_dir: Path, findings: dict) -> None:
    """Сформировать Markdown-отчёт для просмотра человеком."""
    md = [
        f"# Отчёт по видео `{findings['video']}`",
        "",
        f"## Вердикт: **{findings['verdict'].upper()}** "
        f"(score = {findings['destructive_score']})",
        "",
        "## Признаки",
        f"- Стоп-слова в титрах: {', '.join(findings['subs_keywords']) or 'не найдены'}",
        f"- Стоп-слова в транскрипте: {', '.join(findings['transcript_keywords']) or 'не найдены'}",
        f"- LLM-классификация фрагментов: `{json.dumps(findings['llm_labels'], ensure_ascii=False)}`",
        f"- Подозрительные объекты YOLO: `{json.dumps(findings['suspicious_objects'], ensure_ascii=False)}`",
    ]
    if findings.get("vlm_frames_total"):
        md += [
            f"- VLM (Gemini 2.5 Flash) обработано кадров: "
            f"{findings['vlm_frames_total']}, с деструктивом: "
            f"{findings['vlm_frames_with_destructive']}",
            f"- VLM деструктивные объекты: "
            f"`{json.dumps(findings.get('vlm_destructive', {}), ensure_ascii=False)}`",
            f"- VLM деструктивные подклассы: "
            f"`{json.dumps(findings.get('vlm_destructive_by_subclass', {}), ensure_ascii=False)}`",
            f"- VLM топ-объекты: "
            f"`{json.dumps(findings.get('vlm_top_objects', {}), ensure_ascii=False)}`",
        ]
    md += [
        "",
        "## Все обнаруженные классы YOLO (топ-15)",
    ]
    if findings.get("all_classes"):
        items = sorted(findings["all_classes"].items(), key=lambda x: -x[1])[:15]
        md.append("| класс | детекций |")
        md.append("|---|---|")
        for k, v in items:
            md.append(f"| {k} | {v} |")
    md += [
        "",
        "## Контрактные артефакты",
        f"- OpenAPI JobResult: `{findings.get('contract_artifacts', {}).get('job_result')}`",
        f"- TIME_BASED_REPORT: `{findings.get('contract_artifacts', {}).get('time_based_report')}`",
        f"- Контрактных детекций: {findings.get('contract_total_detections', 0)}",
        f"- Контрактные подклассы: "
        f"`{json.dumps(findings.get('contract_detected_subclasses', []), ensure_ascii=False)}`",
        "",
        "## Артефакты",
        "- ПЗ2 кадры: `output/pz2/<video>/`",
        "- ПЗ3 титры: `output/pz3/<video>/subtitles.{json,srt}`",
        "- ПЗ4 транскрипт: `output/pz4/<video>/transcript.{json,srt,txt}`",
        "- ПЗ5 детекции: `output/pz5/<video>/detections.jsonl`",
        "- ПЗ5 разметка: `output/pz5/<video>_pred/`",
        "- ПЗ6 классификация кадров: `output/pz6/<video>/predictions.jsonl`",
        "- ПЗ7 LLM: `output/pz7/<video>/classified.jsonl`",
        "- ПЗ8 постобработка: `output/pz8/<video>/`",
    ]
    (case_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    (case_dir / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2),
                                            encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Аргументы командной строки."""
    p = argparse.ArgumentParser(description="Курсовая: pipeline выявления деструктивного контента")
    p.add_argument("source", help="ссылка или путь к видео")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--llm-model", default="qwen3.5:9b")
    p.add_argument("--llm-backend",
                   choices=["ollama", "openrouter", "vlm", "none"],
                   default="ollama",
                   help="ollama (qwen3.5) | openrouter (text LLM) | "
                        "vlm (Gemini 2.5 Flash распознаёт объекты на кадрах) | none")
    p.add_argument("--vlm-every-n", type=int, default=5,
                   help="прореживание кадров для VLM (каждый N-й)")
    p.add_argument("--lang", default="ru")
    p.add_argument("--skip", default="", help="через запятую: pz2,pz3,pz4,pz5,pz6,pz7,pz8")
    p.add_argument("--job-id", default=None, help="идентификатор API-задачи")
    p.add_argument("--job-request", type=Path, default=None, help="JSON запроса /api/jobs")
    p.add_argument("--job-state", type=Path, default=None, help="куда записать state pipeline")
    return p.parse_args()


def main() -> None:
    """Запустить полный сценарий обработки видео."""
    setup_logging()
    started = time.perf_counter()
    args = parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    detection_classes = load_detection_classes(args.job_request)

    # Подготовить исходное видео и кадры.
    src = args.source
    if src.startswith("http"):
        # Ссылки обрабатывает ПЗ 2: он скачивает видео и сразу режет кадры.
        run_step([PYTHON, str(SRC / "pz2_slicer.py"), src,
                  "--fps", str(args.fps), "--quality", "best"], "pz2: скачивание+нарезка")
        # После скачивания берём последний созданный mp4 в рабочем каталоге.
        videos = sorted((DATA_DIR / "videos").glob("*.mp4"), key=lambda p: -p.stat().st_mtime)
        if not videos:
            raise RuntimeError("не нашёл скачанное видео")
        video_path = videos[0]
    else:
        video_path = Path(src)
        if "pz2" not in skip:
            run_step([PYTHON, str(SRC / "pz2_slicer.py"), str(video_path),
                      "--fps", str(args.fps)], "pz2: нарезка")

    video_stem = video_path.stem
    frames_dir = OUTPUT_DIR / "pz2" / video_stem
    case_dir = OUT_ROOT / video_stem
    case_dir.mkdir(parents=True, exist_ok=True)
    ensure_artifact(video_path, "исходное видео")
    ensure_frames(frames_dir)

    # OCR титров.
    if "pz3" not in skip:
        run_step([PYTHON, str(SRC / "pz3_subtitles.py"),
                  "--frames", str(frames_dir),
                  "--src-fps", str(args.fps),
                  "--langs", "ru,en"], "pz3: OCR титров")
        ensure_artifact(OUTPUT_DIR / "pz3" / video_stem / "subtitles.json", "ПЗ3 subtitles.json")
    # Распознавание речи.
    if "pz4" not in skip:
        run_step([PYTHON, str(SRC / "pz4_whisper.py"), str(video_path),
                  "--model", args.whisper_model, "--lang", args.lang], "pz4: whisper")
        ensure_artifact(OUTPUT_DIR / "pz4" / video_stem / "transcript.json", "ПЗ4 transcript.json")
    # Детекция объектов.
    if "pz5" not in skip:
        run_step([PYTHON, str(SRC / "pz5_yolo.py"), str(video_path),
                  "--every-n", "5"], "pz5: yolo")
        ensure_artifact(OUTPUT_DIR / "pz5" / video_stem / "summary.json", "ПЗ5 summary.json")
    # Классификация кадров.
    if "pz6" not in skip:
        run_step([PYTHON, str(SRC / "pz6_resnet.py"), str(frames_dir)],
                 "pz6: resnet")
        ensure_artifact(OUTPUT_DIR / "pz6" / video_stem / "summary.json", "ПЗ6 summary.json")
    # LLM/VLM-анализ. Для текста сначала используем OCR, затем транскрипт.
    if "pz7" not in skip:
        subs_json = OUTPUT_DIR / "pz3" / video_stem / "subtitles.json"
        tr_json = OUTPUT_DIR / "pz4" / video_stem / "transcript.json"
        target = None
        if subs_json.exists():
            data = json.loads(subs_json.read_text(encoding="utf-8"))
            if data:
                target = subs_json
        if target is None and tr_json.exists():
            target = tr_json
        if args.llm_backend == "none":
            logger.info("pz7: пропущено по llm_backend=none")
        elif args.llm_backend == "vlm":
            # VLM анализирует прореженную выборку кадров.
            model_name = args.llm_model if "/" in args.llm_model \
                else "google/gemini-2.5-flash"
            run_step([PYTHON, str(SRC / "pz7_vlm_gemini.py"), str(frames_dir),
                      "--model", model_name, "--out-name", video_stem,
                      "--every-n", str(args.vlm_every_n)],
                     "pz7: VLM распознавание объектов (gemini)")
            ensure_artifact(OUTPUT_DIR / "pz7" / video_stem / "summary.json", "ПЗ7 summary.json")
        elif target is not None:
            if args.llm_backend == "openrouter":
                model_name = args.llm_model if "/" in args.llm_model \
                    else "inclusionai/ling-2.6-1t:free"
                run_step([PYTHON, str(SRC / "pz7_openrouter.py"), str(target),
                          "--model", model_name, "--out-name", video_stem],
                         "pz7: LLM классификация (openrouter)")
            else:
                run_step([PYTHON, str(SRC / "pz7_llm.py"), str(target),
                          "--model", args.llm_model, "--out-name", video_stem],
                         "pz7: LLM классификация (ollama)")
            ensure_artifact(OUTPUT_DIR / "pz7" / video_stem / "summary.json", "ПЗ7 summary.json")
        else:
            logger.warning("pz7: нет титров/транскрипта для LLM-классификации")
    # Постобработка OCR и YOLO.
    if "pz8" not in skip:
        cmd = [PYTHON, str(SRC / "pz8_postprocess.py"), "--out-name", video_stem]
        subs_json = OUTPUT_DIR / "pz3" / video_stem / "subtitles.json"
        if subs_json.exists():
            cmd += ["--subs", str(subs_json)]
        det_jsonl = OUTPUT_DIR / "pz5" / video_stem / "detections.jsonl"
        if det_jsonl.exists():
            cmd += ["--detections", str(det_jsonl)]
        run_step(cmd, "pz8: постобработка")
        ensure_artifact(OUTPUT_DIR / "pz8" / video_stem / "tracks.json", "ПЗ8 tracks.json")

    # Финальная агрегация и контрактные файлы.
    findings = aggregate_findings(case_dir, video_stem, ("ru", "en"))
    processing_duration_seconds = time.perf_counter() - started
    contract_result = build_job_result(
        video_path=video_path,
        video_stem=video_stem,
        analysis_fps=args.fps,
        processing_duration_seconds=processing_duration_seconds,
        detection_classes=detection_classes,
    )
    time_report = write_contract_artifacts(case_dir, contract_result, video_path=video_path)
    findings["contract_total_detections"] = contract_result.total_detections
    findings["contract_detection_classes"] = [
        item.model_dump(by_alias=True, exclude_none=True)
        for item in (detection_classes or [])
    ]
    findings["contract_artifacts"] = {
        "job_result": str(case_dir / "job_result.json"),
        "time_based_report": str(case_dir / "time_based_report.json"),
    }
    apply_contract_result_to_findings(findings, contract_result)
    write_report(case_dir, findings)
    write_pipeline_state(
        args.job_state,
        {
            "jobId": args.job_id,
            "video_stem": video_stem,
            "video_path": str(video_path),
            "report_path": str(case_dir / "report.md"),
            "findings_path": str(case_dir / "findings.json"),
            "job_result_path": str(case_dir / "job_result.json"),
            "time_based_report_path": str(case_dir / "time_based_report.json"),
            "totalDetections": contract_result.total_detections,
            "result": contract_result.model_dump(by_alias=True),
            "timeBasedReport": time_report,
        },
    )
    logger.info("=== ВЕРДИКТ: %s (score=%.2f) ===", findings["verdict"], findings["destructive_score"])
    logger.info("отчёт: %s", case_dir / "report.md")


if __name__ == "__main__":
    main()
