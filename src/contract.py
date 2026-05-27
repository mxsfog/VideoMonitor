"""Преобразование артефактов pipeline в REST-контракт."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from common import OUTPUT_DIR
from content_taxonomy import keyword_matches, label_to_subclass
from dto import (
    VALID_SUBCLASSES,
    Detection,
    DetectionClass,
    DetectionClassStatistic,
    EstimatedPenalty,
    JobResult,
    Modality,
    SourceInfo,
)

SUBCLASS_TO_CLASS = {
    subclass: violation_class
    for violation_class, subclasses in VALID_SUBCLASSES.items()
    for subclass in subclasses
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            yield json.loads(line)


def requested_subclasses(detection_classes: list[DetectionClass] | None) -> set[str] | None:
    """Выбрать подклассы из запроса или вернуть `None` без фильтрации."""
    if not detection_classes:
        return None
    selected: set[str] = set()
    for item in detection_classes:
        class_name = str(item.violation_class)
        if item.subclasses:
            selected.update(item.subclasses)
        else:
            selected.update(VALID_SUBCLASSES[class_name])
    return selected


def _is_requested(subclass: str, selected: set[str] | None) -> bool:
    return selected is None or subclass in selected


def _to_detection(
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    subclass: str,
    confidence: float,
    modality: Modality,
) -> Detection:
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame, int(end_frame))
    start_seconds = round(start_frame / fps, 6) if fps else 0.0
    end_seconds = round(max(end_frame + 1, start_frame + 1) / fps, 6) if fps else start_seconds
    return Detection(
        startFrame=start_frame,
        endFrame=end_frame,
        startSeconds=start_seconds,
        endSeconds=max(end_seconds, start_seconds),
        **{"class": SUBCLASS_TO_CLASS[subclass]},
        subclass=subclass,
        confidence=round(max(0.0, min(float(confidence), 1.0)), 3),
        modality=modality,
    )


def read_source_info(video_path: Path, video_stem: str, analysis_fps: float) -> SourceInfo:
    """Прочитать метаданные видео; при ошибке использовать кадры ПЗ 2."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if frame_count > 0 and fps > 0:
                return SourceInfo(
                    frameCount=frame_count,
                    fps=round(fps, 6),
                    durationSeconds=round(frame_count / fps, 6),
                )
        cap.release()
    except Exception:
        pass

    frames = sorted((OUTPUT_DIR / "pz2" / video_stem).glob("frame_*.jpg"))
    frame_count = len(frames)
    duration = frame_count / analysis_fps if analysis_fps else 0.0
    return SourceInfo(
        frameCount=frame_count,
        fps=analysis_fps,
        durationSeconds=round(duration, 6),
    )


def _detections_from_tracks(
    video_stem: str,
    fps: float,
    selected: set[str] | None,
) -> list[Detection]:
    path = OUTPUT_DIR / "pz8" / video_stem / "tracks.json"
    if not path.exists():
        return []
    tracks = json.loads(path.read_text(encoding="utf-8"))
    detections: list[Detection] = []
    for item in tracks:
        subclass = label_to_subclass(item.get("label"))
        if not subclass or not _is_requested(subclass, selected):
            continue
        detections.append(
            _to_detection(
                start_frame=item.get("start_frame", 0),
                end_frame=item.get("end_frame", item.get("start_frame", 0)),
                fps=fps,
                subclass=subclass,
                confidence=item.get("max_conf", 0.5),
                modality=Modality.VIDEO,
            )
        )
    return detections


def _detections_from_yolo_jsonl(
    video_stem: str,
    fps: float,
    selected: set[str] | None,
) -> list[Detection]:
    path = OUTPUT_DIR / "pz5" / video_stem / "detections.jsonl"
    detections: list[Detection] = []
    for item in _iter_jsonl(path):
        subclass = label_to_subclass(item.get("label"))
        if not subclass or not _is_requested(subclass, selected):
            continue
        frame = int(item.get("frame", 0))
        detections.append(
            _to_detection(
                start_frame=frame,
                end_frame=frame,
                fps=fps,
                subclass=subclass,
                confidence=item.get("conf", 0.5),
                modality=Modality.VIDEO,
            )
        )
    return detections


def _detections_from_vlm(
    video_stem: str,
    fps: float,
    selected: set[str] | None,
) -> list[Detection]:
    path = OUTPUT_DIR / "pz7" / video_stem / "classified.jsonl"
    detections: list[Detection] = []
    for item in _iter_jsonl(path):
        frame_name = str(item.get("frame", "frame_000000.jpg"))
        try:
            frame = int(Path(frame_name).stem.split("_")[-1])
        except ValueError:
            frame = 0
        for obj in item.get("objects", []):
            if not obj.get("is_destructive"):
                continue
            subclass = (
                label_to_subclass(obj.get("subclass"))
                or label_to_subclass(obj.get("object"))
                or label_to_subclass(obj.get("reason"))
            )
            if not subclass or not _is_requested(subclass, selected):
                continue
            count = max(1, int(obj.get("count") or 1))
            for _ in range(count):
                detections.append(
                    _to_detection(
                        start_frame=frame,
                        end_frame=frame,
                        fps=fps,
                        subclass=subclass,
                        confidence=obj.get("confidence", 0.7),
                        modality=Modality.VIDEO,
                    )
                )
    return detections


def _keyword_detections(
    *,
    text: str,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    modality: Modality,
    selected: set[str] | None,
) -> list[Detection]:
    lowered = text.lower()
    detections: list[Detection] = []
    for _, subclass in keyword_matches(lowered):
        if not _is_requested(subclass, selected):
            continue
        start_frame = int(start_seconds * fps)
        end_frame = max(start_frame, int(end_seconds * fps))
        detections.append(
            _to_detection(
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                subclass=subclass,
                confidence=0.65,
                modality=modality,
            )
        )
    return detections


def _detections_from_transcript(
    video_stem: str,
    fps: float,
    selected: set[str] | None,
) -> list[Detection]:
    path = OUTPUT_DIR / "pz4" / video_stem / "transcript.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    detections: list[Detection] = []
    for segment in data.get("segments", []):
        detections.extend(
            _keyword_detections(
                text=segment.get("text", ""),
                start_seconds=float(segment.get("start", 0.0)),
                end_seconds=float(segment.get("end", segment.get("start", 0.0))),
                fps=fps,
                modality=Modality.AUDIO,
                selected=selected,
            )
        )
    return detections


def _detections_from_subtitles(
    video_stem: str,
    fps: float,
    selected: set[str] | None,
) -> list[Detection]:
    path = OUTPUT_DIR / "pz3" / video_stem / "subtitles.json"
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    detections: list[Detection] = []
    for item in items:
        detections.extend(
            _keyword_detections(
                text=item.get("text", ""),
                start_seconds=float(item.get("start_s", 0.0)),
                end_seconds=float(item.get("end_s", item.get("start_s", 0.0))),
                fps=fps,
                modality=Modality.VIDEO,
                selected=selected,
            )
        )
    return detections


def build_job_result(
    *,
    video_path: Path,
    video_stem: str,
    analysis_fps: float,
    processing_duration_seconds: float,
    detection_classes: list[DetectionClass] | None = None,
) -> JobResult:
    """Собрать результат `JobResult` из локальных артефактов."""
    selected = requested_subclasses(detection_classes)
    detections = []
    detections.extend(_detections_from_tracks(video_stem, analysis_fps, selected))
    if not detections:
        detections.extend(_detections_from_yolo_jsonl(video_stem, analysis_fps, selected))
    detections.extend(_detections_from_vlm(video_stem, analysis_fps, selected))
    detections.extend(_detections_from_transcript(video_stem, analysis_fps, selected))
    detections.extend(_detections_from_subtitles(video_stem, analysis_fps, selected))
    detections.sort(key=lambda item: (item.start_frame, item.end_frame, item.subclass))

    stats = _build_statistics(detections, analysis_fps)
    return JobResult(
        processingDurationSeconds=round(processing_duration_seconds, 6),
        sourceInfo=read_source_info(video_path, video_stem, analysis_fps),
        totalDetections=len(detections),
        detectionClassStatistics=stats,
        detections=detections,
    )


def _build_statistics(
    detections: list[Detection],
    fps: float,
) -> list[DetectionClassStatistic]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "duration_frames": 0.0}
    )
    for item in detections:
        key = (item.violation_class, item.subclass)
        duration_frames = max(1, item.end_frame - item.start_frame + 1)
        grouped[key]["count"] += 1
        grouped[key]["duration_frames"] += duration_frames

    stats: list[DetectionClassStatistic] = []
    for (violation_class, subclass), values in sorted(grouped.items()):
        duration_frames = int(values["duration_frames"])
        duration_seconds = round(duration_frames / fps, 6) if fps else 0.0
        count = int(values["count"])
        stats.append(
            DetectionClassStatistic(
                **{"class": violation_class},
                subclass=subclass,
                durationFrames=duration_frames,
                durationSeconds=duration_seconds,
                count=count,
                estimatedPenalty=EstimatedPenalty(
                    amount=max(0, int(count * 1000 + duration_seconds * 100)),
                    currency="RUB",
                ),
            )
        )
    return stats


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_hms_compact(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def build_time_based_report(result: JobResult, video_path: Path | str | None = None) -> dict:
    """Собрать временной отчёт в формате `TIME_BASED_REPORT`."""
    processing_seconds = result.processing_duration_seconds
    duration_seconds = result.source_info.duration_seconds
    processing_efficiency = (
        round(processing_seconds / duration_seconds, 2)
        if duration_seconds
        else 0.0
    )
    source_info = {
        "frameCount": result.source_info.frame_count,
        "fps": result.source_info.fps,
        "video_duration_formatted": _format_hms_compact(duration_seconds),
        "analysis_timestamp": _utc_now(),
    }
    extended_source_info = {
        "frameCount": result.source_info.frame_count,
        "fps": result.source_info.fps,
        "video_path": str(video_path or ""),
        "video_duration_seconds": duration_seconds,
        "processing_time_seconds": processing_seconds,
        "processing_efficiency": processing_efficiency,
        "video_duration_formatted": _format_hms_compact(duration_seconds),
        "processing_time_formatted": _format_hms_compact(processing_seconds),
        "analysis_timestamp": _utc_now(),
    }
    detections = []
    for item in result.detections:
        start = _format_hms(item.start_seconds)
        end = _format_hms(item.end_seconds)
        detections.append(
            {
                "startFrame": item.start_frame,
                "endFrame": item.end_frame,
                "start_time": start,
                "end_time": end,
                "time_interval": f"{start} - {end}",
                "subclass": item.subclass.lower(),
                "confidence": item.confidence,
                "type": str(item.modality).lower(),
            }
        )
    return {
        "report_type": "TIME_BASED_REPORT",
        "source_info": source_info,
        "detections": detections,
        "sourceInfo": extended_source_info,
    }


def write_contract_artifacts(
    case_dir: Path,
    result: JobResult,
    video_path: Path | str | None = None,
) -> dict:
    """Записать контрактные JSON-артефакты."""
    case_dir.mkdir(parents=True, exist_ok=True)
    result_payload = result.model_dump(by_alias=True)
    time_report = build_time_based_report(result, video_path=video_path)
    (case_dir / "job_result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (case_dir / "time_based_report.json").write_text(
        json.dumps(time_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return time_report
