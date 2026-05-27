from __future__ import annotations

import json
from pathlib import Path

import content_taxonomy
import contract
from dto import VALID_SUBCLASSES, DetectionClass


def test_build_job_result_from_local_artifacts(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    video_stem = "sample"
    monkeypatch.setattr(contract, "OUTPUT_DIR", output_dir)

    frames_dir = output_dir / "pz2" / video_stem
    frames_dir.mkdir(parents=True)
    for idx in range(3):
        (frames_dir / f"frame_{idx:06d}.jpg").write_bytes(b"")

    tracks_dir = output_dir / "pz8" / video_stem
    tracks_dir.mkdir(parents=True)
    (tracks_dir / "tracks.json").write_text(
        json.dumps(
            [
                {
                    "label": "knife",
                    "start_frame": 1,
                    "end_frame": 2,
                    "max_conf": 0.8,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    vlm_dir = output_dir / "pz7" / video_stem
    vlm_dir.mkdir(parents=True)
    (vlm_dir / "classified.jsonl").write_text(
        json.dumps(
            {
                "frame": "frame_000002.jpg",
                "objects": [
                    {
                        "object": "сигарета",
                        "count": 2,
                        "confidence": 0.9,
                        "is_destructive": True,
                    }
                ],
                "destructive_count": 2,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    transcript_dir = output_dir / "pz4" / video_stem
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "текст про наркотик"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = contract.build_job_result(
        video_path=Path("missing.mp4"),
        video_stem=video_stem,
        analysis_fps=1.0,
        processing_duration_seconds=12.5,
        detection_classes=[
            DetectionClass.model_validate({"class": "DRUGS"}),
            DetectionClass.model_validate({"class": "DEVIANT", "subclasses": ["VIOLENCE"]}),
        ],
    )

    payload = result.model_dump(by_alias=True)
    assert payload["sourceInfo"]["frameCount"] == 3
    assert payload["totalDetections"] == 4
    assert {item["subclass"] for item in payload["detections"]} == {
        "DRUGS",
        "SMOKING",
        "VIOLENCE",
    }
    assert payload["detectionClassStatistics"]

    time_report = contract.build_time_based_report(result, video_path="missing.mp4")
    assert time_report["report_type"] == "TIME_BASED_REPORT"
    assert time_report["detections"][0]["start_time"] == "00:00:00"
    assert "class" not in time_report["detections"][0]
    assert time_report["sourceInfo"]["video_path"] == "missing.mp4"
    assert "processing_time_seconds" in time_report["sourceInfo"]


def test_time_based_report_shape_matches_bundled_sample(tmp_path, monkeypatch) -> None:
    sample = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "time_based_report.sample.json")
        .read_text(encoding="utf-8")
    )
    output_dir = tmp_path / "output"
    video_stem = "sample"
    monkeypatch.setattr(contract, "OUTPUT_DIR", output_dir)
    frames_dir = output_dir / "pz2" / video_stem
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000000.jpg").write_bytes(b"")

    result = contract.build_job_result(
        video_path=Path("missing.mp4"),
        video_stem=video_stem,
        analysis_fps=1.0,
        processing_duration_seconds=1.0,
    )

    generated = contract.build_time_based_report(result, video_path="missing.mp4")

    assert set(generated) == set(sample)
    assert set(generated["source_info"]) == set(sample["source_info"])
    assert set(generated["sourceInfo"]) == set(sample["sourceInfo"])


def test_age_rating_taxonomy_contains_official_marks_and_internal_3_plus_alias() -> None:
    assert set(content_taxonomy.OFFICIAL_AGE_MARKS) == {"0+", "6+", "12+", "16+", "18+"}
    assert content_taxonomy.AGE_RATING_RULES["3+"]["official"] is False
    assert content_taxonomy.AGE_RATING_RULES["3+"]["alias_for"] == "0+"


def test_risk_taxonomy_maps_problem_video_markers_to_contract_subclasses() -> None:
    assert content_taxonomy.label_to_subclass("пистолет") == "VIOLENCE"
    assert content_taxonomy.label_to_subclass("нацистская символика") == "EXTREMISM"
    assert content_taxonomy.label_to_subclass("свастика") == "EXTREMISM"
    assert content_taxonomy.label_to_subclass("ублюдки") == "OBSCENE_LANGUAGE"
    assert content_taxonomy.label_to_subclass("терроризм") == "TERROR"
    assert content_taxonomy.label_to_subclass("чайлдфри") == "CHILDFREE"
    assert content_taxonomy.label_to_subclass("смена пола") == "LGBT"


def test_risk_taxonomy_uses_only_contract_subclasses() -> None:
    allowed = {subclass for subclasses in VALID_SUBCLASSES.values() for subclass in subclasses}
    for rule in content_taxonomy.RISK_RULES:
        assert rule["subclass"] in allowed
        assert rule["subclass"] in VALID_SUBCLASSES[rule["contract_class"]]


def test_risk_taxonomy_does_not_match_english_substrings() -> None:
    assert content_taxonomy.keyword_matches("ordinary studies") == []


def test_build_job_result_detects_legislative_text_markers(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    video_stem = "sample"
    monkeypatch.setattr(contract, "OUTPUT_DIR", output_dir)

    frames_dir = output_dir / "pz2" / video_stem
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000000.jpg").write_bytes(b"")

    subtitles_dir = output_dir / "pz3" / video_stem
    subtitles_dir.mkdir(parents=True)
    (subtitles_dir / "subtitles.json").write_text(
        json.dumps(
            [
                {"start_s": 4.0, "end_s": 5.0, "text": "свастика"},
                {"start_s": 6.0, "end_s": 7.0, "text": "ублюдки"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = contract.build_job_result(
        video_path=Path("missing.mp4"),
        video_stem=video_stem,
        analysis_fps=1.0,
        processing_duration_seconds=1.0,
        detection_classes=[
            DetectionClass.model_validate({"class": "TERRORISM"}),
            DetectionClass.model_validate({"class": "DEVIANT"}),
        ],
    )

    payload = result.model_dump(by_alias=True)
    assert payload["totalDetections"] == 2
    assert {item["subclass"] for item in payload["detections"]} == {
        "EXTREMISM",
        "OBSCENE_LANGUAGE",
    }
