"""Pydantic-модели REST-контракта."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractModel(BaseModel):
    """Базовая модель с alias-именами из OpenAPI."""

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)


class JobStatus(StrEnum):
    """Состояние задачи."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    ERROR = "ERROR"


class Profile(StrEnum):
    """Профиль детализации результата."""

    FULL = "FULL"
    PREVIEW = "PREVIEW"


class Modality(StrEnum):
    """Источник обнаруженного признака."""

    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


class ViolationClass(StrEnum):
    """Верхнеуровневые классы нарушений."""

    DRUGS = "DRUGS"
    DEVIANT = "DEVIANT"
    TERRORISM = "TERRORISM"
    SEX = "SEX"
    ANTITRADITIONAL = "ANTITRADITIONAL"
    ANTIPATRIOTIC = "ANTIPATRIOTIC"
    LUDOMANIA = "LUDOMANIA"


VALID_SUBCLASSES: dict[str, set[str]] = {
    "DRUGS": {"ALCOHOL", "SMOKING", "DRUGS", "DRUGS2KIDS"},
    "DEVIANT": {"VANDALISM", "VIOLENCE", "SUICIDE", "KIDSSUICIDE", "OBSCENE_LANGUAGE"},
    "TERRORISM": {"TERROR", "EXTREMISM", "TERRORCONTENT"},
    "SEX": {"NUDE", "SEX", "KIDSPORN"},
    "ANTITRADITIONAL": {"LGBT", "CHILDFREE"},
    "ANTIPATRIOTIC": {"INOAGENT", "INOAGENTCONTENT", "ANTIWAR"},
    "LUDOMANIA": {"LUDOMANIA"},
}
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9._~:-]+$")


class DetectionClass(ContractModel):
    """Класс проверки, запрошенный клиентом."""

    violation_class: ViolationClass = Field(alias="class")
    subclasses: list[str] | None = None

    @field_validator("subclasses")
    @classmethod
    def validate_subclasses(cls, value: list[str] | None, info: Any) -> list[str] | None:
        """Проверить список подклассов для выбранного класса."""
        if value is None:
            return value
        if not value:
            raise ValueError("список subclasses не должен быть пустым")
        violation_class = info.data.get("violation_class")
        if violation_class is None:
            return value
        allowed = VALID_SUBCLASSES[str(violation_class)]
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"неизвестные subclasses для {violation_class}: {', '.join(unknown)}"
            )
        return value


class ErrorInfo(ContractModel):
    """Тело ошибки."""

    message: str


class ErrorResponse(ContractModel):
    """Контрактная форма ошибки."""

    error: ErrorInfo


class SourceCredentials(ContractModel):
    """Учётные данные из контракта; в состоянии задачи не сохраняются."""

    login: str
    password: str
    domain: str | None = None


class JobRequest(ContractModel):
    """Запрос, сохранённый внутри задачи."""

    source: str
    customer_id: str | None = Field(default=None, alias="customerId")
    profile: Profile
    detection_classes: list[DetectionClass] = Field(
        min_length=1,
        alias="detectionClasses",
    )

    @field_validator("profile", mode="before")
    @classmethod
    def normalize_profile(cls, value: Any) -> Any:
        """Принять `profile` без учёта регистра."""
        if isinstance(value, str):
            return value.upper()
        return value


class JobCreateRequest(JobRequest):
    """Запрос создания задачи."""

    job_id: str = Field(min_length=1, max_length=128, alias="jobId")
    source_credentials: SourceCredentials | None = Field(
        default=None,
        alias="sourceCredentials",
    )

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        """Ограничить `jobId` безопасными символами для URL и файловой системы."""
        if not JOB_ID_PATTERN.fullmatch(value):
            raise ValueError("jobId must match [A-Za-z0-9._~:-]+")
        return value

    def to_job_request(self) -> JobRequest:
        """Получить безопасную версию запроса для хранения."""
        return JobRequest.model_validate(
            self.model_dump(by_alias=True, exclude={"job_id", "source_credentials"})
        )


class EstimatedPenalty(ContractModel):
    """Оценка штрафа в рублях."""

    amount: int
    currency: str = "RUB"


class DetectionClassStatistic(ContractModel):
    """Статистика по одной паре класс/подкласс."""

    violation_class: str = Field(alias="class")
    subclass: str
    duration_frames: int = Field(alias="durationFrames")
    duration_seconds: float = Field(alias="durationSeconds")
    count: int
    estimated_penalty: EstimatedPenalty = Field(alias="estimatedPenalty")


class SourceInfo(ContractModel):
    """Метаданные исходного видео."""

    frame_count: int = Field(alias="frameCount")
    fps: float
    duration_seconds: float = Field(alias="durationSeconds")


class Detection(ContractModel):
    """Один временной интервал с обнаруженным признаком."""

    start_frame: int = Field(alias="startFrame")
    end_frame: int = Field(alias="endFrame")
    start_seconds: float = Field(alias="startSeconds")
    end_seconds: float = Field(alias="endSeconds")
    violation_class: str = Field(alias="class")
    subclass: str
    confidence: float
    modality: Modality


class JobResult(ContractModel):
    """Результат завершённой задачи."""

    processing_duration_seconds: float = Field(alias="processingDurationSeconds")
    source_info: SourceInfo = Field(alias="sourceInfo")
    total_detections: int = Field(alias="totalDetections")
    detection_class_statistics: list[DetectionClassStatistic] = Field(
        alias="detectionClassStatistics",
    )
    detections: list[Detection]


class Job(ContractModel):
    """Публичное состояние задачи."""

    job_id: str = Field(alias="jobId")
    status: JobStatus
    created_at: str = Field(alias="createdAt")
    request: JobRequest
    started_at: str | None = Field(default=None, alias="startedAt")
    finished_at: str | None = Field(default=None, alias="finishedAt")
    result: JobResult | None = None
    error: ErrorInfo | None = None


JobRecord = Job


class JobCreateResponse(ContractModel):
    """Ответ на создание задачи."""

    job_id: str = Field(alias="jobId")


class MoneyAmount(ContractModel):
    """Денежная сумма."""

    amount: float
    currency: str = "RUB"


class TariffInfo(ContractModel):
    """Информация о тарифе."""

    name: str
    start: str | None = None
    end: str | None = None


class BillingPeriod(ContractModel):
    """Расчётный период."""

    start: str
    end: str


class BillingUsage(ContractModel):
    """Использованные или оставшиеся ресурсы."""

    time_minutes: int = Field(alias="timeMinutes")
    detections: int


class BillingResponse(ContractModel):
    """Ответ маршрута `/api/billing/{customerId}`."""

    customer_id: str = Field(alias="customerId")
    balance: MoneyAmount | None = None
    tariff: TariffInfo | None = None
    billing_period: BillingPeriod | None = Field(default=None, alias="billingPeriod")
    spent: BillingUsage | None = None
    remaining: BillingUsage | None = None
