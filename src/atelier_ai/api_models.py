"""Public response contracts for the historical recommendation API."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints


API_VERSION = "0.1.0"
CustomerId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Count = Annotated[int, Field(ge=0)]


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ApiResponse(PublicModel):
    api_version: str = API_VERSION


class SnapshotResponse(ApiResponse):
    historical_date: date
    bundle_version: Digest


class HealthResponse(ApiResponse):
    status: Literal["ok"] = "ok"


class ReadyResponse(SnapshotResponse):
    status: Literal["ready"] = "ready"
    serving_mode: Literal["precomputed_recommendations"]
    model_sha256: Digest
    default_method: str
    methods: list[str]
    customers: Count
    catalogue_articles: Count
    recommendations_per_customer: Literal[12]


class CustomerSummary(PublicModel):
    customer_id: CustomerId
    customer_has_history: bool
    customer_history_rows: Count
    customer_distinct_articles: Count


class CustomerPage(SnapshotResponse):
    total: Count
    offset: Count
    limit: Annotated[int, Field(ge=1, le=100)]
    next_offset: Count | None
    items: list[CustomerSummary]


class ProfileSummary(PublicModel):
    customer_has_history: bool
    customer_history_rows: Count
    customer_distinct_articles: Count
    customer_distinct_product_types: Count
    customer_distinct_colours: Count
    customer_recency_days: Annotated[FiniteFloat, Field(ge=1, le=180)] | None
    customer_rows_7d: Count
    customer_rows_28d: Count
    customer_rows_90d: Count
    customer_channel_2_share: Annotated[FiniteFloat, Field(ge=0, le=1)]
    dominant_product_group: str | None


class PurchaseRow(PublicModel):
    t_dat: date
    customer_id: CustomerId
    article_id: str
    price_source_units: FiniteFloat | None
    sales_channel_id: int | None


class CustomerProfile(SnapshotResponse):
    customer_id: CustomerId
    history_days: Literal[180]
    summary: ProfileSummary
    recent_purchase_rows: Annotated[list[PurchaseRow], Field(max_length=20)]


class RecommendedArticle(PublicModel):
    article_id: str
    rank: Annotated[int, Field(ge=1, le=12)]
    score: FiniteFloat | None = Field(description="Raw ranking score or historical count; null for rank-only references.")
    prod_name: str | None
    product_type_name: str
    product_group_name: str
    colour_group_name: str
    index_name: str | None
    section_name: str | None
    garment_group_name: str | None
    detail_desc: str | None


class RecommendationsResponse(SnapshotResponse):
    customer_id: CustomerId
    strategy: str
    score_kind: Literal["uncalibrated_lightgbm_ranking_score", "historical_purchase_rows_7d", "baseline_rank_only"]
    model_sha256: Digest | None
    items: Annotated[list[RecommendedArticle], Field(min_length=12, max_length=12)]


class ErrorDetail(PublicModel):
    code: str
    message: str


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="allow")
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(PublicModel):
    detail: ErrorDetail | list[ValidationIssue]
