"""
Modèles de données Pydantic partagés dans tout le projet.
Validation structurée = PAS de LLM ici.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Attachment(BaseModel):
    filename: str
    content_type: str
    text: Optional[str] = None  # texte extrait du PDF


class Email(BaseModel):
    id: str
    from_address: str
    subject: str
    body: str
    received_at: datetime
    attachments: list[Attachment] = Field(default_factory=list)


class Classification(BaseModel):
    category: Literal[
        "facture",
        "devis",
        "question_comptable",
        "document_upload",
        "relance",
        "autre",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @property
    def needs_human_review(self) -> bool:
        threshold = float(__import__("os").getenv("CONFIDENCE_THRESHOLD", "0.70"))
        return self.confidence < threshold


class ExtractedInfo(BaseModel):
    montant: Optional[float] = None
    devise: str = "EUR"
    date_document: Optional[date] = None
    client_name: Optional[str] = None
    urgency: Literal["low", "medium", "high"] = "low"
    action_required: str = ""
    tva_rate: Optional[float] = None  # ex: 20.0


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentAction(BaseModel):
    action_type: Literal["respond", "classify_document", "escalate", "create_devis"]
    response: Optional[str] = None
    escalation_reason: Optional[str] = None
    document_category: Optional[str] = None


class AgentResult(BaseModel):
    email_id: str
    classification: Classification
    extracted_info: ExtractedInfo
    validation: ValidationResult
    action: AgentAction
    processing_time_ms: int
    rag_context_used: bool
