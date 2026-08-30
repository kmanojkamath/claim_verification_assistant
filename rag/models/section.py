"""Optional structural representation for a source document."""

from pydantic import field_validator

from .common import ContractModel, ensure_non_empty


class Section(ContractModel):
    """A named portion of a document, independent of future chunking logic."""

    section_id: str
    heading: str | None = None
    content: str

    @field_validator("section_id", "content")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str, info: object) -> str:
        return ensure_non_empty(value, getattr(info, "field_name", "required field"))
