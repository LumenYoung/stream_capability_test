from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class StreamState(BaseModel):
    state: List[float] = Field(..., min_length=50, max_length=50)
    # Variable-length sequence of 22-dim action chunks.
    # Outer length is allowed to vary; inner length is enforced to be 22.
    remaining_action_chunks: List[List[float]] = Field(...)
    timestamp_ns: int

    @field_validator("remaining_action_chunks")
    @classmethod
    def _remaining_action_chunks_50x22(cls, v: List[List[float]]) -> List[List[float]]:
        for row in v:
            if len(row) != 22:
                raise ValueError("remaining_action_chunks must have inner length 22")
        return v
