from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class StreamState(BaseModel):
    state: List[float] = Field(..., min_length=50, max_length=50)
    remaining_action_chunks: List[List[float]] = Field(..., min_length=50, max_length=50)
    timestamp_ns: int

    @field_validator("remaining_action_chunks")
    @classmethod
    def _remaining_action_chunks_50x22(cls, v: List[List[float]]) -> List[List[float]]:
        # Outer length is already constrained via Field, but keep explicit checks for clarity/safety.
        if len(v) != 50:
            raise ValueError("remaining_action_chunks must have outer length 50")
        for row in v:
            if len(row) != 22:
                raise ValueError("remaining_action_chunks must have inner length 22")
        return v
