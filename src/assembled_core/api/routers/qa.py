# src/assembled_core/api/routers/qa.py
"""QA/Health check endpoints."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from src.assembled_core.api.models import QaCheck, QaStatus, QaStatusEnum
from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.qa.health import aggregate_qa_status

router = APIRouter()


@router.get("/qa/status", response_model=QaStatus)
def get_qa_status(freq: str = Query(default="1d", description="Trading frequency")) -> QaStatus:
    """Get QA/Health check status for a given frequency.
    
    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"
    
    Returns:
        QaStatus with overall status and list of checks
    
    Raises:
        HTTPException: 400 if freq is not supported, 500 for unexpected errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}"
        )
    
    try:
        # Call aggregate_qa_status
        result = aggregate_qa_status(freq, output_dir=OUTPUT_DIR)
        
        # Map to QaStatus Pydantic model
        checks = []
        for check_dict in result["checks"]:
            checks.append(
                QaCheck(
                    check_name=check_dict["name"],
                    status=QaStatusEnum(check_dict["status"]),
                    message=check_dict["message"],
                    details=check_dict.get("details")
                )
            )
        
        # Map overall_status
        overall_status = QaStatusEnum(result["overall_status"])
        
        # Build summary
        summary = {
            "ok": sum(1 for c in checks if c.status == QaStatusEnum.OK),
            "warning": sum(1 for c in checks if c.status == QaStatusEnum.WARNING),
            "error": sum(1 for c in checks if c.status == QaStatusEnum.ERROR)
        }
        
        return QaStatus(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            checks=checks,
            summary=summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing QA status: {e}"
        )

