from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.enums import RiskLevel
from backend.app.models.resource_catalog import ResourceCatalog

_HIGH_RISK_LEVELS: frozenset[str] = frozenset({"L2", "L3"})


class ResourceService:
    """Looks up active crisis resources from resource_catalog for a given risk level.

    Returns an empty list for L0 and L1 — those paths do not require resource cards.
    For L3 (emergency), resources marked for both L2 and L3 are returned so the
    frontend can display the widest available set of options.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_for_risk_level(self, risk_level: str) -> list[dict[str, str]]:
        if risk_level not in _HIGH_RISK_LEVELS:
            return []

        levels_to_query = (
            [RiskLevel.L3, RiskLevel.L2] if risk_level == "L3" else [RiskLevel.L2]
        )

        with self._session_factory() as session:
            rows = session.scalars(
                select(ResourceCatalog)
                .where(
                    ResourceCatalog.risk_level.in_(levels_to_query),
                    ResourceCatalog.is_active.is_(True),
                )
                .order_by(ResourceCatalog.risk_level.desc())
            ).all()

        return [
            {k: v for k, v in {
                "title": row.title,
                "phone": row.phone or "",
                "url": row.link_url or "",
                "description": row.description,
            }.items() if v}
            for row in rows
        ]
