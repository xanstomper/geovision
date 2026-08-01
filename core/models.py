from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class PhaseResult:
    phase_name: str
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class LocationEstimate:
    latitude: float
    longitude: float
    confidence: float
    sources: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    region_name: Optional[str] = None
