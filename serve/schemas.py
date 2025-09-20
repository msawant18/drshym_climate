from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class SegmentRequest(BaseModel):
    domain: str = Field("flood_sar")
    image_uri: str
    options: Optional[Dict[str, Any]] = None

class SegmentResponse(BaseModel):
    scene_id: str
    outputs: Dict[str, str]
    caption: str
    provenance: Dict[str, str]
    policy: Dict[str, bool]
