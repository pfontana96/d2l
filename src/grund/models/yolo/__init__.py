from typing import Type

from d2l_pep.models.yolo.v1.model import YOLOv1


_SUPPORTED_VERSIONS = {
    "v1": YOLOv1
}


# YOLO model factory
def get_yolo(version: str) -> Type:

    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"Expected YOLO version to be within '{list(_SUPPORTED_VERSIONS.keys())}', got '{version}'")
    
    return _SUPPORTED_VERSIONS.get(version)
