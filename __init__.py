from .nodes import PhotoDoodleSamplerAdvanced
from .nodes import PhotoDoodleCrop
from .nodes import PhotoDoodleParams
# 定义要导出的节点类
NODE_CLASS_MAPPINGS = {
    "PhotoDoodleSamplerAdvanced": PhotoDoodleSamplerAdvanced,
    "PhotoDoodleCrop": PhotoDoodleCrop,
    "PhotoDoodleParams": PhotoDoodleParams
}

# 定义节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoodleSamplerAdvanced": "照片涂鸦高级采样器",
    "PhotoDoodleCrop": "照片涂鸦裁切",
    "PhotoDoodleParams": "照片涂鸦参数"
}

# 导出节点
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 