"""
Copyright 2024 The KhulnaSoft DeepScale Team
"""

from .builder import OpBuilder


class UtilsBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_UTILS"
    NAME = "utils"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepscale.ops.{self.NAME}_op"

    def sources(self):
        return ["csrc/utils/flatten_unflatten.cpp"]
