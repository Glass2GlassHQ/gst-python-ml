# CaptionQwen
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")

    from gi.repository import Gst, GObject  # noqa: E402

    from engine.caption_qwen_engine import CaptionQwenEngine
    from engine.engine_factory import EngineFactory
    from base_caption import BaseCaption

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption_qwen' element will not be available. Error {e}"
    )


class CaptionQwen(BaseCaption):
    """
    GStreamer element for captioning video frames using Qwen Vision.
    """

    __gstmetadata__ = (
        "CaptionQwen",
        "Transform",
        "Captions video clips using Qwen Vision model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        # set engine_name directly on mgr, as engine_name property is read only
        self.mgr.engine_name = "pyml_caption_qwen_engine"
        EngineFactory.register(self.engine_name, CaptionQwenEngine)


if CAN_REGISTER_ELEMENT:
    GObject.type_register(CaptionQwen, "pyml_caption_qwen")
    __gstelementfactory__ = ("pyml_caption_qwen", Gst.Rank.NONE, CaptionQwen)
else:
    GlobalLogger().warning(
        "The 'pyml_caption_qwen' element will not be registered because required modules are missing."
    )
