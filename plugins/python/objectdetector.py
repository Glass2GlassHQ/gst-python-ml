# ObjectDetector
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
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
from base_objectdetector import BaseObjectDetector
import backend


class ObjectDetector(BaseObjectDetector):
    """
    GStreamer element for a general object detector where the user sets the model-name property.
    """

    __gstmetadata__ = (
        "ObjectDetector",
        "Transform",
        "General purpose object",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.logger.info(
            "ObjectDetector created without a model. Please set the 'model-name' property."
        )


# The class is backend-agnostic: under g2g the host imports this module and
# instantiates ObjectDetector directly, so no GObject registration applies.
# GStreamer factory registration runs only under the gst backend.
if backend.BACKEND == "gst":
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstBase", "1.0")
        gi.require_version("GLib", "2.0")
        from gi.repository import Gst, GObject  # noqa: E402

        GObject.type_register(ObjectDetector)
        __gstelementfactory__ = ("pyml_objectdetector", Gst.Rank.NONE, ObjectDetector)
    except ImportError as e:
        GlobalLogger().warning(
            f"The 'pyml_objectdetector' element will not be available. Error: {e}"
        )
