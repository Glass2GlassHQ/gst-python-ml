# object_detector_base.py
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

import traceback
from utils.runtime_utils import runtime_check_gstreamer_version
from video_transform import VideoTransform
from utils.format_converter import FormatConverter
from backend import analytics, frameio, FlowReturn, GObject
from tasks.object_detector import ObjectDetectorTask
from utils.metadata import Metadata


class BaseObjectDetector(VideoTransform, ObjectDetectorTask):
    """
    GStreamer element shell for object detection with batch processing support.
    Handles both single-frame buffers (no metadata) and batch buffers (metadata
    in last chunk). The inference and metadata steps (do_forward / do_decode)
    are inherited from the backend-agnostic ObjectDetectorTask; this class only
    supplies the GStreamer per-buffer glue.
    """

    def __init__(self):
        super().__init__()
        runtime_check_gstreamer_version()
        self.framerate_num = 30
        self.framerate_denom = 1
        self.format_converter = FormatConverter()
        self.metadata = Metadata("si")
        self.logger.info("Initialized BaseObjectDetector")
        self.__track = False

    @GObject.Property(type=bool, default=False)
    def track(self):
        "Enable or disable tracking mode"
        if self.engine:
            return self.engine.track
        return self.__track

    @track.setter
    def track(self, value):
        self.__track = value
        if self.engine:
            self.engine.track = value

    def do_transform_ip(self, buf):
        """
        Transform the input buffer, extracting frame(s) through the backend
        frame I/O (single or batched/muxed sources).
        """
        self.logger.info(f"Transforming buffer: {hex(id(buf))}")
        try:
            # Extract frame(s) and source count through the backend's frame I/O.
            frames, num_sources, format = frameio.read_frames(
                buf,
                self.sinkpad,
                self.width,
                self.height,
                (self.framerate_num, self.framerate_denom),
            )
            if frames is None:
                self.logger.error("Failed to extract frames")
                return FlowReturn.ERROR

            # Process frames (single or batch)
            results = self.do_forward(frames)
            if results is None:
                self.logger.error("Inference returned None")
                return FlowReturn.ERROR

            # Handle single-frame case
            if num_sources == 1:
                self.do_decode(buf, results, stream_idx=0)
            # Handle batch case
            else:
                self.logger.info(f"Processing batch with num_sources={num_sources}")
                results_list = results if isinstance(results, list) else [results]
                if len(results_list) != num_sources:
                    self.logger.error(
                        f"Expected {num_sources} results, got {len(results_list)}"
                    )
                    return FlowReturn.ERROR

                for idx, result in enumerate(results_list):
                    if result is None:
                        self.logger.warning(f"Frame {idx} result is None")
                        continue
                    self.do_decode(buf, result, stream_idx=idx)

            attached_meta = analytics.get_relation_meta(buf)
            if attached_meta:
                count = analytics.relation_length(attached_meta)
                self.logger.info(f"Total metadata relations attached: {count}")
            else:
                self.logger.debug("No detections on this buffer")

            return FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Transform error: {e}\n{traceback.format_exc()}")
            return FlowReturn.ERROR
