# Anomaly Detection
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

CAN_REGISTER_ELEMENT = True
try:
    import ctypes
    import json

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GObject

    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from utils.muxed_buffer_processor import MuxedBufferProcessor
    from engine.anomaly_engine import AnomalyEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'anomaly' element will not be available. Error {e}")

# Header prefix for anomaly detection buffer metadata
ANOMALY_META_HEADER = b"GST-ANOMALY:"


class AnomalyTransform(VideoTransform):
    """
    GStreamer element for anomaly detection in video frames.

    Uses a pretrained feature extractor to compute patch-level anomaly scores
    against a reference distribution of normal frames.

    Set reference-path to a .npy file containing precomputed reference features
    from normal samples. When draw-heatmap=True (default), an anomaly heatmap
    is overlaid on frames that exceed the threshold.

    Anomaly scores are always attached as a GST-ANOMALY: memory chunk (JSON).
    """

    __gstmetadata__ = (
        "Anomaly Detection",
        "Transform",
        "Video anomaly detection using feature extraction and PatchCore scoring",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    threshold = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=100.0,
        nick="Anomaly Threshold",
        blurb="Anomaly score threshold above which a frame is flagged",
        flags=GObject.ParamFlags.READWRITE,
    )

    reference_path = GObject.Property(
        type=str,
        default="",
        nick="Reference Path",
        blurb="Path to .npy file with reference feature vectors from normal frames",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_heatmap = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Heatmap",
        blurb="Overlay anomaly heatmap on frames above threshold",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_anomaly_engine"
        EngineFactory.register(self.mgr.engine_name, AnomalyEngine)
        self.format_converter = FormatConverter()
        self._reference_loaded = False

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_anomaly")

    def do_transform_ip(self, buf):
        try:
            # Load reference features on first transform if path is set
            if not self._reference_loaded and self.reference_path and self.engine:
                self.engine.load_reference(self.reference_path)
                self._reference_loaded = True

            processor = MuxedBufferProcessor(
                self.logger, self.width, self.height, 30, 1
            )
            frames, _, num_sources, fmt = processor.extract_frames(buf, self.sinkpad)
            if frames is None:
                return Gst.FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames
            result = self._do_forward(frame)
            if result is None:
                return Gst.FlowReturn.OK

            self._apply_anomaly(buf, result, fmt, frame)
            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Anomaly detection transform error: {e}")
            return Gst.FlowReturn.ERROR

    def _do_forward(self, frame):
        if self.engine:
            return self.engine.do_forward(frame, threshold=self.threshold)
        return None

    def _apply_anomaly(self, buf, result, fmt, frame):
        """Overlay heatmap on frame and append anomaly metadata."""
        import cv2
        import numpy as np

        is_anomaly = result.get("is_anomaly", False)
        heatmap = result.get("heatmap")

        # Draw heatmap overlay before appending read-only metadata memory
        if self.draw_heatmap and is_anomaly and heatmap is not None:
            H, W = frame.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (W, H))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 0.6, heatmap_rgb, 0.4, 0)

            # Draw anomaly score text
            score = result.get("score", 0.0)
            text = f"ANOMALY: {score:.3f}"
            cv2.putText(
                overlay,
                text,
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            output = self._convert_rgb_to_format(overlay, fmt)
            if output is not None:
                success, map_info = buf.map(Gst.MapFlags.WRITE)
                if success:
                    try:
                        frame_bytes = np.ascontiguousarray(output).tobytes()
                        dst = (ctypes.c_char * map_info.size).from_buffer(map_info.data)
                        ctypes.memmove(
                            dst, frame_bytes, min(len(frame_bytes), map_info.size)
                        )
                    finally:
                        buf.unmap(map_info)

        # Append anomaly metadata (without the numpy heatmap)
        meta = {
            "score": result.get("score", 0.0),
            "is_anomaly": is_anomaly,
        }
        meta_bytes = ANOMALY_META_HEADER + json.dumps(meta).encode("utf-8")
        tmp = Gst.Buffer.new_allocate(None, len(meta_bytes), None)
        tmp.fill(0, meta_bytes)
        buf.append_memory(tmp.get_memory(0))

    @staticmethod
    def _convert_rgb_to_format(rgb, fmt):
        """Convert an RGB numpy array to the target GStreamer video format."""
        import cv2
        import numpy as np

        if fmt == "RGB":
            return rgb
        elif fmt == "BGR":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif fmt == "RGBA":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        elif fmt == "BGRA":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        elif fmt == "ARGB":
            rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
            return np.roll(rgba, 1, axis=-1)
        elif fmt == "ABGR":
            bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
            return np.roll(bgra, 1, axis=-1)
        else:
            return rgb


if CAN_REGISTER_ELEMENT:
    GObject.type_register(AnomalyTransform)
    __gstelementfactory__ = ("pyml_anomaly", Gst.Rank.NONE, AnomalyTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_anomaly' element will not be registered because required modules are missing."
    )
