# ClipTask
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

"""Backend-agnostic CLIP classification task.

`ClipTask` turns zero-shot classification results into GstAnalytics-style
metadata via the analytics facade. It never touches the buffer; the backend
element shell does the frame read, runs inference on a background thread, and
passes the opaque metadata target to `decode`.

Contract expected from the host element:
  self.logger     - logger (provided by MLEngineMixin)
  self.top_k      - number of top labels to attach (a backend-declared property)
  self.threshold  - minimum probability to include (a backend-declared property)
  self.width      - frame width (host geometry)
  self.height     - frame height (host geometry)

`decode`'s `target` argument is the opaque metadata target (a Gst buffer on the
gst backend); it is only passed through to the analytics facade.
"""

from backend import analytics


class ClipTask:
    """Result-to-metadata step, independent of any framework."""

    def decode(self, target, results):
        """Attach top-k classification results above threshold as GstAnalytics metadata."""
        meta = analytics.add_relation_meta(target)
        if not meta:
            self.logger.error("Failed to add analytics relation metadata")
            return

        attached = 0
        for label, prob in results:
            if attached >= self.top_k:
                break
            if prob < self.threshold:
                break

            mtd = analytics.add_object(
                meta,
                f"clip_{label.replace(' ', '_')}",
                0,
                0,
                self.width,
                self.height,
                prob,
            )
            if mtd is not None:
                attached += 1
                self.logger.info(f"CLIP: {label} = {prob:.3f}")
