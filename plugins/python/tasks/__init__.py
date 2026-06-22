# Portable ML task logic
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""Backend-agnostic ML task logic.

Each module here holds the inference and result-handling steps for one kind of
element (object detection, ...), expressed only in terms of the ML engine,
numpy results, and the analytics facade (`backend.analytics`). A backend
combines a task mixin with a framework element shell, which supplies the
per-buffer glue (do_transform_ip / process, pad templates, buffer I/O).
"""
