# VlmTask
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

"""Backend-agnostic Vision-Language Model task.

`VlmTask` runs the VLM forward pass and serializes the generated text into the
payload that follows the GST-VLM: header. It uses only json and the engine; it
never touches the buffer. The backend element shell does the frame read and
appends the payload via the frameio facade, prepending the header.

Contract expected from the host element:
  self.engine        - the ML engine (provided by MLEngineMixin)
  self.prompt        - user prompt (a backend-declared property)
  self.system_prompt - optional system prompt (a backend-declared property)
  self.max_tokens    - max tokens to generate (a backend-declared property)
  self.temperature   - sampling temperature (a backend-declared property)
"""

import json


class VlmTask:
    """Inference + payload serialization, independent of any framework."""

    def forward(self, frame):
        return self.engine.do_forward(
            frame,
            prompt=self.prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def decode(self, text):
        """Serialize a generated caption into ``(None, payload_bytes)``.

        ``payload_bytes`` is the JSON result that follows the GST-VLM: header;
        the shell prepends the header when appending the blob.
        """
        result = {"text": text}
        payload = json.dumps(result).encode("utf-8")
        return None, payload
