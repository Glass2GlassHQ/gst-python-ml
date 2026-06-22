# DemucsEngine
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

from .pytorch_engine import PyTorchEngine


class DemucsEngine(PyTorchEngine):
    def __init__(self):
        super().__init__()
        self.sample_rate = 0

    def do_load_model(self, model_name, **kwargs):
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

        if not model_name:
            return
        self.logger.info(f"Loading Demucs model on device: {self.device}")
        bundle = (
            HDEMUCS_HIGH_MUSDB_PLUS  # You can choose other bundles like DEMUCS_MUSDB
        )
        self.model = bundle.get_model()
        if hasattr(self.model, "to") and callable(getattr(self.model, "to")):
            self.model = self.model.to(self.device)
        self.sample_rate = bundle.sample_rate  # 44100 Hz
        self.sources = self.model.sources  # ['drums', 'bass', 'other', 'vocals']

    def separate_sources(
        self,
        mix,
        segment=10.0,
        overlap=0.1,
    ):
        import torch
        from torchaudio.transforms import Fade

        device = mix.device
        batch, channels, length = mix.shape
        chunk_len = int(self.sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = int(overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(self.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = self.model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = overlap_frames
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final
