import asyncio
from collections.abc import AsyncIterator

from PIL import Image as PILImage

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class VideoFrameSource:
    """Async generator that yields (frame_index, PIL.Image) at a fixed time interval.

    When max_frames is None, stops when the video ends (no looping).
    When max_frames is set, loops the video until that frame count is reached.
    """

    def __init__(
        self,
        path: str,
        interval_s: float,
        resolution: tuple[int, int] | None = None,
        downscale_factor: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        if not _CV2_AVAILABLE:
            raise RuntimeError("opencv-python-headless is required: uv add opencv-python-headless")
        self._path = path
        self._interval_s = interval_s
        self._resolution = resolution
        self._downscale_factor = downscale_factor
        self._max_frames = max_frames

    def _resize(self, img: PILImage.Image) -> PILImage.Image:
        if self._resolution is not None:
            return img.resize(self._resolution, PILImage.LANCZOS)
        if self._downscale_factor is not None and self._downscale_factor != 1.0:
            w = max(1, int(img.width * self._downscale_factor))
            h = max(1, int(img.height * self._downscale_factor))
            return img.resize((w, h), PILImage.LANCZOS)
        return img

    async def frames(self) -> AsyncIterator[tuple[int, PILImage.Image]]:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration_ms = (total_frames / fps) * 1000.0
        interval_ms = self._interval_s * 1000.0

        frame_index = 0
        pos_ms = 0.0

        try:
            while True:
                if self._max_frames is not None and frame_index >= self._max_frames:
                    break

                if video_duration_ms > 0 and pos_ms >= video_duration_ms:
                    if self._max_frames is None:
                        break
                    pos_ms = pos_ms % video_duration_ms

                cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
                ret, frame_bgr = cap.read()
                if not ret:
                    if self._max_frames is None:
                        break
                    # Loop mode: retry from beginning on failed read
                    pos_ms = 0.0
                    cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = self._resize(PILImage.fromarray(frame_rgb))

                yield frame_index, img
                frame_index += 1
                pos_ms += interval_ms

                await asyncio.sleep(self._interval_s)
        finally:
            cap.release()
