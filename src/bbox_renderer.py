"""Parse model JSON output and draw bounding boxes onto saved frames."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("src.bbox_renderer")

_EVENT_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (59, 130, 246),
    "vehicle": (249, 115, 22),
    "car": (249, 115, 22),
    "truck": (249, 115, 22),
    "fire": (239, 68, 68),
    "fire or smoke": (239, 68, 68),
    "intrusion": (168, 85, 247),
    "smoke": (107, 114, 128),
}
_DEFAULT_COLOR = (107, 114, 128)


def _color_for(event: str) -> tuple[int, int, int]:
    return _EVENT_COLORS.get(event.lower(), _DEFAULT_COLOR)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _detect_scale(boxes: list[tuple[int, int, int, int]], img_w: int, img_h: int) -> tuple[float, float]:
    """Return (sx, sy) to convert bbox coords to pixel space.

    Qwen VL (and similar) outputs coords in [0, 1000] regardless of image size.
    If any coord exceeds the image dimensions it can't be pixels — scale from 1000.
    If all coords are ≤ 1.0 treat as normalized [0, 1].
    """
    if not boxes:
        return 1.0, 1.0
    max_val = max(v for box in boxes for v in box)
    if max_val <= 1.0:
        return float(img_w), float(img_h)
    if max_val > max(img_w, img_h):
        return img_w / 1000.0, img_h / 1000.0
    return 1.0, 1.0


def _scale_box(
    box: tuple[int, int, int, int], sx: float, sy: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(img_w, int(x1 * sx))),
        max(0, min(img_h, int(y1 * sy))),
        max(0, min(img_w, int(x2 * sx))),
        max(0, min(img_h, int(y2 * sy))),
    )


def _extract_boxes(raw: Any) -> list[tuple[int, int, int, int]]:
    if not isinstance(raw, list) or not raw:
        return []
    if isinstance(raw[0], (int, float)):
        if len(raw) >= 4:
            return [(int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))]
        return []
    result = []
    for box in raw:
        if isinstance(box, list) and len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
            result.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return result


def parse_detections(output_text: str) -> list[dict[str, Any]]:
    """Parse model JSON output into a list of ``{event, boxes}`` dicts.

    Handles four output structures:
    - ``{"detections": [{event, bbox}, ...]}``
    - ``{"events": [{type, bbox}, ...]}``
    - ``{"detections": {"<event>": {"localization": [{bbox}, ...]}}}``
    - ``{"<event>": [[x1,y1,x2,y2], ...], ...}``
    """
    cleaned = _strip_fences(output_text)
    try:
        data = json.loads(cleaned)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    detections: list[dict[str, Any]] = []

    if isinstance(data.get("detections"), list):
        for item in data["detections"]:
            if not isinstance(item, dict):
                continue
            event = str(item.get("event") or item.get("type") or "unknown")
            boxes = _extract_boxes(item.get("bbox") or item.get("bounding_boxes"))
            detections.append({"event": event, "boxes": boxes})
        return detections

    if isinstance(data.get("events"), list):
        for item in data["events"]:
            if not isinstance(item, dict):
                continue
            event = str(item.get("type") or item.get("event") or "unknown")
            boxes = _extract_boxes(item.get("bbox") or item.get("bounding_boxes"))
            detections.append({"event": event, "boxes": boxes})
        return detections

    if isinstance(data.get("detections"), dict):
        for event, value in data["detections"].items():
            if not isinstance(value, dict) or value.get("presence") is False:
                continue
            boxes: list[tuple[int, int, int, int]] = []
            for loc in value.get("localization", []):
                if isinstance(loc, dict):
                    boxes.extend(_extract_boxes(loc.get("bbox")))
            detections.append({"event": event, "boxes": boxes})
        return detections

    _SKIP = {"notes", "description", "general_observation_reference", "general_observation_current"}
    for key, value in data.items():
        if key in _SKIP or not isinstance(value, list):
            continue
        boxes = _extract_boxes(value)
        detections.append({"event": key, "boxes": boxes})

    return detections


def annotate_frame(image: Image.Image, output_text: str) -> Image.Image:
    """Return a copy of *image* with bounding boxes and labels drawn."""
    drawable = [(d["event"], d["boxes"]) for d in parse_detections(output_text) if d["boxes"]]
    if not drawable:
        return image

    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    line_w = max(2, img.width // 800)
    font_size = max(14, img.width // 120)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    all_boxes = [box for _, boxes in drawable for box in boxes]
    sx, sy = _detect_scale(all_boxes, img.width, img.height)

    for event, boxes in drawable:
        rgb = _color_for(event)

        for raw_box in boxes:
            x1, y1, x2, y2 = _scale_box(raw_box, sx, sy, img.width, img.height)
            draw.rectangle([x1, y1, x2, y2], fill=rgb + (51,), outline=rgb + (220,), width=line_w)

            bbox_dims = font.getbbox(event)
            tw = bbox_dims[2] - bbox_dims[0]
            th = bbox_dims[3] - bbox_dims[1]
            pad = max(2, font_size // 6)
            lx = x1
            above = y1 - th - pad * 2 - line_w
            ly = above if above >= 0 else y2 + line_w
            draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=rgb + (200,))
            draw.text((lx + pad, ly + pad), event, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def annotate_run(frames_dir: Path, samples: list[Any]) -> int:
    """Overwrite frames in *frames_dir* with annotated versions. Returns count annotated."""
    if not frames_dir.exists():
        return 0
    count = 0
    for s in samples:
        if getattr(s, "status", None) != "ok" or not getattr(s, "output_text", None):
            continue
        path = frames_dir / f"feed{s.feed_id:02d}_frame{s.frame_index:04d}.jpg"
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            annotated = annotate_frame(img, s.output_text)
            annotated.save(path, format="JPEG", quality=90)
            count += 1
        except Exception as exc:
            log.warning("failed to annotate %s: %s", path.name, exc)
    return count


if __name__ == "__main__":
    import sys

    from src.metrics import MetricSample

    if len(sys.argv) < 2:
        print("usage: python -m src.bbox_renderer <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    outputs_path = run_dir / "outputs.json"
    if not outputs_path.exists():
        print(f"outputs.json not found in {run_dir}")
        sys.exit(1)

    samples = [MetricSample(**r) for r in json.loads(outputs_path.read_text())]
    n = annotate_run(run_dir / "frames", samples)
    print(f"annotated {n} frames in {run_dir / 'frames'}")
