#!/usr/bin/env python3
"""fgfill - replace the people in photos or video with a fill (random noise by default).

Uses MediaPipe's selfie segmentation models (selfie_segmenter.tflite), run through
OpenCV's TFLite loader, so the only dependencies are opencv-python and numpy.
Videos are written as frames, split into shots; ffmpeg is optional but decodes more codecs.

Examples:
  fgfill.py clip.mp4                      # -> clip_frames/{original,filled}/shot_NNN/
  fgfill.py photos/                       # -> photos_filled/
  fgfill.py a.jpg b.jpg -o out/ --fill color:#00ff00
  fgfill.py clip.mp4 --fill blur:41 --invert   # blur the background instead
"""

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/{0}/float16/latest/{0}.tflite"
MODELS = {
    # name: (file stem, input height, input width)
    "square": ("selfie_segmenter", 256, 256),
    "landscape": ("selfie_segmenter_landscape", 144, 256),
}
MODEL_DIRS = [Path(__file__).resolve().parent / "models", Path.home() / ".cache" / "fgfill"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --- segmentation -----------------------------------------------------------

class Segmenter:
    def __init__(self, name):
        stem, self.h, self.w = MODELS[name]
        path = self._find_or_download(stem)
        self.net = cv2.dnn.readNetFromTFLite(str(path))

    @staticmethod
    def _find_or_download(stem):
        for d in MODEL_DIRS:
            p = d / f"{stem}.tflite"
            if p.exists():
                return p
        d = MODEL_DIRS[-1]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stem}.tflite"
        log(f"downloading {stem}.tflite -> {p}")
        urllib.request.urlretrieve(MODEL_URL.format(stem), p)
        return p

    def __call__(self, bgr):
        """Return a float32 person-confidence mask (0..1) at the frame's resolution."""
        x = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.net.setInput(x.transpose(2, 0, 1)[None])
        m = self.net.forward()[0, 0]
        return cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def pick_model(name, frame_shape):
    if name != "auto":
        return name
    h, w = frame_shape[:2]
    return "landscape" if w / h > 1.3 else "square"


class Ensemble:
    """Per-pixel max over both models, each run on the frame and its mirror image.

    The models miss different bits of a person (limbs, people at the edges, unusual poses),
    so taking the max catches more of them, at four times the cost of a single pass.
    """

    def __init__(self):
        self.segmenters = [Segmenter(name) for name in MODELS]

    def __call__(self, bgr):
        flipped = bgr[:, ::-1]
        masks = [seg(bgr) for seg in self.segmenters]
        masks += [seg(flipped)[:, ::-1] for seg in self.segmenters]
        return np.max(masks, axis=0)


# --- fills ------------------------------------------------------------------

def parse_hex(s):
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad colour {s!r}, expected #rrggbb")
    r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    return np.array([b, g, r], np.uint8)


class Fill:
    """Builds the replacement pixels for a frame. Spec is 'kind[:arg]'."""

    def __init__(self, spec, seed=None, static=False):
        self.kind, _, self.arg = spec.partition(":")
        self.rng = np.random.default_rng(seed)
        self.static = static
        self._cache = None
        if self.kind == "color":
            self.color = parse_hex(self.arg or "000000")
        elif self.kind == "image":
            self.image = cv2.imread(self.arg)
            if self.image is None:
                raise ValueError(f"can't read fill image {self.arg!r}")
        elif self.kind in ("blur", "pixelate"):
            self.n = int(self.arg) if self.arg else (51 if self.kind == "blur" else 24)
        elif self.kind not in ("noise", "mono"):
            raise ValueError(f"unknown fill {spec!r}")

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if self.kind in ("noise", "mono"):
            if self.static and self._cache is not None and self._cache.shape[:2] == (h, w):
                return self._cache
            if self.kind == "noise":
                out = self.rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            else:
                out = np.repeat(self.rng.integers(0, 256, (h, w, 1), dtype=np.uint8), 3, axis=2)
            self._cache = out
            return out
        if self.kind == "color":
            return np.broadcast_to(self.color, frame.shape)
        if self.kind == "image":
            return cv2.resize(self.image, (w, h), interpolation=cv2.INTER_AREA)
        if self.kind == "blur":
            k = self.n | 1
            return cv2.GaussianBlur(frame, (k, k), 0)
        if self.kind == "pixelate":
            small = cv2.resize(frame, (max(1, w // self.n), max(1, h // self.n)), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# --- per-frame processing ---------------------------------------------------

class Processor:
    def __init__(self, args):
        self.args = args
        self.fill = Fill(args.fill, seed=args.seed, static=args.static)
        self.segmenters = {}
        self.prev_mask = None

    def segmenter(self, frame):
        name = pick_model(self.args.model, frame.shape)
        if name not in self.segmenters:
            self.segmenters[name] = Ensemble() if name == "all" else Segmenter(name)
        return self.segmenters[name]

    def mask(self, frame):
        segment = self.segmenter(frame)
        m = segment(frame)
        if self.args.smooth and self.prev_mask is not None and self.prev_mask.shape == m.shape:
            m = self.args.smooth * self.prev_mask + (1 - self.args.smooth) * m
        self.prev_mask = m
        if self.args.threshold is not None:
            m = (m > self.args.threshold).astype(np.float32)
        if self.args.dilate:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.args.dilate + 1,) * 2)
            m = cv2.dilate(m, k)
        if self.args.feather:
            k = 2 * self.args.feather + 1
            m = cv2.GaussianBlur(m, (k, k), 0)
        if self.args.invert:
            m = 1 - m
        return m

    def __call__(self, frame):
        m = self.mask(frame)
        if self.args.mask_only:
            return cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        fill = self.fill(frame)
        a = m[..., None]
        return (frame * (1 - a) + fill * a).astype(np.uint8)

    def reset(self):
        self.prev_mask = None


class ShotDetector:
    """Flags hard cuts by the Bhattacharyya distance between consecutive frames' hue/saturation histograms."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.prev = None

    def is_cut(self, frame):
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        prev, self.prev = self.prev, hist
        if prev is None or not self.threshold:
            return False
        return cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA) > self.threshold


# --- I/O --------------------------------------------------------------------

def process_image(proc, src, dst):
    frame = cv2.imread(str(src))
    if frame is None:
        log(f"skip {src}: not a readable image")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc.reset()
    if not cv2.imwrite(str(dst), proc(frame)):
        log(f"failed to write {dst}")
        return False
    log(f"{src} -> {dst}")
    return True


def probe(src):
    """(width, height, frame count) of the first video stream, as ffmpeg will decode it (rotation applied)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,nb_frames:stream_side_data=rotation",
         str(src)], capture_output=True, text=True)
    streams = json.loads(out.stdout or "{}").get("streams")
    if out.returncode or not streams:
        return None
    s = streams[0]
    w, h = s["width"], s["height"]
    rotation = next((d.get("rotation") for d in s.get("side_data_list", []) if "rotation" in d), 0)
    if abs(int(rotation)) % 180 == 90:
        w, h = h, w
    return w, h, int(s.get("nb_frames") or 0)


def ffmpeg_frames(src, w, h):
    pipe = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:v:0",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = pipe.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        pipe.stdout.close()
        pipe.kill()
        pipe.wait()


def opencv_frames(cap):
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def process_video(proc, src, dst):
    """Write each frame to dst/{original,filled}/shot_NNN/000001.jpg, numbered by frame in the video."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        info = probe(src)
        if info is None:
            log(f"skip {src}: no video stream")
            return False
        w, h, total = info
        frames = ffmpeg_frames(src, w, h)
    else:
        log("ffmpeg not found: decoding with OpenCV")
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            log(f"skip {src}: can't open video")
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames = opencv_frames(cap)
    frame_ext = proc.args.ext or ".jpg"
    shots = ShotDetector(proc.args.shot_threshold)
    shot = 0

    n = 0
    try:
        for frame in frames:
            if shots.is_cut(frame) or n == 0:
                shot += 1
                proc.reset()  # don't smooth masks across a cut
                shot_dir = f"shot_{shot:03d}"
                for sub in ("original", "filled"):
                    (dst / sub / shot_dir).mkdir(parents=True, exist_ok=True)
            out = proc(frame)
            name = f"{n + 1:06d}{frame_ext}"
            if not (cv2.imwrite(str(dst / "original" / shot_dir / name), frame)
                    and cv2.imwrite(str(dst / "filled" / shot_dir / name), out)):
                log(f"failed to write frame {n + 1} to {dst}")
                return False
            n += 1
            if n % 25 == 0 or n == total:
                log(f"{src.name}: {n}/{total or '?'} frames")
    finally:
        frames.close()
    if n == 0:
        log(f"failed {src}: no frames decoded")
        return False
    log(f"{src} -> {dst} ({n} frames, {shot} shot{'s' if shot != 1 else ''})")
    return True


def collect(inputs):
    """Expand inputs into (path, base_dir) pairs; base_dir is set for files found inside a directory."""
    items = []
    for p in map(Path, inputs):
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
            if not files:
                log(f"no images or videos in {p}")
            items += [(f, p) for f in files]
        elif p.exists():
            items.append((p, None))
        else:
            log(f"not found: {p}")
    return items


def output_path(src, base, out, multiple, ext):
    suffix = src.suffix if ext is None else ext
    if out is None:
        if base is not None:
            return base.with_name(base.name + "_filled") / src.relative_to(base).with_suffix(suffix)
        tag = "_frames" if ext == "" else "_filled"
        return src.with_name(f"{src.stem}{tag}{suffix}")
    out = Path(out)
    if ext == "" and not multiple:  # a single video's frames go straight into the -o directory
        return out
    if multiple or out.is_dir() or str(out).endswith("/") or not out.suffix:
        rel = src.relative_to(base) if base is not None else Path(src.name)
        return out / rel.with_suffix(suffix)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Replace people (the foreground) in photos or video with a fill, using MediaPipe selfie segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="fills: noise (default, colour static), mono (greyscale static), color:#rrggbb,\n"
               "       blur[:kernel], pixelate[:block], image:path/to/picture.jpg")
    ap.add_argument("inputs", nargs="+", help="video files, image files and/or directories of images")
    ap.add_argument("-o", "--output", help="output file or frames directory (single input), or directory; default: <name>_filled next to the input")
    ap.add_argument("-f", "--fill", default="noise", help="what to put in place of the foreground (default: noise)")
    ap.add_argument("--invert", action="store_true", help="fill the background instead of the person")
    ap.add_argument("-t", "--threshold", type=float, default=0.25,
                    help="mask cutoff 0..1, lower catches more (default 0.25); use --soft for the raw confidence mask")
    ap.add_argument("--soft", dest="threshold", action="store_const", const=None,
                    help="blend by model confidence instead of a hard cutoff")
    ap.add_argument("--dilate", type=int, default=0, metavar="PX", help="grow the mask by PX pixels")
    ap.add_argument("--feather", type=int, default=2, metavar="PX", help="soften mask edges by PX pixels (default 2)")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="A",
                    help="video only: blend each mask with the previous one (0..0.95) to reduce flicker")
    ap.add_argument("-m", "--model", choices=["all", "auto", "square", "landscape"], default="all",
                    help="all (default): combine both models and their mirror images, catches the most; "
                         "auto: one model, landscape for wide frames (4x faster); or force square/landscape")
    ap.add_argument("--static", action="store_true", help="noise fills: use the same noise every frame")
    ap.add_argument("--seed", type=int, help="random seed for noise fills")
    ap.add_argument("--mask-only", action="store_true", help="output the mask itself (white = filled)")
    ap.add_argument("--ext", help="output image extension, e.g. .png (video frames default to .jpg)")
    ap.add_argument("--shot-threshold", type=float, default=0.3, metavar="D",
                    help="histogram distance 0..1 between consecutive frames that counts as a cut (default 0.3); "
                         "lower splits more, 0 keeps the whole video as one shot")
    args = ap.parse_args(argv)

    if not 0 <= args.smooth < 1:
        ap.error("--smooth must be in [0, 1)")
    if args.ext and not args.ext.startswith("."):
        args.ext = "." + args.ext

    try:
        proc = Processor(args)
    except ValueError as e:
        ap.error(str(e))

    items = collect(args.inputs)
    if not items:
        return 1
    multiple = len(items) > 1
    failed = 0
    for src, base in items:
        is_video = src.suffix.lower() in VIDEO_EXTS
        ext = "" if is_video else args.ext  # a video's frames go in a directory named after it
        dst = output_path(src, base, args.output, multiple, ext)
        if dst.resolve() == src.resolve():
            log(f"skip {src}: output would overwrite input")
            failed += 1
            continue
        ok = process_video(proc, src, dst) if is_video else process_image(proc, src, dst)
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
