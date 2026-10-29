# fgfill

Replace the people in a video or a set of photos with a fill (random noise by default)
using MediaPipe's selfie segmentation model, and save the result as photos: one image per
input photo, and for each input video `original/` and `filled/` directories, each with a
`shot_NNN/` directory per shot (split at hard cuts) of numbered frames.

The MediaPipe `selfie_segmenter` / `selfie_segmenter_landscape` TFLite models (in `models/`,
re-downloaded to `~/.cache/fgfill` if missing) run through OpenCV's TFLite loader, so the
full `mediapipe` package isn't needed. `ffmpeg`/`ffprobe` are optional but recommended: with
them any video codec ffmpeg can decode works (including AV1); without them, OpenCV's decoder is used.

## Install

    pip install -r requirements.txt

## Usage

    ./fgfill.py clip.mp4                         # -> clip_frames/original/shot_001/000001.jpg, ...
                                                 #    clip_frames/filled/shot_001/000001.jpg, ...
                                                 #    clip_frames/original/shot_002/000029.jpg, ...
    ./fgfill.py clip.mp4 -o frames --ext .png    # -> frames/{original,filled}/shot_NNN/000001.png, ...
    ./fgfill.py photos/                          # -> photos_filled/ (recursive, keeps structure)
    ./fgfill.py a.jpg b.png -o out/              # several files into a directory
    ./fgfill.py clip.mp4 --fill mono --static    # one frozen frame of greyscale static
    ./fgfill.py clip.mp4 --fill color:#00ff00    # green-screen the person
    ./fgfill.py clip.mp4 --fill blur:41 --invert # keep the person, blur the background
    ./fgfill.py clip.mp4 --mask-only             # just the mask

Fills: `noise` (default, colour static, new each frame), `mono`, `color:#rrggbb`,
`blur[:kernel]`, `pixelate[:block]`, `image:path`.

Segmentation: by default (`-m auto`) the MediaPipe selfie model matching the frame's shape
(landscape for frames wider than 1.3:1, otherwise square) is run on the frame and on its mirror
image, the masks combined, and cut off at `-t 0.5`. For more inclusive masks use `-m all`
(both models) and/or a lower threshold such as `-t 0.25`; these catch more of each person,
e.g. through motion blur, but also more background.

Shots: a cut is where the hue/saturation histogram distance between consecutive frames exceeds
`--shot-threshold` (default 0.3, calibrated on broadcast sports footage where real cuts scored
0.33-0.75 and fast motion within a shot stayed below 0.23). Frame numbers count from the start
of the video, not the shot. Mask smoothing is reset at each cut. `--shot-threshold 0` disables
splitting.

Mask tuning: `-t/--threshold` (default 0.5, lower catches more) or `--soft` to blend by confidence,
`--dilate PX` to grow the mask, `--feather PX` (default 2) to soften edges,
`--smooth 0.5` to reduce flicker in video, `-m square|landscape` to force one model.

## Limits

It's a *selfie* model: best on one or two people reasonably large in frame. Small,
distant or partly hidden figures can be missed, and some non-people may be picked up.

## Credits

The segmentation models in `models/` are MediaPipe's selfie segmenter models
(https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter), released by Google under the Apache 2.0 licence.
