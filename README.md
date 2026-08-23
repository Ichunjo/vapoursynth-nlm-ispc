# NLM-ISPC

A Non-Local Means (NLMeans) denoise filter for **VapourSynth (API 4)**, implemented with [ISPC](https://github.com/ispc/ispc) for CPU-only execution.

Serves as a CPU-based drop-in replacement for NLMeansCL without requiring OpenCL or GPU runtime. Supports x86 (SSE2, AVX, AVX2, AVX-512) and ARM (NEON).

## Installation

### Python Wheel

```bash
pip install vapoursynth-nlm-ispc
```

## API

```python
nlm_ispc.NLMeans(
    clip: vs.VideoNode,
    d: int | None = 1,
    a: int | None = 2,
    s: int | None = 4,
    h: float | None = 1.2,
    channels: str | None = "AUTO",
    wmode: int | None = 0,
    wref: float | None = 1.0,
    rclip: vs.VideoNode | None = None,
) -> vs.VideoNode:
```

### Parameters

- **`clip`**: Input clip. Supports 8-16-bit integer and 32-bit float formats (Gray, YUV, RGB).
- **`d`**: Temporal radius. `0` = spatial filtering only; `> 0` = temporal radius ($2d + 1$ frames).
- **`a`**: Search window radius. Search area is $(2a + 1) \times (2a + 1)$ pixels.
- **`s`**: Similarity neighborhood patch radius. Patch size is $(2s + 1) \times (2s + 1)$ pixels.
- **`h`**: Filtering strength. Higher values increase smoothing.
- **`channels`**: Color channels to process (`"Y"`, `"UV"`, `"YUV"`, `"RGB"`, `"AUTO"`).
  - `"AUTO"` processes `"RGB"` for RGB clips and `"Y"` (luma) for Gray/YUV clips.
  - `"YUV"` processes all 3 planes (requires YUV444).
- **`wmode`**: Weight function:
  - `0`: Welsch
  - `1`: Modified Bisquare A
  - `2`: Modified Bisquare B
  - `3`: Modified Bisquare C
- **`wref`**: Weight multiplier for the central pixel.
- **`rclip`**: Reference clip used to calculate similarity weights. Must match `clip` format and frame count.

## Compilation

### Requirements

- CMake >= 3.20
- C++20 compliant compiler
- [ISPC](https://github.com/ispc/ispc)

### Build

```bash
uv build --wheel
```
