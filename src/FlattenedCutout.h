/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

// High-precision test for a flattened cutout: an isolated subject on an empty
// near-white canvas. Uncertain images, engravings, and full scenes are rejected.
// A rejection leaves the caller's pixels alone.

struct FlattenedCutoutScores {
    float pureWhite = 0;
    float nearWhite = 0;
    float borderWhite = 0;
    float canvas = 0;
    float foreground = 0;
    float edgeContact = 0;
    float contourSharp = 0;
    float grayInk = 0;
    float borderStd = 0;
    float fillRatio = 0;
    const char* reason = nullptr;
};

// rgb is packed RGB. alpha is w*h bytes, written only when this returns true.
// On success, a 1-3px fringe in rgb is un-matted against white.
bool TryFlattenedCutout(u8* rgb, int w, int h, int stride, u8* alpha, FlattenedCutoutScores* scores);

// Same test as TryFlattenedCutout, without writing a mask. Used to skip a
// full-resolution flood when a downscaled copy is already a clear reject.
bool ClassifyFlattenedCutout(const u8* rgb, int w, int h, int stride, FlattenedCutoutScores* scores);

// Box-filter so the longest side is maxSide. Caller frees the returned RGB.
// Returns null when the image is already small enough; *dw and *dh are then w and h.
u8* FlattenedCutoutScaledRgb(const u8* rgb, int w, int h, int stride, int maxSide, int* dw, int* dh);
