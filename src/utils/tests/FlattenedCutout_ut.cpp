/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

#include "utils/BaseUtil.h"

#include "FlattenedCutout.h"
#include "utils/UtAssert.h"

static void Put(u8* rgb, int w, int x, int y, int r, int g, int b) {
    u8* p = rgb + ((i64)y * w + x) * 3;
    p[0] = (u8)r;
    p[1] = (u8)g;
    p[2] = (u8)b;
}

static void Fill(u8* rgb, int w, int h, int r, int g, int b) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Put(rgb, w, x, y, r, g, b);
        }
    }
}

static bool Run(u8* rgb, int w, int h, u8* alpha, FlattenedCutoutScores* scores) {
    return TryFlattenedCutout(rgb, w, h, w * 3, alpha, scores);
}

static void ColoredBlobOnWhiteIsCutOut() {
    constexpr int w = 80;
    constexpr int h = 80;
    u8 rgb[w * h * 3];
    Fill(rgb, w, h, 255, 255, 255);
    int cx = 40;
    int cy = 40;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int dx = x - cx;
            int dy = y - cy;
            int d2 = dx * dx + dy * dy;
            if (d2 <= 22 * 22) {
                Put(rgb, w, x, y, 255, 0, 0);
            } else if (d2 <= 24 * 24) {
                Put(rgb, w, x, y, 255, 190, 190);
            }
        }
    }
    u8 alpha[w * h];
    memset(alpha, 0x5A, sizeof(alpha));
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(ok);
    utassert(alpha[0] == 0);
    utassert(alpha[cy * w + cx] == 255);
    u8* center = rgb + ((i64)cy * w + cx) * 3;
    utassert(center[0] == 255);
    utassert(center[1] == 0);
    utassert(center[2] == 0);
    // Pink fringe loses the white matte and becomes partly transparent.
    bool fringe = false;
    for (int y = 0; y < h && !fringe; y++) {
        for (int x = 0; x < w; x++) {
            u8 a = alpha[y * w + x];
            if (a > 0 && a < 255) {
                u8* p = rgb + ((i64)y * w + x) * 3;
                utassert(p[0] > 200);
                utassert(p[1] < 40);
                fringe = true;
                break;
            }
        }
    }
    utassert(fringe);
}

static void EnclosedWhiteStaysOpaque() {
    constexpr int w = 80;
    constexpr int h = 80;
    u8 rgb[w * h * 3];
    Fill(rgb, w, h, 252, 252, 252);
    int cx = 40;
    int cy = 40;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int dx = x - cx;
            int dy = y - cy;
            int d2 = dx * dx + dy * dy;
            if (d2 <= 26 * 26 && d2 > 10 * 10) {
                Put(rgb, w, x, y, 120, 60, 30);
            }
        }
    }
    u8 alpha[w * h];
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(ok);
    utassert(alpha[0] == 0);
    utassert(alpha[cy * w + cx] == 255);
    utassert(alpha[cy * w + (cx + 18)] == 255);
}

static void FadingEngravingStaysUntouched() {
    constexpr int w = 96;
    constexpr int h = 96;
    u8 rgb[w * h * 3];
    Fill(rgb, w, h, 255, 255, 255);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int inset = x;
            if (y < inset) {
                inset = y;
            }
            if (w - 1 - x < inset) {
                inset = w - 1 - x;
            }
            if (h - 1 - y < inset) {
                inset = h - 1 - y;
            }
            if (inset < 8) {
                continue;
            }
            if ((x % 4) != 0 && (y % 4) != 0) {
                continue;
            }
            int depth = inset - 8;
            int lum = 228;
            if (depth >= 20 && x > 36 && x < 60 && y > 36 && y < 60) {
                lum = 24;
            }
            Put(rgb, w, x, y, lum, lum, lum);
        }
    }
    u8 alpha[w * h];
    memset(alpha, 0x5A, sizeof(alpha));
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(!ok);
    utassert(s.borderWhite >= 0.88f);
    utassert(alpha[w / 2] == 0x5A);
    utassert(rgb[0] == 255);
}

static void FullBleedSceneStaysUntouched() {
    constexpr int w = 64;
    constexpr int h = 64;
    u8 rgb[w * h * 3];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Put(rgb, w, x, y, (x * 17 + y * 3) & 255, (x * 5 + 40) & 255, (y * 13 + 20) & 255);
        }
    }
    u8 alpha[w * h];
    memset(alpha, 0x5A, sizeof(alpha));
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(!ok);
    utassert(alpha[0] == 0x5A);
    utassert(rgb[0] == ((0 * 17 + 0 * 3) & 255));
}

static void TexturedBorderStaysUntouched() {
    constexpr int w = 72;
    constexpr int h = 72;
    u8 rgb[w * h * 3];
    Fill(rgb, w, h, 255, 255, 255);
    for (int y = 16; y < 56; y++) {
        for (int x = 16; x < 56; x++) {
            int dx = x - 36;
            int dy = y - 36;
            if (dx * dx + dy * dy < 16 * 16) {
                Put(rgb, w, x, y, 20, 90, 200);
            }
        }
    }
    for (int x = 0; x < w; x++) {
        Put(rgb, w, x, 0, 20, 20, 20);
        Put(rgb, w, x, 1, 180, 40, 40);
        Put(rgb, w, x, h - 1, 30, 30, 90);
        Put(rgb, w, x, h - 2, 200, 80, 20);
    }
    for (int y = 0; y < h; y++) {
        Put(rgb, w, 0, y, 10, 80, 10);
        Put(rgb, w, 1, y, 90, 90, 10);
        Put(rgb, w, w - 1, y, 40, 10, 10);
        Put(rgb, w, w - 2, y, 10, 10, 80);
    }
    u8 alpha[w * h];
    memset(alpha, 0x5A, sizeof(alpha));
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(!ok);
    utassert(alpha[(h / 2) * w + w / 2] == 0x5A);
}

static void RectangularPhotoOnWhiteStaysUntouched() {
    constexpr int w = 80;
    constexpr int h = 80;
    u8 rgb[w * h * 3];
    Fill(rgb, w, h, 255, 255, 255);
    for (int y = 8; y < 72; y++) {
        for (int x = 8; x < 72; x++) {
            Put(rgb, w, x, y, (x * 13 + y * 9) & 200, (x * 3 + 80) & 255, (y * 11 + 30) & 255);
        }
    }
    u8 alpha[w * h];
    memset(alpha, 0x5A, sizeof(alpha));
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(!ok);
    utassert(str::Eq(s.reason, "plate"));
    utassert(alpha[0] == 0x5A);
}

static void SharpBlackSilhouetteIsCutOut() {
    constexpr int w = 80;
    constexpr int h = 80;
    u8 rgb[w * h * 3];
    Fill(rgb, w, h, 255, 255, 255);
    for (int y = 18; y < 64; y++) {
        for (int x = 22; x < 58; x++) {
            int dx = x - 40;
            int dy = y - 42;
            if (dx * dx + (dy * dy) / 2 < 16 * 16) {
                Put(rgb, w, x, y, 8, 8, 8);
            }
        }
    }
    u8 alpha[w * h];
    FlattenedCutoutScores s{};
    bool ok = Run(rgb, w, h, alpha, &s);
    utassert(ok);
    utassert(alpha[0] == 0);
    utassert(alpha[42 * w + 40] == 255);
}

void FlattenedCutout_UnitTests() {
    ColoredBlobOnWhiteIsCutOut();
    EnclosedWhiteStaysOpaque();
    SharpBlackSilhouetteIsCutOut();
    FadingEngravingStaysUntouched();
    FullBleedSceneStaysUntouched();
    TexturedBorderStaysUntouched();
    RectangularPhotoOnWhiteStaysUntouched();
}
