/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

#include "utils/BaseUtil.h"
#include "EbookImagePaper.h"
#include "utils/UtAssert.h"

static EbookPaperColors TestColors() {
    EbookPaperColors c;
    c.bgR = 17;
    c.bgG = 17;
    c.bgB = 17;
    c.textR = 230;
    c.textG = 225;
    c.textB = 216;
    return c;
}

static void PutPx(u8* samples, int comps, int stride, int x, int y, int r, int g, int b, int a = 255) {
    u8* p = samples + (i64)y * stride + x * comps;
    p[0] = (u8)r;
    p[1] = (u8)g;
    p[2] = (u8)b;
    if (comps == 4) {
        p[3] = (u8)a;
    }
}

static u8* Px(u8* samples, int comps, int stride, int x, int y) {
    return samples + (i64)y * stride + x * comps;
}

static int Near(int got, int want, int tol = 8) {
    int d = got - want;
    if (d < 0) {
        d = -d;
    }
    return d <= tol;
}

static void LineArtKeepsColorAndDropsPaper() {
    constexpr int w = 24;
    constexpr int h = 24;
    constexpr int comps = 4;
    u8 samples[w * h * comps];
    memset(samples, 0, sizeof(samples));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            PutPx(samples, comps, w * comps, x, y, 255, 255, 255);
        }
    }
    for (int x = 2; x < 22; x++) {
        PutPx(samples, comps, w * comps, x, 12, 128, 40, 168);
    }
    PutPx(samples, comps, w * comps, 8, 8, 0, 0, 0);
    PutPx(samples, comps, w * comps, 9, 8, 210, 210, 210);

    EbookPaperColors colors = TestColors();
    utassert(EbookKnockoutPaperBackground(samples, w, h, comps, w * comps, colors));

    u8* paper = Px(samples, comps, w * comps, 1, 1);
    utassert(paper[0] == colors.bgR && paper[1] == colors.bgG && paper[2] == colors.bgB);

    u8* ink = Px(samples, comps, w * comps, 10, 12);
    utassert(Near(ink[0], 128) && Near(ink[1], 40) && Near(ink[2], 168));

    u8* black = Px(samples, comps, w * comps, 8, 8);
    utassert(Near(black[0], colors.textR) && Near(black[1], colors.textG) && Near(black[2], colors.textB));

    // A light gray fringe must not stay light gray on the dark page.
    u8* fringe = Px(samples, comps, w * comps, 9, 8);
    utassert(fringe[0] < 180 && fringe[1] < 180 && fringe[2] < 180);
}

static void CreamPaperIsKnockedOut() {
    constexpr int w = 16;
    constexpr int h = 16;
    constexpr int comps = 3;
    u8 samples[w * h * comps];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            PutPx(samples, comps, w * comps, x, y, 248, 240, 226);
        }
    }
    for (int y = 4; y < 12; y++) {
        PutPx(samples, comps, w * comps, 8, y, 20, 20, 20);
    }
    EbookPaperColors colors = TestColors();
    utassert(EbookKnockoutPaperBackground(samples, w, h, comps, w * comps, colors));
    u8* paper = Px(samples, comps, w * comps, 0, 0);
    utassert(paper[0] == colors.bgR && paper[1] == colors.bgG && paper[2] == colors.bgB);
    u8* ink = Px(samples, comps, w * comps, 8, 6);
    utassert(Near(ink[0], colors.textR, 40));
    utassert(ink[0] > 140);
}

static void PhotographIsLeftAlone() {
    constexpr int w = 20;
    constexpr int h = 20;
    constexpr int comps = 3;
    u8 samples[w * h * comps];
    u8 original[w * h * comps];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int r = 40 + (x * 9 + y * 3) % 160;
            int g = 30 + (x * 4 + y * 11) % 150;
            int b = 20 + (x * 7 + y * 5) % 120;
            PutPx(samples, comps, w * comps, x, y, r, g, b);
        }
    }
    for (int x = 0; x < w; x++) {
        PutPx(samples, comps, w * comps, x, 0, 255, 255, 255);
        PutPx(samples, comps, w * comps, x, h - 1, 255, 255, 255);
    }
    memcpy(original, samples, sizeof(samples));
    utassert(!EbookKnockoutPaperBackground(samples, w, h, comps, w * comps, TestColors()));
    utassert(memcmp(samples, original, sizeof(samples)) == 0);
}

static void TransparentPixelStaysTransparent() {
    constexpr int w = 12;
    constexpr int h = 12;
    constexpr int comps = 4;
    u8 samples[w * h * comps];
    memset(samples, 0, sizeof(samples));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            PutPx(samples, comps, w * comps, x, y, 255, 255, 255, 255);
        }
    }
    PutPx(samples, comps, w * comps, 2, 2, 255, 255, 255, 0);
    for (int x = 3; x < 9; x++) {
        PutPx(samples, comps, w * comps, x, 6, 0, 0, 0, 255);
    }
    utassert(EbookKnockoutPaperBackground(samples, w, h, comps, w * comps, TestColors()));
    u8* clear = Px(samples, comps, w * comps, 2, 2);
    utassert(clear[3] == 0);
    utassert(clear[0] == 255 && clear[1] == 255 && clear[2] == 255);
}

static void PortraitOnWhiteKeepsTheFace() {
    constexpr int w = 20;
    constexpr int h = 20;
    constexpr int comps = 3;
    u8 samples[w * h * comps];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            if (border) {
                PutPx(samples, comps, w * comps, x, y, 255, 255, 255);
            } else if (y < 8) {
                PutPx(samples, comps, w * comps, x, y, 28, 22, 20);
            } else {
                PutPx(samples, comps, w * comps, x, y, 214, 164, 142);
            }
        }
    }
    EbookPaperColors colors = TestColors();
    utassert(EbookKnockoutPaperBackground(samples, w, h, comps, w * comps, colors));
    u8* paper = Px(samples, comps, w * comps, 0, 10);
    utassert(paper[0] == colors.bgR && paper[1] == colors.bgG && paper[2] == colors.bgB);
    u8* skin = Px(samples, comps, w * comps, 10, 14);
    utassert(skin[0] == 214 && skin[1] == 164 && skin[2] == 142);
    u8* hair = Px(samples, comps, w * comps, 10, 3);
    utassert(hair[0] == 28 && hair[1] == 22 && hair[2] == 20);
}

void EbookImagePaper_UnitTests() {
    LineArtKeepsColorAndDropsPaper();
    CreamPaperIsKnockedOut();
    PhotographIsLeftAlone();
    TransparentPixelStaysTransparent();
    PortraitOnWhiteKeepsTheFace();
}
