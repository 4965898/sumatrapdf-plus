/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

#include "utils/BaseUtil.h"

#include "FlattenedCutout.h"

#include <math.h>
#include <stdlib.h>

// Stage 1 only drops obvious non-cutouts. Acceptance also requires a
// border-connected empty canvas, a subject that does not touch the frame,
// and a hard silhouette. A soft gray fade is an engraving, not a cutout.

static constexpr int kMinSide = 32;
static constexpr int kPaperMin = 244;
static constexpr int kPaperChroma = 12;
static constexpr float kBorderMin = 0.88f;
static constexpr float kSideMin = 0.80f;
static constexpr float kCanvasMin = 0.10f;
static constexpr float kCanvasMax = 0.97f;
static constexpr float kInkMin = 0.03f;
static constexpr float kInkMax = 0.90f;
static constexpr float kSharpMin = 0.72f;
static constexpr int kSharpDrop = 40;
static constexpr int kSharpChroma = 24;
static constexpr float kGrayInkEngrave = 0.78f;
static constexpr float kSharpForGray = 0.90f;
static constexpr float kLightGrayFrac = 0.28f;
static constexpr int kLightGrayLo = 168;
static constexpr int kLightGrayHi = 242;
static constexpr int kGrayChroma = 16;
static constexpr float kPlateFill = 0.90f;
static constexpr float kBusy = 0.50f;
static constexpr float kBorderMadMax = 14.f;
static constexpr int kEdgeBand = 3;
static constexpr float kEdgeInkMax = 0.03f;
// A photo cutout (owl, kitten) has a soft fur edge, so the 1px jump test fails.
// Accept it when the canvas is a clean white frame and the subject is a solid
// shape that does not sit on that frame. Line engravings stay busy or pale.
static constexpr float kPhotoBorder = 0.95f;
static constexpr float kPhotoMad = 8.f;
static constexpr float kPhotoBusy = 0.12f;
static constexpr float kPhotoLight = 0.22f;
static constexpr int kFringeDist = 3;
static constexpr int kFringeLum = 176;

struct Pix {
    int r, g, b;
};

static Pix At(const u8* rgb, int stride, int x, int y) {
    const u8* p = rgb + (i64)y * stride + (i64)x * 3;
    return Pix{p[0], p[1], p[2]};
}

static int Chroma(Pix p) {
    int mx = p.r;
    int mn = p.r;
    if (p.g > mx) {
        mx = p.g;
    }
    if (p.b > mx) {
        mx = p.b;
    }
    if (p.g < mn) {
        mn = p.g;
    }
    if (p.b < mn) {
        mn = p.b;
    }
    return mx - mn;
}

static int Lum(Pix p) {
    return (p.r + p.g + p.b) / 3;
}

static int MinC(Pix p) {
    int mn = p.r;
    if (p.g < mn) {
        mn = p.g;
    }
    if (p.b < mn) {
        mn = p.b;
    }
    return mn;
}

static bool IsPaper(Pix p) {
    return MinC(p) >= kPaperMin && Chroma(p) <= kPaperChroma;
}

struct Measure {
    u8* tag = nullptr;
    int* queue = nullptr;
    int w = 0;
    int h = 0;
    bool accept = false;

    ~Measure() {
        free(tag);
        free(queue);
    }
};

static void Fail(FlattenedCutoutScores* scores, const char* reason) {
    if (scores && !scores->reason) {
        scores->reason = reason;
    }
}

static bool MeasureCutout(const u8* rgb, int w, int h, int stride, FlattenedCutoutScores* scores, Measure* m) {
    if (scores) {
        *scores = FlattenedCutoutScores{};
    }
    if (!rgb || w < kMinSide || h < kMinSide || stride < w * 3) {
        Fail(scores, "small");
        return false;
    }
    i64 nPix = (i64)w * h;
    if (nPix > 40000000) {
        Fail(scores, "large");
        return false;
    }

    i64 pureN = 0;
    i64 nearN = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Pix p = At(rgb, stride, x, y);
            int chroma = Chroma(p);
            int mn = MinC(p);
            if (mn >= 250 && chroma <= 2) {
                pureN++;
            }
            if (IsPaper(p)) {
                nearN++;
            }
        }
    }
    if (scores) {
        scores->pureWhite = (float)pureN / (float)nPix;
        scores->nearWhite = (float)nearN / (float)nPix;
    }

    auto borderPaper = [&](int x, int y) -> bool { return IsPaper(At(rgb, stride, x, y)); };

    int topN = 0, botN = 0, leftN = 0, rightN = 0;
    int borderN = 0;
    int borderPaperN = 0;
    i64 lumSum = 0;
    for (int x = 0; x < w; x++) {
        borderN += 2;
        bool t = borderPaper(x, 0);
        bool b = borderPaper(x, h - 1);
        topN += t ? 1 : 0;
        botN += b ? 1 : 0;
        borderPaperN += (t ? 1 : 0) + (b ? 1 : 0);
        lumSum += Lum(At(rgb, stride, x, 0));
        lumSum += Lum(At(rgb, stride, x, h - 1));
    }
    for (int y = 1; y < h - 1; y++) {
        borderN += 2;
        bool l = borderPaper(0, y);
        bool r = borderPaper(w - 1, y);
        leftN += l ? 1 : 0;
        rightN += r ? 1 : 0;
        borderPaperN += (l ? 1 : 0) + (r ? 1 : 0);
        lumSum += Lum(At(rgb, stride, 0, y));
        lumSum += Lum(At(rgb, stride, w - 1, y));
    }
    float borderWhite = borderN > 0 ? (float)borderPaperN / (float)borderN : 0;
    float top = w > 0 ? (float)topN / (float)w : 0;
    float bot = w > 0 ? (float)botN / (float)w : 0;
    float left = (h > 2) ? (float)leftN / (float)(h - 2) : 0;
    float right = (h > 2) ? (float)rightN / (float)(h - 2) : 0;
    int meanLum = borderN > 0 ? (int)(lumSum / borderN) : 0;
    i64 madSum = 0;
    for (int x = 0; x < w; x++) {
        madSum += abs(Lum(At(rgb, stride, x, 0)) - meanLum);
        madSum += abs(Lum(At(rgb, stride, x, h - 1)) - meanLum);
    }
    for (int y = 1; y < h - 1; y++) {
        madSum += abs(Lum(At(rgb, stride, 0, y)) - meanLum);
        madSum += abs(Lum(At(rgb, stride, w - 1, y)) - meanLum);
    }
    float borderMad = borderN > 0 ? (float)madSum / (float)borderN : 99.f;
    if (scores) {
        scores->borderWhite = borderWhite;
        scores->edgeContact = 1.f - borderWhite;
        scores->borderStd = borderMad;
    }
    if (borderWhite < kBorderMin || top < kSideMin || bot < kSideMin || left < kSideMin || right < kSideMin) {
        Fail(scores, "border");
        return false;
    }
    if (borderMad > kBorderMadMax) {
        Fail(scores, "std");
        return false;
    }

    m->w = w;
    m->h = h;
    m->tag = (u8*)calloc((size_t)nPix, 1);
    m->queue = (int*)malloc((size_t)nPix * sizeof(int));
    if (!m->tag || !m->queue) {
        Fail(scores, "oom");
        return false;
    }

    int qh = 0;
    int qt = 0;
    auto pushPaper = [&](int x, int y) {
        i64 i = (i64)y * w + x;
        if (m->tag[i]) {
            return;
        }
        if (!IsPaper(At(rgb, stride, x, y))) {
            return;
        }
        m->tag[i] = 1;
        m->queue[qt++] = (int)i;
    };
    for (int x = 0; x < w; x++) {
        pushPaper(x, 0);
        pushPaper(x, h - 1);
    }
    for (int y = 1; y < h - 1; y++) {
        pushPaper(0, y);
        pushPaper(w - 1, y);
    }
    while (qh < qt) {
        int i = m->queue[qh++];
        int x = i % w;
        int y = i / w;
        if (x > 0) {
            pushPaper(x - 1, y);
        }
        if (x + 1 < w) {
            pushPaper(x + 1, y);
        }
        if (y > 0) {
            pushPaper(x, y - 1);
        }
        if (y + 1 < h) {
            pushPaper(x, y + 1);
        }
    }

    i64 canvasN = qt;
    i64 inkN = 0;
    i64 grayN = 0;
    i64 lightN = 0;
    i64 edgeInkN = 0;
    int minX = w;
    int minY = h;
    int maxX = -1;
    int maxY = -1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (m->tag[(i64)y * w + x]) {
                continue;
            }
            Pix p = At(rgb, stride, x, y);
            if (IsPaper(p)) {
                continue;
            }
            inkN++;
            int chroma = Chroma(p);
            int lum = Lum(p);
            if (chroma <= kGrayChroma) {
                grayN++;
            }
            if (chroma <= kGrayChroma && lum >= kLightGrayLo && lum <= kLightGrayHi) {
                lightN++;
            }
            if (x < kEdgeBand || y < kEdgeBand || x >= w - kEdgeBand || y >= h - kEdgeBand) {
                edgeInkN++;
            }
            if (x < minX) {
                minX = x;
            }
            if (y < minY) {
                minY = y;
            }
            if (x > maxX) {
                maxX = x;
            }
            if (y > maxY) {
                maxY = y;
            }
        }
    }
    float canvas = (float)canvasN / (float)nPix;
    float ink = (float)inkN / (float)nPix;
    if (scores) {
        scores->canvas = canvas;
        scores->foreground = ink;
    }
    if (canvas < kCanvasMin || canvas > kCanvasMax || ink < kInkMin || ink > kInkMax || inkN <= 0) {
        Fail(scores, "canvas");
        return false;
    }
    i64 edgeN = nPix;
    if (w > kEdgeBand * 2 && h > kEdgeBand * 2) {
        edgeN = nPix - (i64)(w - kEdgeBand * 2) * (h - kEdgeBand * 2);
    }
    float edgeFrac = edgeN > 0 ? (float)edgeInkN / (float)edgeN : 1.f;
    if (scores) {
        scores->edgeContact = edgeFrac;
    }
    // A few JPEG specks may touch the frame. A subject that actually sits on
    // the edge puts a much larger share of the rim into ink.
    if (edgeFrac > kEdgeInkMax) {
        Fail(scores, "inset");
        return false;
    }

    i64 boxW = (i64)maxX - minX + 1;
    i64 boxH = (i64)maxY - minY + 1;
    float fill = (float)inkN / (float)(boxW * boxH);
    if (scores) {
        scores->fillRatio = fill;
    }
    bool cornerInk = !IsPaper(At(rgb, stride, minX, minY)) && !IsPaper(At(rgb, stride, maxX, minY)) &&
                     !IsPaper(At(rgb, stride, minX, maxY)) && !IsPaper(At(rgb, stride, maxX, maxY));
    if (fill >= kPlateFill && cornerInk) {
        Fail(scores, "plate");
        return false;
    }

    i64 contourN = 0;
    i64 sharpN = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!m->tag[(i64)y * w + x]) {
                continue;
            }
            Pix paper = At(rgb, stride, x, y);
            int paperLum = Lum(paper);
            const int nx[4] = {x - 1, x + 1, x, x};
            const int ny[4] = {y, y, y - 1, y + 1};
            for (int k = 0; k < 4; k++) {
                int xx = nx[k];
                int yy = ny[k];
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                    continue;
                }
                if (m->tag[(i64)yy * w + xx]) {
                    continue;
                }
                Pix nb = At(rgb, stride, xx, yy);
                if (IsPaper(nb)) {
                    continue;
                }
                contourN++;
                int drop = paperLum - Lum(nb);
                if (drop >= kSharpDrop || Chroma(nb) >= kSharpChroma) {
                    sharpN++;
                }
            }
        }
    }
    float sharp = contourN > 0 ? (float)sharpN / (float)contourN : 0;
    float gray = inkN > 0 ? (float)grayN / (float)inkN : 0;
    float light = inkN > 0 ? (float)lightN / (float)inkN : 0;
    float busy = inkN > 0 ? (float)contourN / (float)inkN : 99.f;
    if (scores) {
        scores->contourSharp = sharp;
        scores->grayInk = gray;
    }
    bool graphic = contourN >= 16 && sharp >= kSharpMin && light < kLightGrayFrac &&
                   !(gray >= kGrayInkEngrave && sharp < kSharpForGray) && busy <= kBusy;
    // Soft fur still counts when the white frame is clean and the subject is one solid shape.
    bool photo = borderWhite >= kPhotoBorder && borderMad <= kPhotoMad && edgeFrac <= kEdgeInkMax &&
                 busy <= kPhotoBusy && light < kPhotoLight && contourN >= 16;
    if (!graphic && !photo) {
        if (light >= kLightGrayFrac) {
            Fail(scores, "fade");
        } else if (gray >= kGrayInkEngrave && sharp < kSharpForGray) {
            Fail(scores, "engrave");
        } else if (busy > kBusy) {
            Fail(scores, "busy");
        } else {
            Fail(scores, "soft");
        }
        return false;
    }
    if (scores) {
        scores->reason = graphic ? "cutout" : "photo";
    }
    m->accept = true;
    return true;
}

bool ClassifyFlattenedCutout(const u8* rgb, int w, int h, int stride, FlattenedCutoutScores* scores) {
    Measure m;
    return MeasureCutout(rgb, w, h, stride, scores, &m);
}

static int Unmatte(int c, int a) {
    if (a <= 0 || a >= 255) {
        return c;
    }
    int f = (c - 255) * 255 / a + 255;
    if (f < 0) {
        f = 0;
    }
    if (f > 255) {
        f = 255;
    }
    return f;
}

static bool IsFringe(Pix p) {
    // A pixel just outside the canvas that is still mostly white, including a
    // colored mix (red on white becomes pink). Dark subject pixels stay opaque.
    if (IsPaper(p)) {
        return false;
    }
    return Lum(p) >= kFringeLum && MinC(p) >= 140;
}

bool TryFlattenedCutout(u8* rgb, int w, int h, int stride, u8* alpha, FlattenedCutoutScores* scores) {
    Measure m;
    if (!MeasureCutout(rgb, w, h, stride, scores, &m) || !alpha) {
        return false;
    }
    i64 nPix = (i64)w * h;
    for (i64 i = 0; i < nPix; i++) {
        alpha[i] = m.tag[i] ? 0 : 255;
    }

    // tag values: 1 = canvas, 2..4 = fringe distance. Dilate only through light fringe.
    for (int dist = 1; dist <= kFringeDist; dist++) {
        u8 prev = (u8)dist;
        u8 next = (u8)(dist + 1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                i64 i = (i64)y * w + x;
                if (m.tag[i] != 0) {
                    continue;
                }
                bool touch = false;
                if (x > 0 && m.tag[i - 1] == prev) {
                    touch = true;
                }
                if (x + 1 < w && m.tag[i + 1] == prev) {
                    touch = true;
                }
                if (y > 0 && m.tag[i - w] == prev) {
                    touch = true;
                }
                if (y + 1 < h && m.tag[i + w] == prev) {
                    touch = true;
                }
                if (!touch) {
                    continue;
                }
                Pix p = At(rgb, stride, x, y);
                if (!IsFringe(p)) {
                    continue;
                }
                m.tag[i] = next;
                int a = 255 - MinC(p);
                if (a < 12) {
                    a = 0;
                }
                if (a > 255) {
                    a = 255;
                }
                alpha[i] = (u8)a;
                if (a > 0 && a < 255) {
                    u8* px = rgb + (i64)y * stride + (i64)x * 3;
                    px[0] = (u8)Unmatte(px[0], a);
                    px[1] = (u8)Unmatte(px[1], a);
                    px[2] = (u8)Unmatte(px[2], a);
                }
            }
        }
    }
    return true;
}

u8* FlattenedCutoutScaledRgb(const u8* rgb, int w, int h, int stride, int maxSide, int* dw, int* dh) {
    if (dw) {
        *dw = w;
    }
    if (dh) {
        *dh = h;
    }
    if (!rgb || w < 1 || h < 1 || maxSide < 8) {
        return nullptr;
    }
    int longSide = w > h ? w : h;
    if (longSide <= maxSide) {
        return nullptr;
    }
    int outW = w;
    int outH = h;
    if (w >= h) {
        outW = maxSide;
        outH = (int)((i64)h * maxSide / w);
    } else {
        outH = maxSide;
        outW = (int)((i64)w * maxSide / h);
    }
    if (outW < 1) {
        outW = 1;
    }
    if (outH < 1) {
        outH = 1;
    }
    u8* dst = (u8*)malloc((size_t)outW * outH * 3);
    if (!dst) {
        return nullptr;
    }
    for (int y = 0; y < outH; y++) {
        int y0 = (int)((i64)y * h / outH);
        int y1 = (int)((i64)(y + 1) * h / outH);
        if (y1 <= y0) {
            y1 = y0 + 1;
        }
        for (int x = 0; x < outW; x++) {
            int x0 = (int)((i64)x * w / outW);
            int x1 = (int)((i64)(x + 1) * w / outW);
            if (x1 <= x0) {
                x1 = x0 + 1;
            }
            int rs = 0, gs = 0, bs = 0, n = 0;
            for (int yy = y0; yy < y1 && yy < h; yy++) {
                const u8* row = rgb + (i64)yy * stride;
                for (int xx = x0; xx < x1 && xx < w; xx++) {
                    const u8* p = row + (i64)xx * 3;
                    rs += p[0];
                    gs += p[1];
                    bs += p[2];
                    n++;
                }
            }
            u8* d = dst + ((i64)y * outW + x) * 3;
            if (n < 1) {
                n = 1;
            }
            d[0] = (u8)(rs / n);
            d[1] = (u8)(gs / n);
            d[2] = (u8)(bs / n);
        }
    }
    if (dw) {
        *dw = outW;
    }
    if (dh) {
        *dh = outH;
    }
    return dst;
}
