/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

#include "utils/BaseUtil.h"
#include "EbookImagePaper.h"

static int ChannelMax(int a, int b, int c) {
    int m = a;
    if (b > m) {
        m = b;
    }
    if (c > m) {
        m = c;
    }
    return m;
}

static int ChannelMin(int a, int b, int c) {
    int m = a;
    if (b < m) {
        m = b;
    }
    if (c < m) {
        m = c;
    }
    return m;
}

static int PixelLuma(int r, int g, int b) {
    return (r * 77 + g * 150 + b * 29) >> 8;
}

static int PixelChroma(int r, int g, int b) {
    return ChannelMax(r, g, b) - ChannelMin(r, g, b);
}

static int ChannelCoverage(int paper, int sample) {
    if (sample >= paper) {
        return 0;
    }
    if (paper < 8) {
        return 255;
    }
    return (paper - sample) * 255 / paper;
}

static int BlendChannel(int ink, int bg, int coverage) {
    return (ink * coverage + bg * (255 - coverage) + 127) / 255;
}

// Near-paper: almost the mat color, and not a saturated fringe.
static bool PixelIsNearPaper(int r, int g, int b, int srcA, int paperR, int paperG, int paperB) {
    if (srcA < 16) {
        return false;
    }
    int ar = ChannelCoverage(paperR, r);
    int ag = ChannelCoverage(paperG, g);
    int ab = ChannelCoverage(paperB, b);
    int coverage = ChannelMax(ar, ag, ab);
    coverage = coverage * srcA / 255;
    return coverage < 18 && PixelChroma(r, g, b) < 36;
}

static void PaintPageBackground(u8* p, EbookPaperColors colors) {
    p[0] = (u8)colors.bgR;
    p[1] = (u8)colors.bgG;
    p[2] = (u8)colors.bgB;
}

struct PaperFlood {
    u8* samples;
    int w;
    int h;
    int comps;
    int stride;
    int paperR;
    int paperG;
    int paperB;
    EbookPaperColors colors;
    u8* seen;
    int* queue;
    int nQueued;
    bool painted;
};

static void FloodConsider(PaperFlood* f, int x, int y) {
    if (x < 0 || y < 0 || x >= f->w || y >= f->h) {
        return;
    }
    int i = y * f->w + x;
    if (f->seen[i]) {
        return;
    }
    f->seen[i] = 1;
    u8* p = f->samples + (i64)y * f->stride + x * f->comps;
    int srcA = f->comps == 4 ? p[3] : 255;
    if (!PixelIsNearPaper(p[0], p[1], p[2], srcA, f->paperR, f->paperG, f->paperB)) {
        return;
    }
    PaintPageBackground(p, f->colors);
    f->queue[f->nQueued++] = i;
    f->painted = true;
}

// Photographs: only the mat that touches the image edge. White fur enclosed by
// darker pixels is not reachable, so it stays.
static bool FloodKnockPaperFromEdges(u8* samples, int w, int h, int comps, int stride, int paperR, int paperG,
                                     int paperB, EbookPaperColors colors) {
    i64 n = (i64)w * (i64)h;
    u8* seen = AllocArray<u8>((size_t)n);
    int* queue = AllocArray<int>((size_t)n);
    if (!seen || !queue) {
        free(seen);
        free(queue);
        return false;
    }
    PaperFlood f{};
    f.samples = samples;
    f.w = w;
    f.h = h;
    f.comps = comps;
    f.stride = stride;
    f.paperR = paperR;
    f.paperG = paperG;
    f.paperB = paperB;
    f.colors = colors;
    f.seen = seen;
    f.queue = queue;

    for (int x = 0; x < w; x++) {
        FloodConsider(&f, x, 0);
        if (h > 1) {
            FloodConsider(&f, x, h - 1);
        }
    }
    for (int y = 1; y < h - 1; y++) {
        FloodConsider(&f, 0, y);
        if (w > 1) {
            FloodConsider(&f, w - 1, y);
        }
    }
    int qh = 0;
    while (qh < f.nQueued) {
        int i = f.queue[qh++];
        int x = i % w;
        int y = i / w;
        FloodConsider(&f, x - 1, y);
        FloodConsider(&f, x + 1, y);
        FloodConsider(&f, x, y - 1);
        FloodConsider(&f, x, y + 1);
    }
    bool painted = f.painted;
    free(seen);
    free(queue);
    return painted;
}

static void RecolorLineArt(u8* samples, int w, int h, int comps, int stride, int paperR, int paperG, int paperB,
                           EbookPaperColors colors) {
    for (int y = 0; y < h; y++) {
        u8* row = samples + (i64)y * stride;
        for (int x = 0; x < w; x++) {
            u8* p = row + x * comps;
            int srcA = comps == 4 ? p[3] : 255;
            if (srcA < 16) {
                continue;
            }
            int r = p[0];
            int g = p[1];
            int b = p[2];
            if (PixelIsNearPaper(r, g, b, srcA, paperR, paperG, paperB)) {
                PaintPageBackground(p, colors);
                continue;
            }
            if (PixelChroma(r, g, b) >= 36) {
                continue;
            }
            int ar = ChannelCoverage(paperR, r);
            int ag = ChannelCoverage(paperG, g);
            int ab = ChannelCoverage(paperB, b);
            int coverage = ChannelMax(ar, ag, ab);
            coverage = coverage * srcA / 255;
            p[0] = (u8)BlendChannel(colors.textR, colors.bgR, coverage);
            p[1] = (u8)BlendChannel(colors.textG, colors.bgG, coverage);
            p[2] = (u8)BlendChannel(colors.textB, colors.bgB, coverage);
        }
    }
}

bool EbookKnockoutPaperBackground(u8* samples, int w, int h, int comps, int stride, EbookPaperColors colors) {
    if (!samples || w < 2 || h < 2 || (comps != 3 && comps != 4) || stride < w * comps) {
        return false;
    }

    i64 paperN = 0;
    i64 darkNeutral = 0;
    i64 colorful = 0;
    i64 opaque = 0;
    i64 sumR = 0;
    i64 sumG = 0;
    i64 sumB = 0;
    for (int y = 0; y < h; y++) {
        u8* row = samples + (i64)y * stride;
        for (int x = 0; x < w; x++) {
            u8* p = row + x * comps;
            int a = comps == 4 ? p[3] : 255;
            if (a < 16) {
                continue;
            }
            opaque++;
            int r = p[0];
            int g = p[1];
            int b = p[2];
            int chroma = PixelChroma(r, g, b);
            int luma = PixelLuma(r, g, b);
            if (chroma <= 32 && luma >= 220) {
                paperN++;
                sumR += r;
                sumG += g;
                sumB += b;
            } else if (chroma < 36 && luma < 90) {
                darkNeutral++;
            }
            if (chroma >= 28) {
                colorful++;
            }
        }
    }
    // A line drawing is mostly paper plus a few strokes. A photograph is mostly
    // mid-tones, even on a white mat. Brown fur is mid-tone, not "dark", so a
    // dark-pixel count would call the kitten line art and punch out its chest.
    if (opaque < 16 || paperN * 100 < opaque * 15) {
        return false;
    }
    i64 mid = opaque - paperN - darkNeutral;
    bool photo = mid * 100 >= opaque * 12;

    int paperR = (int)(sumR / paperN);
    int paperG = (int)(sumG / paperN);
    int paperB = (int)(sumB / paperN);
    if (PixelLuma(paperR, paperG, paperB) < 220) {
        return false;
    }

    // Line art: paper inside a closed stroke is the page too, and black strokes
    // become the text color. A photograph only loses the mat that touches the
    // image edge, so white fur enclosed by the subject stays.
    if (!photo) {
        RecolorLineArt(samples, w, h, comps, stride, paperR, paperG, paperB, colors);
        return true;
    }
    // A gray photograph's light sky is the picture, not a paper mat around a
    // drawing. Color next to a white field (a dancer on a white ground) still
    // floods from the edge.
    if (colorful * 100 < opaque * 6) {
        return false;
    }
    return FloodKnockPaperFromEdges(samples, w, h, comps, stride, paperR, paperG, paperB, colors);
}
