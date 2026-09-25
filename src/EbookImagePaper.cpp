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

bool EbookKnockoutPaperBackground(u8* samples, int w, int h, int comps, int stride, EbookPaperColors colors) {
    if (!samples || w < 2 || h < 2 || (comps != 3 && comps != 4) || stride < w * comps) {
        return false;
    }

    i64 paperN = 0;
    i64 darkNeutral = 0;
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
        }
    }
    // A line drawing is mostly paper. A photograph is mostly mid-tones, even
    // when it has a thin white border. Lots of dark neutral pixels means hair
    // and shadows, not a few black strokes — those stay photographic.
    if (opaque < 16 || paperN * 100 < opaque * 15) {
        return false;
    }
    bool recolorNeutralInk = darkNeutral * 100 < opaque * 12;

    int paperR = (int)(sumR / paperN);
    int paperG = (int)(sumG / paperN);
    int paperB = (int)(sumB / paperN);
    if (PixelLuma(paperR, paperG, paperB) < 220) {
        return false;
    }

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
            int ar = ChannelCoverage(paperR, r);
            int ag = ChannelCoverage(paperG, g);
            int ab = ChannelCoverage(paperB, b);
            int coverage = ChannelMax(ar, ag, ab);
            coverage = coverage * srcA / 255;
            int chroma = PixelChroma(r, g, b);
            int outR;
            int outG;
            int outB;
            if (coverage < 18 && chroma < 36) {
                outR = colors.bgR;
                outG = colors.bgG;
                outB = colors.bgB;
            } else if (chroma >= 36 || !recolorNeutralInk) {
                continue;
            } else {
                outR = BlendChannel(colors.textR, colors.bgR, coverage);
                outG = BlendChannel(colors.textG, colors.bgG, coverage);
                outB = BlendChannel(colors.textB, colors.bgB, coverage);
            }
            p[0] = (u8)outR;
            p[1] = (u8)outG;
            p[2] = (u8)outB;
        }
    }
    return true;
}
