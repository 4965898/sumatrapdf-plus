/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3

   Postl / Leptonica-style skew finder (differential square-sum of row sums
   after vertical shear). Cherry-picked algorithm; no libleptonica link. */

#include "utils/BaseUtil.h"
#include "DeskewPostl.h"

#include <math.h>

namespace DeskewPostl {

static int Popcount32(u32 v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (int)((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
}

static int OtsuThreshold(const u8* gray, int n) {
    int hist[256];
    memset(hist, 0, sizeof(hist));
    for (int i = 0; i < n; i++) {
        hist[gray[i]]++;
    }
    double sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += (double)i * hist[i];
    }
    double sumB = 0;
    int wB = 0;
    double maxv = 0;
    int best = 128;
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (!wB) {
            continue;
        }
        int wF = n - wB;
        if (!wF) {
            break;
        }
        sumB += (double)t * hist[t];
        double d = sumB / wB - (sum - sumB) / wF;
        double v = (double)wB * wF * d * d;
        if (v > maxv) {
            maxv = v;
            best = t;
        }
    }
    return best;
}

// Pack ink (dark) pixels as 1 bits. wpl = words per line.
static u32* GrayTo1BitInk(const u8* gray, int w, int h, int thresh, bool lightInk, int* outWpl) {
    int wpl = (w + 31) / 32;
    *outWpl = wpl;
    u32* bits = (u32*)calloc((size_t)wpl * h, sizeof(u32));
    if (!bits) {
        return nullptr;
    }
    for (int y = 0; y < h; y++) {
        const u8* row = gray + y * w;
        u32* line = bits + y * wpl;
        for (int x = 0; x < w; x++) {
            if (lightInk ? row[x] > thresh : row[x] < thresh) {
                line[x >> 5] |= (1u << (31 - (x & 31)));
            }
        }
    }
    return bits;
}

static u32* Reduce2x(const u32* src, int w, int h, int wpl, int* outW, int* outH, int* outWpl) {
    int rw = w / 2;
    int rh = h / 2;
    if (rw < 8 || rh < 8) {
        return nullptr;
    }
    int rwpl = (rw + 31) / 32;
    *outW = rw;
    *outH = rh;
    *outWpl = rwpl;
    u32* dst = (u32*)calloc((size_t)rwpl * rh, sizeof(u32));
    if (!dst) {
        return nullptr;
    }
    auto getBit = [](const u32* line, int x) -> int { return (line[x >> 5] >> (31 - (x & 31))) & 1; };
    for (int y = 0; y < rh; y++) {
        const u32* r0 = src + (y * 2) * wpl;
        const u32* r1 = src + (y * 2 + 1) * wpl;
        u32* dline = dst + y * rwpl;
        for (int x = 0; x < rw; x++) {
            int sx = x * 2;
            if (getBit(r0, sx) || getBit(r0, sx + 1) || getBit(r1, sx) || getBit(r1, sx + 1)) {
                dline[x >> 5] |= (1u << (31 - (x & 31)));
            }
        }
    }
    return dst;
}

// Vertical shear about image center. Positive angle_rad shears the same way
// Leptonica pixVShearCorner does for skew search.
static void VShear1Bit(const u32* src, u32* dst, int w, int h, int wpl, float angleRad) {
    memset(dst, 0, (size_t)wpl * h * sizeof(u32));
    float tanA = tanf(angleRad);
    int cx = w / 2;
    for (int x = 0; x < w; x++) {
        int shift = (int)(tanA * (float)(x - cx) + (tanA >= 0 ? 0.5f : -0.5f));
        for (int y = 0; y < h; y++) {
            int sy = y - shift;
            if (sy < 0 || sy >= h) {
                continue;
            }
            if ((src[sy * wpl + (x >> 5)] >> (31 - (x & 31))) & 1) {
                dst[y * wpl + (x >> 5)] |= (1u << (31 - (x & 31)));
            }
        }
    }
}

static void RowSums(const u32* bits, int h, int wpl, int* sums) {
    for (int y = 0; y < h; y++) {
        const u32* line = bits + y * wpl;
        int count = 0;
        for (int i = 0; i < wpl; i++) {
            count += Popcount32(line[i]);
        }
        sums[y] = count;
    }
}

// Leptonica pixFindDifferentialSquareSum: skip margins, sum (row[i]-row[i-1])^2.
static double DiffSquareSum(const int* sums, int h, int w) {
    int skiph = (int)(0.05 * w);
    int skip = skiph;
    if (h / 10 < skip) {
        skip = h / 10;
    }
    int nskip = skip / 2;
    if (nskip < 1) {
        nskip = 1;
    }
    double score = 0;
    for (int i = nskip; i < h - nskip; i++) {
        double d = (double)(sums[i] - sums[i - 1]);
        score += d * d;
    }
    return score;
}

// Vertical shear at 30°–60° slides the text out of the bitmap, so the score
// never peaks. Rotate about the center instead. The small-angle form matches
// VShear (src y ≈ y - tan(deg) * (x - cx)), which is the angle fz_rotate uses.
static double ScoreRotatedLines(const u8* gray, int w, int h, float deg, bool lightInk, int thresh, u8* tmp) {
    float rad = deg * (3.14159265f / 180.f);
    float c = cosf(rad);
    float s = sinf(rad);
    float cx = (w - 1) * 0.5f;
    float cy = (h - 1) * 0.5f;
    u8 paper = lightInk ? 0 : 255;
    memset(tmp, paper, (size_t)w * (size_t)h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            int sx = (int)(c * dx + s * dy + cx + 0.5f);
            int sy = (int)(-s * dx + c * dy + cy + 0.5f);
            if ((unsigned)sx < (unsigned)w && (unsigned)sy < (unsigned)h) {
                tmp[(size_t)y * w + x] = gray[(size_t)sy * w + sx];
            }
        }
    }
    double score = 0;
    int prev = -1;
    int margin = h / 20;
    if (margin < 1) {
        margin = 1;
    }
    for (int y = margin; y < h - margin; y++) {
        const u8* row = tmp + (size_t)y * w;
        int ink = 0;
        for (int x = 0; x < w; x++) {
            if (lightInk ? row[x] > thresh : row[x] < thresh) {
                ink++;
            }
        }
        if (prev >= 0) {
            double d = (double)ink - (double)prev;
            score += d * d;
        }
        prev = ink;
    }
    return score;
}

// Short ink runs are glyph strokes, not rules. Upright hanzi are rich in
// horizontal strokes. A page turned on its side has those strokes vertical.
// Returns horizontal/vertical. Above 1 means the glyphs are already upright.
static float UprightStrokeBias(const u8* gray, int w, int h, bool lightInk, int thresh) {
    int step = (w * (i64)h > 500000) ? 2 : 1;
    int hShort = 0;
    int vShort = 0;
    for (int y = 0; y < h; y += step) {
        int run = 0;
        const u8* row = gray + (size_t)y * w;
        for (int x = 0; x < w; x += step) {
            bool on = lightInk ? row[x] > thresh : row[x] < thresh;
            if (on) {
                run++;
            } else {
                if (run >= 2 && run <= 10) {
                    hShort += run;
                }
                run = 0;
            }
        }
    }
    for (int x = 0; x < w; x += step) {
        int run = 0;
        for (int y = 0; y < h; y += step) {
            u8 px = gray[(size_t)y * w + x];
            bool on = lightInk ? px > thresh : px < thresh;
            if (on) {
                run++;
            } else {
                if (run >= 2 && run <= 10) {
                    vShort += run;
                }
                run = 0;
            }
        }
    }
    if (vShort < 1) {
        return 2.f;
    }
    return (float)hShort / (float)vShort;
}

DeskewPostlResult FindSkew(const unsigned char* gray, int w, int h) {
    DeskewPostlResult r{};
    if (!gray || w < 80 || h < 80) {
        return r;
    }

    int nPix = w * h;
    int thresh = OtsuThreshold(gray, nPix);
    if (thresh < 255) {
        thresh++;
    }
    // A dark-theme render is white type on black. Those light glyphs are the lines.
    i64 sumLum = 0;
    for (int i = 0; i < nPix; i++) {
        sumLum += gray[i];
    }
    bool lightInk = nPix > 0 && sumLum / nPix < 128;

    // Coarse rotation search, ±70°. Shear cannot see a page photographed this far over.
    int dw = w;
    int dh = h;
    const u8* rotSrc = gray;
    u8* rotSmall = nullptr;
    int side = w > h ? w : h;
    if (side > 360) {
        dw = (w * 360) / side;
        dh = (h * 360) / side;
        if (dw < 80) {
            dw = 80;
        }
        if (dh < 80) {
            dh = 80;
        }
        rotSmall = (u8*)malloc((size_t)dw * (size_t)dh);
        if (rotSmall) {
            for (int y = 0; y < dh; y++) {
                int sy = (y * h) / dh;
                u8* dst = rotSmall + (size_t)y * dw;
                const u8* srcRow = gray + (size_t)sy * w;
                for (int x = 0; x < dw; x++) {
                    dst[x] = srcRow[(x * w) / dw];
                }
            }
            rotSrc = rotSmall;
        } else {
            dw = w;
            dh = h;
        }
    }
    u8* rotTmp = (u8*)malloc((size_t)dw * (size_t)dh);
    float rotBest = 0.f;
    double rotBestScore = -1.0;
    double rotScore0 = 0;
    if (rotTmp) {
        for (int i = 0; i <= 36; i++) {
            float deg = -90.f + (float)i * 5.f;
            double score = ScoreRotatedLines(rotSrc, dw, dh, deg, lightInk, thresh, rotTmp);
            if (fabsf(deg) < 0.01f) {
                rotScore0 = score;
            }
            if (score > rotBestScore) {
                rotBestScore = score;
                rotBest = deg;
            }
        }
    }
    if (rotTmp && fabsf(rotBest) > 8.f) {
        float angles[32];
        double scores[32];
        int nAngles = 0;
        double minScore = 1e300;
        double maxScore = 0;
        int maxIdx = 0;
        for (float deg = rotBest - 6.f; deg <= rotBest + 6.f + 0.01f && nAngles < 32; deg += 1.f) {
            angles[nAngles] = deg;
            scores[nAngles] = ScoreRotatedLines(rotSrc, dw, dh, deg, lightInk, thresh, rotTmp);
            if (scores[nAngles] > maxScore) {
                maxScore = scores[nAngles];
                maxIdx = nAngles;
            }
            if (scores[nAngles] < minScore) {
                minScore = scores[nAngles];
            }
            nAngles++;
        }
        float lo = nAngles > 0 && maxIdx > 0 ? angles[maxIdx - 1] : rotBest;
        float hi = nAngles > 0 && maxIdx < nAngles - 1 ? angles[maxIdx + 1] : rotBest;
        float bestAngle = nAngles > 0 ? angles[maxIdx] : rotBest;
        double bestScore = maxScore > 0 ? maxScore : rotBestScore;
        for (int iter = 0; iter < 8 && nAngles >= 3; iter++) {
            float midLo = (lo + bestAngle) * 0.5f;
            float midHi = (bestAngle + hi) * 0.5f;
            double scoreLo = ScoreRotatedLines(rotSrc, dw, dh, midLo, lightInk, thresh, rotTmp);
            double scoreHi = ScoreRotatedLines(rotSrc, dw, dh, midHi, lightInk, thresh, rotTmp);
            if (scoreLo > bestScore) {
                hi = bestAngle;
                bestAngle = midLo;
                bestScore = scoreLo;
            } else if (scoreHi > bestScore) {
                lo = bestAngle;
                bestAngle = midHi;
                bestScore = scoreHi;
            } else {
                lo = midLo;
                hi = midHi;
            }
        }
        // A vertical table header or a vertical rule can outscore the body.
        // If upright horizontal text is still a large share of the peak, the
        // page is not sideways and not 竖排: measure the small skew around 0°.
        bool horizontalBody = rotScore0 > 1.0 && rotScore0 > bestScore * 0.45;
        float applied = bestAngle;
        double confScore = bestScore;
        double confMin = rotScore0;
        if (horizontalBody) {
            float fineBest = 0.f;
            double fineScore = rotScore0;
            double fineMin = rotScore0;
            for (int i = -14; i <= 14; i++) {
                float deg = (float)i * 0.5f;
                double score = (i == 0) ? rotScore0 : ScoreRotatedLines(rotSrc, dw, dh, deg, lightInk, thresh, rotTmp);
                if (score < fineMin) {
                    fineMin = score;
                }
                if (score > fineScore) {
                    fineScore = score;
                    fineBest = deg;
                }
            }
            applied = fineBest;
            confScore = fineScore;
            confMin = fineMin;
            bestScore = fineScore;
        } else if (fabsf(bestAngle) > 45.f) {
            // Upright 竖排: stand the columns up. Sideways glyphs (horizontal
            // strokes turned vertical, and no real horizontal body) take the
            // quarter turn.
            float sign = bestAngle >= 0.f ? 1.f : -1.f;
            float bias = UprightStrokeBias(gray, w, h, lightInk, thresh);
            bool sideways = fabsf(bestAngle) >= 75.f && bias < 0.85f;
            if (sideways) {
                applied = sign * 90.f;
            } else {
                applied = bestAngle - sign * 90.f;
            }
        }
        free(rotTmp);
        free(rotSmall);
        (void)confScore;
        float improvement = (rotScore0 > 1.0) ? (float)((bestScore - rotScore0) / rotScore0) : 0.f;
        r.ok = true;
        r.angleDeg = applied;
        if (horizontalBody && confMin > 1.0) {
            r.confidence = (float)(bestScore / confMin);
        } else {
            r.confidence = (rotScore0 > 1.0) ? (float)(bestScore / rotScore0) : 0.f;
        }
        r.improvement = improvement;
        return r;
    }
    free(rotTmp);
    free(rotSmall);

    int wpl = 0;
    u32* bits = GrayTo1BitInk(gray, w, h, thresh, lightInk, &wpl);
    if (!bits) {
        return r;
    }

    // 4x reduction (two 2x passes) — Leptonica default sweep reduction.
    int rw = 0, rh = 0, rwpl = 0;
    u32* r1 = Reduce2x(bits, w, h, wpl, &rw, &rh, &rwpl);
    free(bits);
    if (!r1) {
        return r;
    }
    int rw2 = 0, rh2 = 0, rwpl2 = 0;
    u32* reduced = Reduce2x(r1, rw, rh, rwpl, &rw2, &rh2, &rwpl2);
    free(r1);
    if (!reduced) {
        return r;
    }

    // Nearly upright pages keep the ±7° shear sweep. Large tilts already returned above.
    constexpr float kDeg2Rad = 3.14159265f / 180.f;
    int* sums = new int[rh2];
    u32* sheared = (u32*)calloc((size_t)rwpl2 * rh2, sizeof(u32));
    if (!sheared) {
        delete[] sums;
        free(reduced);
        return r;
    }

    auto scoreAt = [&](float deg) -> double {
        VShear1Bit(reduced, sheared, rw2, rh2, rwpl2, deg * kDeg2Rad);
        RowSums(sheared, rh2, rwpl2, sums);
        return DiffSquareSum(sums, rh2, rw2);
    };

    float sweepLo = -7.f;
    float sweepHi = 7.f;
    float angles[32];
    double scores[32];
    int nAngles = 0;
    double minScore = 1e300;
    double maxScore = 0;
    int maxIdx = 0;
    double score0 = scoreAt(0.f);
    for (float deg = sweepLo; deg <= sweepHi + 0.01f && nAngles < 32; deg += 1.f) {
        angles[nAngles] = deg;
        scores[nAngles] = (fabsf(deg) < 0.01f) ? score0 : scoreAt(deg);
        if (scores[nAngles] > maxScore) {
            maxScore = scores[nAngles];
            maxIdx = nAngles;
        }
        if (scores[nAngles] < minScore) {
            minScore = scores[nAngles];
        }
        nAngles++;
    }
    if (nAngles < 3) {
        free(sheared);
        delete[] sums;
        free(reduced);
        return r;
    }

    // Binary-search refinement around the coarse peak (Leptonica search stage).
    float lo = maxIdx > 0 ? angles[maxIdx - 1] : angles[0];
    float hi = maxIdx < nAngles - 1 ? angles[maxIdx + 1] : angles[nAngles - 1];
    float bestAngle = angles[maxIdx];
    double bestScore = maxScore;
    for (int iter = 0; iter < 12; iter++) {
        float midLo = (lo + bestAngle) * 0.5f;
        float midHi = (bestAngle + hi) * 0.5f;
        double scoreLo = scoreAt(midLo);
        double scoreHi = scoreAt(midHi);
        if (scoreLo > bestScore) {
            hi = bestAngle;
            bestAngle = midLo;
            bestScore = scoreLo;
        } else if (scoreHi > bestScore) {
            lo = bestAngle;
            bestAngle = midHi;
            bestScore = scoreHi;
        } else {
            lo = midLo;
            hi = midHi;
        }
    }

    free(sheared);
    delete[] sums;
    free(reduced);

    // Confidence is local to the fine window. A ±45° global min would make every page look certain.
    float conf = (minScore > 1.0) ? (float)(bestScore / minScore) : 0.f;
    float improvement = (score0 > 1.0) ? (float)((bestScore - score0) / score0) : 0.f;
    if (improvement < 0.05f && fabsf(bestAngle) < 0.6f) {
        conf *= 0.5f;
    }

    // Shear angle that maximizes Postl score is the page skew in Leptonica's
    // clockwise-positive sense. Fitz page space is y-down; fz_rotate(deg) with
    // that same signed angle straightens the page (negating it rotated the
    // wrong way).
    r.ok = true;
    r.angleDeg = bestAngle;
    r.confidence = conf;
    r.improvement = improvement;
    return r;
}

} // namespace DeskewPostl
