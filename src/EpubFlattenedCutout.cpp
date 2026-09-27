/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

extern "C" {
#include <mupdf/fitz.h>
typedef fz_image* (*fz_html_recolor_image_fn)(fz_context* ctx, fz_image* image);
void fz_html_set_recolor_image_fn(fz_html_recolor_image_fn fn);
}

#include "utils/BaseUtil.h"

#include "EpubFlattenedCutout.h"
#include "PdfDarkMode.h"
#include "SumatraConfig.h"

static constexpr int kCacheCap = 64;

static thread_local int tl_cutoutDepth = 0;

static CRITICAL_SECTION gCutoutCs;
static INIT_ONCE gCutoutOnce = INIT_ONCE_STATIC_INIT;

struct CutoutCacheEntry {
    unsigned char dig[16];
    fz_image* image = nullptr;
    bool rejected = false;
};

static Vec<CutoutCacheEntry> gCutoutCache;

static BOOL CALLBACK InitCutoutCs(PINIT_ONCE once, PVOID param, PVOID* context) {
    (void)once;
    (void)param;
    (void)context;
    InitializeCriticalSection(&gCutoutCs);
    return TRUE;
}

static void LockCutoutCache() {
    InitOnceExecuteOnce(&gCutoutOnce, InitCutoutCs, nullptr, nullptr);
    EnterCriticalSection(&gCutoutCs);
}

static void UnlockCutoutCache() {
    LeaveCriticalSection(&gCutoutCs);
}

bool EpubFlattenedCutoutWanted(const char* defaultExt) {
    if (!defaultExt || !str::EqI(defaultExt, ".epub")) {
        return false;
    }
    if (!IsDarkThemeSelected()) {
        return false;
    }
    if (GetPdfDocumentColorMode() == PdfDocumentColorMode::Light) {
        return false;
    }
    // Original leaves the picture untouched, including its white paper.
    if (GetPdfImageDarkStrategy() == PdfImageDarkStrategy::Original) {
        return false;
    }
    if (ReflowEbookUsesThemeBitmapRecolor()) {
        return false;
    }
    return true;
}
// Returns a kept image, or null. Sets *rejected when this digest was already refused.
static fz_image* CacheFind(fz_context* ctx, const unsigned char* dig, bool* rejected) {
    *rejected = false;
    LockCutoutCache();
    fz_image* hit = nullptr;
    for (size_t i = 0; i < gCutoutCache.size(); i++) {
        CutoutCacheEntry& e = gCutoutCache[i];
        if (memcmp(e.dig, dig, 16) != 0) {
            continue;
        }
        if (e.rejected || !e.image) {
            *rejected = true;
        } else {
            hit = fz_keep_image(ctx, e.image);
        }
        break;
    }
    UnlockCutoutCache();
    return hit;
}

static void CacheStore(fz_context* ctx, const unsigned char* dig, fz_image* image) {
    LockCutoutCache();
    for (size_t i = 0; i < gCutoutCache.size(); i++) {
        if (memcmp(gCutoutCache[i].dig, dig, 16) == 0) {
            UnlockCutoutCache();
            return;
        }
    }
    if (gCutoutCache.size() >= (size_t)kCacheCap) {
        CutoutCacheEntry& old = gCutoutCache[0];
        if (old.image) {
            fz_drop_image(ctx, old.image);
            old.image = nullptr;
        }
        gCutoutCache.RemoveAt(0);
    }
    CutoutCacheEntry e{};
    memcpy(e.dig, dig, 16);
    e.rejected = image == nullptr;
    e.image = image ? fz_keep_image(ctx, image) : nullptr;
    gCutoutCache.Append(e);
    UnlockCutoutCache();
}

static fz_image* HtmlCutout(fz_context* ctx, fz_image* image) {
    if (tl_cutoutDepth <= 0 || !ctx || !image) {
        return nullptr;
    }
    if (image->imagemask || image->w < 8 || image->h < 8) {
        return nullptr;
    }

    unsigned char dig[16] = {};
    bool haveDig = false;
    if (image->get_digest) {
        fz_try(ctx) {
            image->get_digest(ctx, image, dig);
            haveDig = true;
        }
        fz_catch(ctx) {
            fz_report_error(ctx);
        }
    }
    PdfImageDarkStrategy strategy = GetPdfImageDarkStrategy();
    bool recolor = strategy == PdfImageDarkStrategy::Tone;
    DarkModePalette palette = PdfDarkModeThemePalette();
    // Do not share one cache slot across cutout, tone, and theme-baked automatic processing.
    if (haveDig) {
        if (recolor) {
            dig[0] = (unsigned char)(dig[0] ^ (unsigned char)strategy);
        } else {
            dig[1] ^= (unsigned char)(palette.bgR * 255.f + 0.5f);
            dig[2] ^= (unsigned char)(palette.bgG * 255.f + 0.5f);
            dig[3] ^= (unsigned char)(palette.bgB * 255.f + 0.5f);
            dig[4] ^= (unsigned char)(palette.textR * 255.f + 0.5f);
            dig[5] ^= (unsigned char)(palette.textG * 255.f + 0.5f);
            dig[6] ^= (unsigned char)(palette.textB * 255.f + 0.5f);
            dig[7] ^= 0x5A;
        }
    }
    if (haveDig) {
        bool rejected = false;
        fz_image* hit = CacheFind(ctx, dig, &rejected);
        if (rejected) {
            return nullptr;
        }
        if (hit) {
            return hit;
        }
    }

    fz_image* made = nullptr;
    if (recolor) {
        made = PdfDarkModeRecolorImage(ctx, image, palette);
    } else {
        made = PdfDarkModeAutoProcessImage(ctx, image, palette);
    }
    if (haveDig) {
        CacheStore(ctx, dig, made);
    }
    return made;
}

void EpubFlattenedCutoutPush() {
    if (tl_cutoutDepth == 0) {
        fz_html_set_recolor_image_fn(HtmlCutout);
    }
    tl_cutoutDepth++;
}

void EpubFlattenedCutoutPop() {
    if (tl_cutoutDepth > 0) {
        tl_cutoutDepth--;
    }
}
