/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

// Dark-theme EPUB/HTML images are drawn as opaque bitmaps, so a white paper
// mat stays a white rectangle on the dark page. This knocks that paper out:
// near-paper pixels become the page background, gray/black ink is redrawn in
// the theme text color, and saturated ink (a purple line, a red fill) stays.
// Photographs without a paper mat are left alone.

struct EbookPaperColors {
    u8 bgR = 0;
    u8 bgG = 0;
    u8 bgB = 0;
    u8 textR = 230;
    u8 textG = 225;
    u8 textB = 216;
};

// samples are RGB or RGBA (alpha last), stride in bytes. Returns true when
// the buffer was recolored in place. A false return leaves samples unchanged.
bool EbookKnockoutPaperBackground(u8* samples, int w, int h, int comps, int stride, EbookPaperColors colors);
