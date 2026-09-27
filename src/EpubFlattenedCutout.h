/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

// Dark-theme EPUB only. The html image callback is installed once and does
// nothing unless this thread has pushed. The substitute image is an RGBA
// cutout cached by the source image digest, not by the theme color.

bool EpubFlattenedCutoutWanted(const char* defaultExt);
void EpubFlattenedCutoutPush();
void EpubFlattenedCutoutPop();
