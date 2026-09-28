/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

// test_util links the dark-mode image code, which calls face detection.
// The real engine needs ONNX Runtime. Unit tests do not.

#include "utils/BaseUtil.h"

#include "FaceLandmarkEngine.h"

bool FaceLandmarkEngine::Detect(const u8*, int, int, int, Vec<DetectedFace>&, FaceLandmarkTiming*) {
    return false;
}
