/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

#pragma once

// 478-point face landmarker. Image in, landmarks out.
// Does not know about dark mode, PDF pages, or the render cache.

constexpr int kFaceLandmarkCount = 478;

// yakhyo/mediapipe-face-mesh-onnx: 468-472 left iris, 473-477 right iris.
// Each group is center, right, top, left, bottom.
// MediaPipe's own "left" is the subject's left, which is this right iris.
constexpr int kLeftIris[] = {468, 469, 470, 471, 472};
constexpr int kRightIris[] = {473, 474, 475, 476, 477};
constexpr int kIrisCount = 5;

// Eye aperture loops. kLeftEyeContour is the eye that contains kLeftIris
// (MediaPipe FACEMESH_RIGHT_EYE). kRightEyeContour contains kRightIris.
constexpr int kLeftEyeContour[] = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246};
constexpr int kRightEyeContour[] = {263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466};
constexpr int kEyeContourCount = 16;

// Brow bands above each eye aperture. Same left/right naming as the eye contours.
constexpr int kLeftBrowContour[] = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46};
constexpr int kRightBrowContour[] = {300, 293, 334, 296, 336, 285, 295, 282, 283, 276};
constexpr int kBrowContourCount = 10;

// Outer lip, then the inner-mouth opening. Both are closed loops.
constexpr int kOuterLipContour[] = {61,  185, 40,  39,  37,  0,  267, 269, 270, 409,
                                    291, 375, 321, 405, 314, 17, 84,  181, 91,  146};
constexpr int kOuterLipCount = 20;
constexpr int kInnerLipContour[] = {78,  95,  88,  178, 87,  14, 317, 402, 318, 324,
                                    308, 415, 310, 311, 312, 13, 82,  81,  80,  191};
constexpr int kInnerLipCount = 20;

// MediaPipe FACEMESH_FACE_OVAL, forehead → subject's left → chin → subject's right.
constexpr int kFaceOvalContour[] = {10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109};
constexpr int kFaceOvalCount = 36;

// Nose center line, brow to the tip, then the wings. The face oval does not
// include these, so a profile hull otherwise runs a straight chord across the nose.
constexpr int kNoseSilhouette[] = {168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327};
constexpr int kNoseSilhouetteCount = 12;

struct FacePoint {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
};

struct DetectedFace {
    // Axis-aligned detector box in source pixels (x, y, width, height).
    RectF bbox{};
    float detectScore = 0.f;
    // Face-presence probability after sigmoid. The network emits a logit.
    float presenceScore = 0.f;
    FacePoint landmarks[kFaceLandmarkCount]{};
};

struct FaceLandmarkTiming {
    double blazeFaceMs = 0;
    double landmarkerMs = 0;
    double totalMs = 0;
    int nFaces = 0;
};

// Off by default. The PDF renderer does not read this, so document pixels stay
// as they are. The debug command below turns it on only for its own output file.
extern bool gDebugFaceLandmarks;

class FaceLandmarkEngine {
  public:
    bool Init();
    bool Detect(const u8* rgb, int width, int height, int stride, Vec<DetectedFace>& faces, FaceLandmarkTiming* timing);

  private:
    bool ready = false;
};

// `--face-landmark-debug input.png output.png [--full-mesh]`
// Draws the overlay and exits. Returns true when that flag was present.
bool FaceLandmarkDebugTryHandleCommandLine();
