/* Copyright 2026 the SumatraPDF project authors (see AUTHORS file).
   License: GPLv3 */

// BlazeFace + 478-point Face Landmarker.
// Preprocess and postprocess follow yakhyo/mediapipe-face-mesh-onnx
// (models/blazeface.py, models/onnx_model.py). No OpenCV.

#include "utils/BaseUtil.h"
#include "utils/FileUtil.h"
#include "utils/ThreadUtil.h"
#include "utils/Timer.h"
#include "utils/WinUtil.h"

#include "FaceLandmarkEngine.h"
#include "OcrOnnx.h"
#include "PdfDarkMode.h"

#include "utils/Log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <wincodec.h>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include "../ext/onnxruntime/onnxruntime_c_api.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

bool gDebugFaceLandmarks = false;

static const int kDetSize = 128;
static const int kNumAnchors = 896;
static const int kNumKeypoints = 6;
static const int kRegDims = 4 + kNumKeypoints * 2; // 16

struct Affine {
    float a = 1.f;
    float b = 0.f;
    float tx = 0.f;
    float c = 0.f;
    float d = 1.f;
    float ty = 0.f;
};

struct DetRow {
    float score = 0.f;
    float v[kRegDims]{};
};

struct DetWindow {
    float x = 0.f;
    float y = 0.f;
    float w = 0.f;
    float h = 0.f;
};

struct OrtPack {
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* opts = nullptr;
    OrtMemoryInfo* mem = nullptr;
    OrtAllocator* alloc = nullptr;
    OrtSession* det = nullptr;
    OrtSession* mesh = nullptr;
    char* detIn = nullptr;
    char* detReg = nullptr;
    char* detScore = nullptr;
    char* meshIn = nullptr;
    char* meshLm = nullptr;
    char* meshScore = nullptr;
    int meshSize = 256;
    int nLandmarks = kFaceLandmarkCount;
    float anchors[kNumAnchors * 2]{};
};

static OrtPack gOrt;
static bool gInitTried = false;
static char gLastError[512] = "";

static void SetFaceError(const char* msg) {
    str::BufSet(gLastError, dimof(gLastError), msg ? msg : "");
}

static float Sigmoid(float x) {
    if (x > 40.f) {
        return 1.f;
    }
    if (x < -40.f) {
        return 0.f;
    }
    return 1.f / (1.f + expf(-x));
}

static Affine InvertAffine(Affine m) {
    float det = m.a * m.d - m.b * m.c;
    Affine inv;
    if (fabsf(det) < 1e-12f) {
        return inv;
    }
    inv.a = m.d / det;
    inv.b = -m.b / det;
    inv.c = -m.c / det;
    inv.d = m.a / det;
    inv.tx = -(inv.a * m.tx + inv.b * m.ty);
    inv.ty = -(inv.c * m.tx + inv.d * m.ty);
    return inv;
}

static void ApplyAffine(Affine m, float x, float y, float* ox, float* oy) {
    *ox = m.a * x + m.b * y + m.tx;
    *oy = m.c * x + m.d * y + m.ty;
}

// OpenCV getRotationMatrix2D + the crop-center shift used by warp_roi.
static Affine RotationToCrop(float cx, float cy, float side, float angleDeg, int outSize) {
    float scale = (float)outSize / side;
    float rad = angleDeg * 3.14159265f / 180.f;
    float alpha = scale * cosf(rad);
    float beta = scale * sinf(rad);
    Affine m;
    m.a = alpha;
    m.b = beta;
    m.tx = (1.f - alpha) * cx - beta * cy + (float)outSize * 0.5f - cx;
    m.c = -beta;
    m.d = alpha;
    m.ty = beta * cx + (1.f - alpha) * cy + (float)outSize * 0.5f - cy;
    return m;
}

static float Tap(const u8* src, int sw, int sh, int stride, int x, int y, int c) {
    if (x < 0 || y < 0 || x >= sw || y >= sh) {
        return 0.f;
    }
    return (float)src[(size_t)y * (size_t)stride + (size_t)x * 3 + c];
}

static void WarpRgb(const u8* src, int sw, int sh, int stride, Affine dstToSrc, u8* dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        u8* row = dst + (size_t)y * (size_t)dw * 3;
        for (int x = 0; x < dw; x++) {
            float sx, sy;
            ApplyAffine(dstToSrc, (float)x, (float)y, &sx, &sy);
            int x0 = (int)floorf(sx);
            int y0 = (int)floorf(sy);
            float wx = sx - (float)x0;
            float wy = sy - (float)y0;
            for (int c = 0; c < 3; c++) {
                float v = (1.f - wy) * ((1.f - wx) * Tap(src, sw, sh, stride, x0, y0, c) +
                                        wx * Tap(src, sw, sh, stride, x0 + 1, y0, c)) +
                          wy * ((1.f - wx) * Tap(src, sw, sh, stride, x0, y0 + 1, c) +
                                wx * Tap(src, sw, sh, stride, x0 + 1, y0 + 1, c));
                int iv = (int)(v + 0.5f);
                if (iv < 0) {
                    iv = 0;
                }
                if (iv > 255) {
                    iv = 255;
                }
                row[x * 3 + c] = (u8)iv;
            }
        }
    }
}

static void RgbToNchw(const u8* rgb, int w, int h, float scale, float bias, float* out) {
    size_t plane = (size_t)w * (size_t)h;
    for (int y = 0; y < h; y++) {
        const u8* row = rgb + (size_t)y * (size_t)w * 3;
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            for (int c = 0; c < 3; c++) {
                out[c * plane + i] = ((float)row[x * 3 + c] + bias) * scale;
            }
        }
    }
}

static void BuildAnchors(float* anchors) {
    const int strides[] = {8, 16, 16, 16};
    int n = 0;
    int idx = 0;
    while (idx < 4) {
        int last = idx;
        while (last < 4 && strides[last] == strides[idx]) {
            last++;
        }
        int repeats = 2 * (last - idx);
        int cells = kDetSize / strides[idx];
        for (int y = 0; y < cells; y++) {
            for (int x = 0; x < cells; x++) {
                float ax = ((float)x + 0.5f) / (float)cells;
                float ay = ((float)y + 0.5f) / (float)cells;
                for (int r = 0; r < repeats; r++) {
                    anchors[n * 2] = ax;
                    anchors[n * 2 + 1] = ay;
                    n++;
                }
            }
        }
        idx = last;
    }
}

static float BoxIou(const DetRow& a, const DetRow& b) {
    float ax1 = a.v[0] - a.v[2] * 0.5f;
    float ay1 = a.v[1] - a.v[3] * 0.5f;
    float ax2 = a.v[0] + a.v[2] * 0.5f;
    float ay2 = a.v[1] + a.v[3] * 0.5f;
    float bx1 = b.v[0] - b.v[2] * 0.5f;
    float by1 = b.v[1] - b.v[3] * 0.5f;
    float bx2 = b.v[0] + b.v[2] * 0.5f;
    float by2 = b.v[1] + b.v[3] * 0.5f;
    float iw = ax2 < bx2 ? ax2 : bx2;
    iw -= ax1 > bx1 ? ax1 : bx1;
    if (iw < 0.f) {
        iw = 0.f;
    }
    float ih = ay2 < by2 ? ay2 : by2;
    ih -= ay1 > by1 ? ay1 : by1;
    if (ih < 0.f) {
        ih = 0.f;
    }
    float inter = iw * ih;
    float uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    if (uni < 1e-9f) {
        return 0.f;
    }
    return inter / uni;
}

static void WarpRgb(const u8* src, int sw, int sh, int stride, Affine dstToSrc, u8* dst, int dw, int dh);
static void RgbToNchw(const u8* rgb, int w, int h, float scale, float bias, float* out);
static bool RunFloat(OrtSession* session, const char* inName, float* input, const int64_t* inShape, size_t inRank,
                     const char* const* outNames, int nOut, float** outs, size_t* outCounts);

static int TileOrigins(int limit, int tile, int stride, int* out, int cap) {
    if (cap <= 0) {
        return 0;
    }
    if (limit <= tile) {
        out[0] = 0;
        return 1;
    }
    int n = 0;
    for (int p = 0; p + tile <= limit && n < cap; p += stride) {
        out[n++] = p;
    }
    int last = limit - tile;
    if (n == 0 || out[n - 1] != last) {
        if (n < cap) {
            out[n++] = last;
        }
    }
    return n;
}

// One BlazeFace pass over an image window. Boxes are stored in source pixels
// (center, width, height, then the 6 keypoints) so later windows can share one NMS.
static bool AppendBlazeWindow(const u8* rgb, int width, int height, int stride, float winX, float winY, float winW,
                              float winH, DetRow* cand, int* nCand, int cap, double* detMs, u8* canvas, float* detIn) {
    if (!cand || !nCand || *nCand >= cap || winW < 8.f || winH < 8.f) {
        return true;
    }
    float side = winW > winH ? winW : winH;
    float scale = (float)kDetSize / side;
    float padX = ((float)kDetSize - winW * scale) * 0.5f;
    float padY = ((float)kDetSize - winH * scale) * 0.5f;
    Affine letter;
    letter.a = scale;
    letter.d = scale;
    letter.tx = padX - winX * scale;
    letter.ty = padY - winY * scale;
    Affine letterInv = InvertAffine(letter);

    memset(canvas, 0, (size_t)kDetSize * kDetSize * 3);
    WarpRgb(rgb, width, height, stride, letterInv, canvas, kDetSize, kDetSize);
    RgbToNchw(canvas, kDetSize, kDetSize, 1.f / 127.5f, -127.5f, detIn);

    int64_t detShape[4] = {1, 3, kDetSize, kDetSize};
    const char* detOutNames[2] = {gOrt.detReg, gOrt.detScore};
    float* detOuts[2]{};
    size_t detCounts[2]{};
    LARGE_INTEGER tDet = TimeGet();
    bool ran = RunFloat(gOrt.det, gOrt.detIn, detIn, detShape, 4, detOutNames, 2, detOuts, detCounts);
    *detMs += TimeSinceInMs(tDet);
    int regSlot = 0;
    int scoreSlot = 1;
    if (detCounts[1] >= (size_t)kNumAnchors * kRegDims && detCounts[0] < (size_t)kNumAnchors * kRegDims) {
        regSlot = 1;
        scoreSlot = 0;
    }
    if (!ran || detCounts[regSlot] < (size_t)kNumAnchors * kRegDims || detCounts[scoreSlot] < (size_t)kNumAnchors) {
        snprintf(gLastError, sizeof(gLastError), "detector output size %d %d ran=%d", (int)detCounts[0],
                 (int)detCounts[1], ran ? 1 : 0);
        logf("FaceLandmark: %s\n", gLastError);
        free(detOuts[0]);
        free(detOuts[1]);
        return false;
    }
    float* detReg = detOuts[regSlot];
    float* detScore = detOuts[scoreSlot];
    auto toImage = [&](float nx, float ny, float* ox, float* oy) {
        *ox = winX + (nx * (float)kDetSize - padX) / scale;
        *oy = winY + (ny * (float)kDetSize - padY) / scale;
    };
    for (int i = 0; i < kNumAnchors && *nCand < cap; i++) {
        float score = Sigmoid(detScore[i]);
        // 0.75 drops glove / finger false faces. Real faces in the team photo
        // still land above 0.85 once the window actually frames them.
        if (score < 0.75f) {
            continue;
        }
        const float* reg = detReg + (size_t)i * kRegDims;
        float ax = gOrt.anchors[i * 2];
        float ay = gOrt.anchors[i * 2 + 1];
        DetRow row;
        row.score = score;
        float ncx = reg[0] / (float)kDetSize + ax;
        float ncy = reg[1] / (float)kDetSize + ay;
        toImage(ncx, ncy, &row.v[0], &row.v[1]);
        row.v[2] = (reg[2] / (float)kDetSize) * (float)kDetSize / scale;
        row.v[3] = (reg[3] / (float)kDetSize) * (float)kDetSize / scale;
        for (int k = 0; k < kNumKeypoints; k++) {
            float kx = reg[4 + k * 2] / (float)kDetSize + ax;
            float ky = reg[5 + k * 2] / (float)kDetSize + ay;
            toImage(kx, ky, &row.v[4 + k * 2], &row.v[5 + k * 2]);
        }
        cand[(*nCand)++] = row;
    }
    free(detOuts[0]);
    free(detOuts[1]);
    return true;
}

struct DetWorkerArg {
    const u8* rgb = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
    const DetWindow* jobs = nullptr;
    int begin = 0;
    int end = 0;
    DetRow* cand = nullptr;
    int nCand = 0;
    int cap = 0;
    bool ok = true;
};

static DWORD WINAPI DetWorkerMain(LPVOID param) {
    DetWorkerArg* a = (DetWorkerArg*)param;
    a->ok = true;
    a->nCand = 0;
    u8* canvas = (u8*)calloc((size_t)kDetSize * kDetSize * 3, 1);
    float* detIn = (float*)malloc((size_t)kDetSize * kDetSize * 3 * sizeof(float));
    if (!canvas || !detIn) {
        a->ok = false;
        free(canvas);
        free(detIn);
        return 0;
    }
    double ms = 0;
    for (int i = a->begin; i < a->end; i++) {
        const DetWindow& j = a->jobs[i];
        if (!AppendBlazeWindow(a->rgb, a->width, a->height, a->stride, j.x, j.y, j.w, j.h, a->cand, &a->nCand, a->cap,
                               &ms, canvas, detIn)) {
            a->ok = false;
            break;
        }
    }
    free(canvas);
    free(detIn);
    return 0;
}

static int WeightedNms(DetRow* rows, int n, DetRow* out) {
    bool* dead = (bool*)calloc((size_t)n, 1);
    if (!dead) {
        return 0;
    }
    int nOut = 0;
    for (;;) {
        int top = -1;
        for (int i = 0; i < n; i++) {
            if (dead[i]) {
                continue;
            }
            if (top < 0 || rows[i].score > rows[top].score) {
                top = i;
            }
        }
        if (top < 0) {
            break;
        }
        float wsum = 0.f;
        DetRow blended = rows[top];
        for (int k = 0; k < kRegDims; k++) {
            blended.v[k] = 0.f;
        }
        for (int i = 0; i < n; i++) {
            if (dead[i]) {
                continue;
            }
            if (BoxIou(rows[top], rows[i]) <= 0.3f) {
                continue;
            }
            float w = rows[i].score;
            wsum += w;
            for (int k = 0; k < kRegDims; k++) {
                blended.v[k] += rows[i].v[k] * w;
            }
            dead[i] = true;
        }
        blended.score = rows[top].score;
        if (wsum > 0.f) {
            for (int k = 0; k < kRegDims; k++) {
                blended.v[k] /= wsum;
            }
        }
        out[nOut++] = blended;
    }
    free(dead);
    return nOut;
}

static char* DupIoName(OrtSession* session, bool input, size_t index) {
    char* name = nullptr;
    OrtStatus* st = nullptr;
    if (input) {
        st = gOrt.api->SessionGetInputName(session, index, gOrt.alloc, &name);
    } else {
        st = gOrt.api->SessionGetOutputName(session, index, gOrt.alloc, &name);
    }
    if (st) {
        logf("FaceLandmark: SessionGetName: %s\n", gOrt.api->GetErrorMessage(st));
        gOrt.api->ReleaseStatus(st);
        return nullptr;
    }
    char* copy = str::Dup(name);
    gOrt.api->AllocatorFree(gOrt.alloc, name);
    return copy;
}

static bool OutputHasDim(OrtSession* session, size_t index, int64_t want) {
    OrtTypeInfo* ti = nullptr;
    OrtStatus* st = gOrt.api->SessionGetOutputTypeInfo(session, index, &ti);
    if (st) {
        gOrt.api->ReleaseStatus(st);
        return false;
    }
    const OrtTensorTypeAndShapeInfo* info = nullptr;
    st = gOrt.api->CastTypeInfoToTensorInfo(ti, &info);
    if (st || !info) {
        if (st) {
            gOrt.api->ReleaseStatus(st);
        }
        gOrt.api->ReleaseTypeInfo(ti);
        return false;
    }
    size_t nd = 0;
    gOrt.api->GetDimensionsCount(info, &nd);
    int64_t dims[8]{};
    if (nd > 8) {
        nd = 8;
    }
    gOrt.api->GetDimensions(info, dims, nd);
    bool hit = false;
    for (size_t i = 0; i < nd; i++) {
        if (dims[i] == want) {
            hit = true;
        }
    }
    gOrt.api->ReleaseTypeInfo(ti);
    return hit;
}

static OrtSession* LoadSession(const char* path) {
    WCHAR wpath[MAX_PATH * 4]{};
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, dimof(wpath)) <= 0) {
        return nullptr;
    }
    OrtSession* session = nullptr;
    OrtStatus* st = gOrt.api->CreateSession(gOrt.env, wpath, gOrt.opts, &session);
    if (st) {
        logf("FaceLandmark: CreateSession %s: %s\n", path, gOrt.api->GetErrorMessage(st));
        gOrt.api->ReleaseStatus(st);
        return nullptr;
    }
    return session;
}

static bool RunFloat(OrtSession* session, const char* inName, float* input, const int64_t* inShape, size_t inRank,
                     const char* const* outNames, int nOut, float** outs, size_t* outCounts) {
    size_t nElem = 1;
    for (size_t i = 0; i < inRank; i++) {
        nElem *= (size_t)inShape[i];
    }
    OrtValue* inVal = nullptr;
    OrtStatus* st = gOrt.api->CreateTensorWithDataAsOrtValue(gOrt.mem, input, nElem * sizeof(float), inShape, inRank,
                                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inVal);
    if (st) {
        snprintf(gLastError, sizeof(gLastError), "CreateTensor: %s", gOrt.api->GetErrorMessage(st));
        logf("FaceLandmark: %s\n", gLastError);
        gOrt.api->ReleaseStatus(st);
        return false;
    }
    OrtValue* outVals[4]{};
    const char* inNames[] = {inName};
    st = gOrt.api->Run(session, nullptr, inNames, (const OrtValue* const*)&inVal, 1, outNames, (size_t)nOut, outVals);
    gOrt.api->ReleaseValue(inVal);
    if (st) {
        snprintf(gLastError, sizeof(gLastError), "Run: %s", gOrt.api->GetErrorMessage(st));
        logf("FaceLandmark: %s\n", gLastError);
        gOrt.api->ReleaseStatus(st);
        return false;
    }
    bool ok = true;
    for (int i = 0; i < nOut; i++) {
        outs[i] = nullptr;
        outCounts[i] = 0;
        float* data = nullptr;
        st = gOrt.api->GetTensorMutableData(outVals[i], (void**)&data);
        if (st || !data) {
            if (st) {
                gOrt.api->ReleaseStatus(st);
            }
            ok = false;
            break;
        }
        OrtTensorTypeAndShapeInfo* info = nullptr;
        gOrt.api->GetTensorTypeAndShape(outVals[i], &info);
        size_t count = 0;
        gOrt.api->GetTensorShapeElementCount(info, &count);
        gOrt.api->ReleaseTensorTypeAndShapeInfo(info);
        outs[i] = (float*)malloc(count * sizeof(float));
        if (!outs[i]) {
            ok = false;
            break;
        }
        memcpy(outs[i], data, count * sizeof(float));
        outCounts[i] = count;
    }
    for (int i = 0; i < nOut; i++) {
        if (outVals[i]) {
            gOrt.api->ReleaseValue(outVals[i]);
        }
    }
    return ok;
}

// Sidecar is {exe}/ocr when that folder exists (the OCR models). Face weights may
// live there, or in the repo ocr/face directory during development.
static TempStr FaceModelPathTemp(const char* fileName) {
    TempStr side = path::JoinTemp(path::JoinTemp(OcrSidecarDirTemp(), "face"), fileName);
    if (file::Exists(side)) {
        return side;
    }
    TempStr up = path::JoinTemp(GetSelfExeDirTemp(), "..", "..");
    TempStr repoFace = path::JoinTemp(path::NormalizeTemp(path::JoinTemp(up, "ocr")), "face");
    TempStr repo = path::JoinTemp(repoFace, fileName);
    if (file::Exists(repo)) {
        return repo;
    }
    return side;
}

bool FaceLandmarkEngine::Init() {
    if (ready) {
        return true;
    }
    if (gInitTried) {
        ready = gOrt.det && gOrt.mesh && gOrt.detIn && gOrt.detReg && gOrt.detScore && gOrt.meshIn && gOrt.meshLm &&
                gOrt.meshScore;
        return ready;
    }
    gInitTried = true;
    if (!OrtRuntimeAcquire(&gOrt.api, &gOrt.env, &gOrt.opts, &gOrt.mem, &gOrt.alloc)) {
        SetFaceError("ONNX Runtime is not available");
        logf("FaceLandmark: %s\n", gLastError);
        return false;
    }
    TempStr detPath = FaceModelPathTemp("face_detection_short_range.onnx");
    TempStr meshPath = FaceModelPathTemp("face_landmarker_Nx3x256x256.onnx");
    if (!file::Exists(detPath) || !file::Exists(meshPath)) {
        snprintf(gLastError, sizeof(gLastError), "missing models, looked for %s", detPath);
        logf("FaceLandmark: %s\n", gLastError);
        return false;
    }
    gOrt.det = LoadSession(detPath);
    gOrt.mesh = LoadSession(meshPath);
    if (!gOrt.det || !gOrt.mesh) {
        SetFaceError("CreateSession failed");
        return false;
    }
    gOrt.detIn = DupIoName(gOrt.det, true, 0);
    size_t nDetOut = 0;
    gOrt.api->SessionGetOutputCount(gOrt.det, &nDetOut);
    if (nDetOut < 2) {
        snprintf(gLastError, sizeof(gLastError), "detector outputs %d, expected 2", (int)nDetOut);
        logf("FaceLandmark: %s\n", gLastError);
        return false;
    }
    char* detOut0 = DupIoName(gOrt.det, false, 0);
    char* detOut1 = DupIoName(gOrt.det, false, 1);
    if (OutputHasDim(gOrt.det, 0, kRegDims)) {
        gOrt.detReg = detOut0;
        gOrt.detScore = detOut1;
    } else {
        gOrt.detReg = detOut1;
        gOrt.detScore = detOut0;
    }
    gOrt.meshIn = DupIoName(gOrt.mesh, true, 0);
    size_t nMeshOut = 0;
    gOrt.api->SessionGetOutputCount(gOrt.mesh, &nMeshOut);
    if (nMeshOut < 2) {
        snprintf(gLastError, sizeof(gLastError), "landmarker outputs %d, expected 2", (int)nMeshOut);
        logf("FaceLandmark: %s\n", gLastError);
        return false;
    }
    char* meshOut0 = DupIoName(gOrt.mesh, false, 0);
    char* meshOut1 = DupIoName(gOrt.mesh, false, 1);
    if (OutputHasDim(gOrt.mesh, 0, kFaceLandmarkCount)) {
        gOrt.meshLm = meshOut0;
        gOrt.meshScore = meshOut1;
    } else {
        gOrt.meshLm = meshOut1;
        gOrt.meshScore = meshOut0;
    }
    BuildAnchors(gOrt.anchors);
    ready = gOrt.detIn && gOrt.detReg && gOrt.detScore && gOrt.meshIn && gOrt.meshLm && gOrt.meshScore;
    if (!ready) {
        SetFaceError("failed to read model IO names");
        logf("FaceLandmark: %s\n", gLastError);
    }
    return ready;
}

static Mutex gFaceDetectMu;

struct FaceDetectHold {
    FaceDetectHold() { gFaceDetectMu.Lock(); }
    ~FaceDetectHold() { gFaceDetectMu.Unlock(); }
};

bool FaceLandmarkEngine::Detect(const u8* rgb, int width, int height, int stride, Vec<DetectedFace>& faces,
                                FaceLandmarkTiming* timing) {
    FaceDetectHold hold;
    faces.Reset();
    if (timing) {
        *timing = FaceLandmarkTiming{};
    }
    if (!ready && !Init()) {
        return false;
    }
    if (!rgb || width < 8 || height < 8 || stride < width * 3) {
        return false;
    }
    LARGE_INTEGER tAll = TimeGet();

    // The short-range detector sees a 128px letterbox. A face in a group photo
    // shrinks to a few pixels there and disappears. Scan overlapping windows
    // as well, then merge the boxes.
    const int kCandCap = 384;
    // Full frame plus overlapping tiles. Same windows as before; several run at
    // once because each BlazeFace pass is independent and that loop was the
    // slow part of smart invert on a photograph.
    DetWindow jobs[40];
    int nJobs = 0;
    jobs[nJobs].x = 0.f;
    jobs[nJobs].y = 0.f;
    jobs[nJobs].w = (float)width;
    jobs[nJobs].h = (float)height;
    nJobs++;
    if (width > 240 || height > 240) {
        // Keep about half the window overlapping. Stretching the stride on a
        // large page drops faces that sit between tiles.
        int tile = 208;
        int step = tile / 2;
        int xs[32];
        int ys[32];
        int nx = TileOrigins(width, tile, step, xs, 32);
        int ny = TileOrigins(height, tile, step, ys, 32);
        int guard = 0;
        while (nx * ny > 36 && guard < 12) {
            int grow = tile / 4;
            if (grow < 32) {
                grow = 32;
            }
            tile += grow;
            step = tile / 2;
            nx = TileOrigins(width, tile, step, xs, 32);
            ny = TileOrigins(height, tile, step, ys, 32);
            guard++;
        }
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                if (nJobs >= 40) {
                    break;
                }
                jobs[nJobs].x = (float)xs[ix];
                jobs[nJobs].y = (float)ys[iy];
                jobs[nJobs].w = (float)tile;
                jobs[nJobs].h = (float)tile;
                nJobs++;
            }
        }
    }
    LARGE_INTEGER tDetAll = TimeGet();
    int nWorkers = nJobs < 4 ? nJobs : 4;
    if (nWorkers < 1) {
        nWorkers = 1;
    }
    DetWorkerArg workers[4];
    HANDLE threads[4] = {};
    int spawned = 0;
    bool workersOk = true;
    for (int wi = 0; wi < nWorkers; wi++) {
        DetWorkerArg* a = &workers[wi];
        a->rgb = rgb;
        a->width = width;
        a->height = height;
        a->stride = stride;
        a->jobs = jobs;
        a->begin = (nJobs * wi) / nWorkers;
        a->end = (nJobs * (wi + 1)) / nWorkers;
        a->cap = kCandCap;
        a->cand = (DetRow*)malloc(sizeof(DetRow) * kCandCap);
        a->nCand = 0;
        a->ok = false;
        if (!a->cand) {
            workersOk = false;
            break;
        }
        if (wi == nWorkers - 1) {
            DetWorkerMain(a);
        } else {
            threads[spawned] = CreateThread(nullptr, 0, DetWorkerMain, a, 0, nullptr);
            if (!threads[spawned]) {
                DetWorkerMain(a);
            } else {
                spawned++;
            }
        }
    }
    if (spawned > 0) {
        WaitForMultipleObjects(spawned, threads, TRUE, INFINITE);
        for (int i = 0; i < spawned; i++) {
            CloseHandle(threads[i]);
        }
    }
    int nCand = 0;
    for (int wi = 0; wi < nWorkers; wi++) {
        if (!workers[wi].ok) {
            workersOk = false;
        }
        nCand += workers[wi].nCand;
    }
    DetRow* cand = nullptr;
    if (workersOk && nCand > 0) {
        cand = (DetRow*)malloc(sizeof(DetRow) * (size_t)nCand);
        if (!cand) {
            workersOk = false;
        }
    }
    if (cand) {
        int at = 0;
        for (int wi = 0; wi < nWorkers; wi++) {
            if (workers[wi].nCand > 0 && workers[wi].cand) {
                memcpy(cand + at, workers[wi].cand, sizeof(DetRow) * (size_t)workers[wi].nCand);
                at += workers[wi].nCand;
            }
        }
        nCand = at;
    }
    for (int wi = 0; wi < nWorkers; wi++) {
        free(workers[wi].cand);
    }
    if (!workersOk) {
        free(cand);
        return false;
    }
    if (!cand) {
        cand = (DetRow*)malloc(sizeof(DetRow));
        nCand = 0;
        if (!cand) {
            return false;
        }
    }

    DetRow* kept = (DetRow*)malloc(sizeof(DetRow) * (nCand > 0 ? nCand : 1));
    int nKept = 0;
    if (kept && nCand > 0) {
        nKept = WeightedNms(cand, nCand, kept);
    }
    free(cand);

    double meshMs = 0;
    int meshSize = gOrt.meshSize;
    size_t cropFloats = (size_t)meshSize * (size_t)meshSize * 3;
    u8* crop = (u8*)malloc((size_t)meshSize * meshSize * 3);
    float* meshIn = (float*)malloc(cropFloats * sizeof(float));
    if (!crop || !meshIn) {
        free(kept);
        free(crop);
        free(meshIn);
        return false;
    }

    for (int fi = 0; fi < nKept && fi < 16; fi++) {
        float cx = kept[fi].v[0];
        float cy = kept[fi].v[1];
        float hx = kept[fi].v[2] * 0.5f;
        float hy = kept[fi].v[3] * 0.5f;
        float x1 = cx - hx;
        float y1 = cy - hy;
        float x2 = cx + hx;
        float y2 = cy + hy;
        float side = (1.f + 2.f * 0.25f) * ((x2 - x1) > (y2 - y1) ? (x2 - x1) : (y2 - y1));
        float kpx0 = kept[fi].v[4];
        float kpy0 = kept[fi].v[5];
        float kpx1 = kept[fi].v[6];
        float kpy1 = kept[fi].v[7];
        float angle = atan2f(kpy1 - kpy0, kpx1 - kpx0) * 180.f / 3.14159265f;
        float roiCx = (x1 + x2) * 0.5f;
        float roiCy = (y1 + y2) * 0.5f;
        Affine toCrop = RotationToCrop(roiCx, roiCy, side, angle, meshSize);
        Affine toImage = InvertAffine(toCrop);
        WarpRgb(rgb, width, height, stride, toImage, crop, meshSize, meshSize);
        RgbToNchw(crop, meshSize, meshSize, 1.f / 255.f, 0.f, meshIn);

        int64_t meshShape[4] = {1, 3, meshSize, meshSize};
        const char* meshOutNames[2] = {gOrt.meshLm, gOrt.meshScore};
        float* meshOuts[2]{};
        size_t meshCounts[2]{};
        LARGE_INTEGER tMesh = TimeGet();
        bool meshOk = RunFloat(gOrt.mesh, gOrt.meshIn, meshIn, meshShape, 4, meshOutNames, 2, meshOuts, meshCounts);
        meshMs += TimeSinceInMs(tMesh);
        int lmSlot = 0;
        int logitSlot = 1;
        if (meshCounts[1] >= (size_t)kFaceLandmarkCount * 3 && meshCounts[0] < (size_t)kFaceLandmarkCount * 3) {
            lmSlot = 1;
            logitSlot = 0;
        }
        if (!meshOk || meshCounts[lmSlot] < (size_t)kFaceLandmarkCount * 3 || meshCounts[logitSlot] < 1) {
            free(meshOuts[0]);
            free(meshOuts[1]);
            continue;
        }
        float presence = Sigmoid(meshOuts[logitSlot][0]);
        float bw = x2 - x1;
        float bh = y2 - y1;
        float aspect = bh > 1.f ? bw / bh : 0.f;
        // A downward or profile head can score high in the detector while the
        // mesh refuses it. Keep that box and blend an ellipse later.
        bool boxFace = presence < 0.5f && kept[fi].score >= 0.85f && bw >= 28.f && bh >= 28.f && bw <= 360.f &&
                       bh <= 360.f && aspect > 0.55f && aspect < 1.85f;
        if (presence < 0.5f && !boxFace) {
            free(meshOuts[0]);
            free(meshOuts[1]);
            continue;
        }
        DetectedFace face{};
        face.bbox = RectF(x1, y1, bw, bh);
        face.detectScore = kept[fi].score;
        face.presenceScore = presence;
        // Profile and looking-down heads often score presence < 0.5 while the
        // mesh points are still good. Keep them so the paste can cover eyes.
        {
            float zScale = side / (float)meshSize;
            for (int li = 0; li < kFaceLandmarkCount; li++) {
                float lx = meshOuts[lmSlot][li * 3];
                float ly = meshOuts[lmSlot][li * 3 + 1];
                float lz = meshOuts[lmSlot][li * 3 + 2];
                float ix, iy;
                ApplyAffine(toImage, lx, ly, &ix, &iy);
                face.landmarks[li].x = ix;
                face.landmarks[li].y = iy;
                face.landmarks[li].z = lz * zScale;
            }
        }
        faces.Append(face);
        free(meshOuts[0]);
        free(meshOuts[1]);
    }
    free(crop);
    free(meshIn);
    free(kept);

    if (timing) {
        double detMs = TimeSinceInMs(tDetAll);
        timing->blazeFaceMs = detMs;
        timing->landmarkerMs = meshMs;
        timing->totalMs = TimeSinceInMs(tAll);
        timing->nFaces = faces.Size();
    }
    return true;
}

static void PutPx(u8* rgb, int w, int h, int stride, int x, int y, u8 r, u8 g, u8 b) {
    if ((unsigned)x >= (unsigned)w || (unsigned)y >= (unsigned)h) {
        return;
    }
    u8* p = rgb + (size_t)y * (size_t)stride + (size_t)x * 3;
    p[0] = r;
    p[1] = g;
    p[2] = b;
}

static void DrawLine(u8* rgb, int w, int h, int stride, int x0, int y0, int x1, int y1, u8 r, u8 g, u8 b) {
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        PutPx(rgb, w, h, stride, x0, y0, r, g, b);
        PutPx(rgb, w, h, stride, x0 + 1, y0, r, g, b);
        PutPx(rgb, w, h, stride, x0, y0 + 1, r, g, b);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        int e2 = err * 2;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void DrawLoop(u8* rgb, int w, int h, int stride, const DetectedFace& face, const int* idx, int n, u8 r, u8 g,
                     u8 b) {
    for (int i = 0; i < n; i++) {
        const FacePoint& a = face.landmarks[idx[i]];
        const FacePoint& c = face.landmarks[idx[(i + 1) % n]];
        DrawLine(rgb, w, h, stride, (int)(a.x + 0.5f), (int)(a.y + 0.5f), (int)(c.x + 0.5f), (int)(c.y + 0.5f), r, g,
                 b);
    }
}

static void DrawIris(u8* rgb, int w, int h, int stride, const DetectedFace& face, const int* idx) {
    float cx = face.landmarks[idx[0]].x;
    float cy = face.landmarks[idx[0]].y;
    float rad = 0.f;
    for (int i = 1; i < kIrisCount; i++) {
        float dx = face.landmarks[idx[i]].x - cx;
        float dy = face.landmarks[idx[i]].y - cy;
        rad += sqrtf(dx * dx + dy * dy);
    }
    rad = rad / 4.f;
    int ir = (int)(rad + 0.5f);
    if (ir < 1) {
        ir = 1;
    }
    int icx = (int)(cx + 0.5f);
    int icy = (int)(cy + 0.5f);
    for (int a = 0; a < 360; a += 6) {
        float radA = (float)a * 3.14159265f / 180.f;
        int x = icx + (int)(cosf(radA) * (float)ir);
        int y = icy + (int)(sinf(radA) * (float)ir);
        PutPx(rgb, w, h, stride, x, y, 255, 40, 40);
    }
    for (int i = 0; i < kIrisCount; i++) {
        int x = (int)(face.landmarks[idx[i]].x + 0.5f);
        int y = (int)(face.landmarks[idx[i]].y + 0.5f);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                PutPx(rgb, w, h, stride, x + dx, y + dy, 255, 40, 40);
            }
        }
    }
}

static void DrawOverlay(u8* rgb, int w, int h, int stride, const Vec<DetectedFace>& faces, bool fullMesh) {
    for (const DetectedFace& face : faces) {
        if (fullMesh) {
            for (int i = 0; i < 468; i++) {
                PutPx(rgb, w, h, stride, (int)(face.landmarks[i].x + 0.5f), (int)(face.landmarks[i].y + 0.5f), 180, 180,
                      180);
            }
        }
        int x0 = (int)(face.bbox.x + 0.5f);
        int y0 = (int)(face.bbox.y + 0.5f);
        int x1 = (int)(face.bbox.x + face.bbox.dx + 0.5f);
        int y1 = (int)(face.bbox.y + face.bbox.dy + 0.5f);
        DrawLine(rgb, w, h, stride, x0, y0, x1, y0, 255, 220, 0);
        DrawLine(rgb, w, h, stride, x1, y0, x1, y1, 255, 220, 0);
        DrawLine(rgb, w, h, stride, x1, y1, x0, y1, 255, 220, 0);
        DrawLine(rgb, w, h, stride, x0, y1, x0, y0, 255, 220, 0);
        if (face.presenceScore < 0.5f) {
            continue;
        }
        DrawLoop(rgb, w, h, stride, face, kLeftEyeContour, kEyeContourCount, 0, 220, 255);
        DrawLoop(rgb, w, h, stride, face, kRightEyeContour, kEyeContourCount, 0, 220, 255);
        DrawIris(rgb, w, h, stride, face, kLeftIris);
        DrawIris(rgb, w, h, stride, face, kRightIris);
        DrawLoop(rgb, w, h, stride, face, kOuterLipContour, kOuterLipCount, 255, 0, 180);
        DrawLoop(rgb, w, h, stride, face, kInnerLipContour, kInnerLipCount, 40, 220, 80);
    }
}

static bool LoadRgbWic(const WCHAR* path, u8** rgb, int* w, int* h) {
    *rgb = nullptr;
    IWICImagingFactory* factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        return false;
    }
    IWICBitmapDecoder* dec = nullptr;
    hr = factory->CreateDecoderFromFilename(path, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &dec);
    IWICBitmapFrameDecode* frame = nullptr;
    if (SUCCEEDED(hr)) {
        hr = dec->GetFrame(0, &frame);
    }
    IWICFormatConverter* conv = nullptr;
    if (SUCCEEDED(hr)) {
        hr = factory->CreateFormatConverter(&conv);
    }
    if (SUCCEEDED(hr)) {
        hr = conv->Initialize(frame, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0,
                              WICBitmapPaletteTypeCustom);
    }
    UINT ww = 0, hh = 0;
    if (SUCCEEDED(hr)) {
        hr = conv->GetSize(&ww, &hh);
    }
    u8* buf = nullptr;
    if (SUCCEEDED(hr) && ww > 0 && hh > 0) {
        buf = (u8*)malloc((size_t)ww * hh * 3);
        if (!buf) {
            hr = E_OUTOFMEMORY;
        } else {
            hr = conv->CopyPixels(nullptr, ww * 3, ww * hh * 3, buf);
        }
    }
    if (conv) {
        conv->Release();
    }
    if (frame) {
        frame->Release();
    }
    if (dec) {
        dec->Release();
    }
    factory->Release();
    if (FAILED(hr)) {
        free(buf);
        return false;
    }
    *rgb = buf;
    *w = (int)ww;
    *h = (int)hh;
    return true;
}

static bool SavePngWic(const WCHAR* path, const u8* rgb, int w, int h) {
    IWICImagingFactory* factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        return false;
    }
    IWICStream* stream = nullptr;
    hr = factory->CreateStream(&stream);
    if (SUCCEEDED(hr)) {
        hr = stream->InitializeFromFilename(path, GENERIC_WRITE);
    }
    IWICBitmapEncoder* enc = nullptr;
    if (SUCCEEDED(hr)) {
        hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc);
    }
    if (SUCCEEDED(hr)) {
        hr = enc->Initialize(stream, WICBitmapEncoderNoCache);
    }
    IWICBitmapFrameEncode* frame = nullptr;
    if (SUCCEEDED(hr)) {
        hr = enc->CreateNewFrame(&frame, nullptr);
    }
    if (SUCCEEDED(hr)) {
        hr = frame->Initialize(nullptr);
    }
    if (SUCCEEDED(hr)) {
        hr = frame->SetSize((UINT)w, (UINT)h);
    }
    WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppRGB;
    if (SUCCEEDED(hr)) {
        // PNG encode often rewrites this to 24bppBGR. WritePixels then reads B,G,R.
        hr = frame->SetPixelFormat(&fmt);
    }
    u8* swapped = nullptr;
    const u8* src = rgb;
    if (SUCCEEDED(hr) && IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR)) {
        swapped = (u8*)malloc((size_t)w * (size_t)h * 3);
        if (!swapped) {
            hr = E_OUTOFMEMORY;
        } else {
            for (size_t i = 0; i < (size_t)w * (size_t)h; i++) {
                swapped[i * 3] = rgb[i * 3 + 2];
                swapped[i * 3 + 1] = rgb[i * 3 + 1];
                swapped[i * 3 + 2] = rgb[i * 3];
            }
            src = swapped;
        }
    } else if (SUCCEEDED(hr) && !IsEqualGUID(fmt, GUID_WICPixelFormat24bppRGB)) {
        hr = E_FAIL;
    }
    if (SUCCEEDED(hr)) {
        hr = frame->WritePixels((UINT)h, (UINT)(w * 3), (UINT)(w * h * 3), (BYTE*)src);
    }
    free(swapped);
    if (SUCCEEDED(hr)) {
        hr = frame->Commit();
    }
    if (SUCCEEDED(hr)) {
        hr = enc->Commit();
    }
    if (frame) {
        frame->Release();
    }
    if (enc) {
        enc->Release();
    }
    if (stream) {
        stream->Release();
    }
    factory->Release();
    return SUCCEEDED(hr);
}

static void ResizeRgb(const u8* src, int sw, int sh, u8* dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        float fy = ((float)y + 0.5f) * (float)sh / (float)dh - 0.5f;
        int y0 = (int)floorf(fy);
        int y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y0 < 0) {
            y0 = 0;
            wy = 0;
        }
        if (y1 >= sh) {
            y1 = sh - 1;
        }
        for (int x = 0; x < dw; x++) {
            float fx = ((float)x + 0.5f) * (float)sw / (float)dw - 0.5f;
            int x0 = (int)floorf(fx);
            int x1 = x0 + 1;
            float wx = fx - (float)x0;
            if (x0 < 0) {
                x0 = 0;
                wx = 0;
            }
            if (x1 >= sw) {
                x1 = sw - 1;
            }
            for (int c = 0; c < 3; c++) {
                float v = (1.f - wy) * ((1.f - wx) * src[(y0 * sw + x0) * 3 + c] + wx * src[(y0 * sw + x1) * 3 + c]) +
                          wy * ((1.f - wx) * src[(y1 * sw + x0) * 3 + c] + wx * src[(y1 * sw + x1) * 3 + c]);
                dst[(y * dw + x) * 3 + c] = (u8)(v + 0.5f);
            }
        }
    }
}

bool FaceLandmarkDebugTryHandleCommandLine() {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) {
        return false;
    }
    bool mine = false;
    for (int i = 1; i < argc; i++) {
        TempStr arg = ToUtf8Temp(argv[i]);
        if (str::EqI(arg, "--face-landmark-debug") || str::EqI(arg, "--face-tone-preview")) {
            mine = true;
        }
    }
    if (!mine) {
        LocalFree(argv);
        return false;
    }
    const WCHAR* inPath = nullptr;
    const WCHAR* outPath = nullptr;
    bool fullMesh = false;
    bool tonePreview = false;
    for (int i = 1; i < argc; i++) {
        TempStr a = ToUtf8Temp(argv[i]);
        if (str::EqI(a, "--face-landmark-debug")) {
            continue;
        }
        if (str::EqI(a, "--face-tone-preview")) {
            tonePreview = true;
            continue;
        }
        if (str::EqI(a, "--full-mesh")) {
            fullMesh = true;
            continue;
        }
        if (!inPath) {
            inPath = argv[i];
        } else if (!outPath) {
            outPath = argv[i];
        }
    }
    if (!inPath || !outPath) {
        logf("usage: --face-landmark-debug input.png output.png [--full-mesh]\n");
        LocalFree(argv);
        return true;
    }

    HRESULT ole = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    gDebugFaceLandmarks = true;
    u8* rgb = nullptr;
    int w = 0, h = 0;
    bool loaded = LoadRgbWic(inPath, &rgb, &w, &h);
    StrBuilder report;
    report.AppendFmt("input %s\nsize %d x %d\n", ToUtf8Temp(inPath), w, h);
    if (!loaded) {
        report.Append("load failed\n");
    } else if (tonePreview) {
        DarkModePalette palette;
        palette.textR = 0.90f;
        palette.textG = 0.90f;
        palette.textB = 0.88f;
        palette.bgR = 0.12f;
        palette.bgG = 0.12f;
        palette.bgB = 0.12f;
        const WCHAR* tags[] = {L"A_dark", L"B_50", L"C_65", L"D_80", L"E_100", L"F_weighted", L"G_mild"};
        const char* tag8[] = {"A_dark", "B_50", "C_65", "D_80", "E_100", "F_weighted", "G_mild"};
        u8* src = (u8*)malloc((size_t)w * (size_t)h * 3);
        if (!src) {
            report.Append("alloc failed\n");
            free(rgb);
        } else {
            memcpy(src, rgb, (size_t)w * (size_t)h * 3);
            WCHAR stem[MAX_PATH];
            wcsncpy_s(stem, outPath, _TRUNCATE);
            WCHAR* dot = wcsrchr(stem, L'.');
            if (dot) {
                *dot = 0;
            }
            for (int variant = 0; variant < 7; variant++) {
                memcpy(rgb, src, (size_t)w * (size_t)h * 3);
                bool ok = PdfDarkModeToneThemeVariant(rgb, w, h, 3, w * 3, palette, variant);
                WCHAR path[MAX_PATH];
                swprintf_s(path, L"%s_%s.png", stem, tags[variant]);
                if (!ok || !SavePngWic(path, rgb, w, h)) {
                    report.AppendFmt("variant %s failed\n", tag8[variant]);
                } else {
                    report.AppendFmt("wrote %s\n", tag8[variant]);
                }
                if (variant == 5) {
                    SavePngWic(outPath, rgb, w, h);
                }
            }
            free(src);
            free(rgb);
        }
    } else {
        FaceLandmarkEngine engine;
        Vec<DetectedFace> faces;
        FaceLandmarkTiming timing;
        bool ok = engine.Detect(rgb, w, h, w * 3, faces, &timing);
        report.AppendFmt("detect ok=%d faces=%d\n", ok ? 1 : 0, faces.Size());
        if (gLastError[0]) {
            report.AppendFmt("error: %s\n", gLastError);
        }
        report.AppendFmt("BlazeFace inference: %.2f ms\n", timing.blazeFaceMs);
        report.AppendFmt("Landmarker inference: %.2f ms", timing.landmarkerMs);
        if (faces.Size() > 0) {
            report.AppendFmt(" (%.2f ms / face)", timing.landmarkerMs / (double)faces.Size());
        }
        report.AppendFmt("\nTotal: %.2f ms\n", timing.totalMs);
        for (int i = 0; i < faces.Size(); i++) {
            const DetectedFace& f = faces[i];
            report.AppendFmt(
                "face %d detect=%.3f presence=%.3f box=%.1f,%.1f %.1fx%.1f irisL=%.1f,%.1f irisR=%.1f,%.1f\n", i,
                f.detectScore, f.presenceScore, f.bbox.x, f.bbox.y, f.bbox.dx, f.bbox.dy, f.landmarks[kLeftIris[0]].x,
                f.landmarks[kLeftIris[0]].y, f.landmarks[kRightIris[0]].x, f.landmarks[kRightIris[0]].y);
        }
        if (ok) {
            DrawOverlay(rgb, w, h, w * 3, faces, fullMesh);
            if (!SavePngWic(outPath, rgb, w, h)) {
                report.Append("png save failed\n");
            }
        }
        const int benches[] = {512, 1024, 2000};
        for (int bi = 0; bi < 3; bi++) {
            int side = benches[bi];
            float s = (float)side / (float)(w > h ? w : h);
            int dw = (int)((float)w * s + 0.5f);
            int dh = (int)((float)h * s + 0.5f);
            if (dw < 8) {
                dw = 8;
            }
            if (dh < 8) {
                dh = 8;
            }
            u8* scaled = (u8*)malloc((size_t)dw * dh * 3);
            if (!scaled) {
                continue;
            }
            ResizeRgb(rgb, w, h, scaled, dw, dh);
            Vec<DetectedFace> scaledFaces;
            FaceLandmarkTiming st;
            engine.Detect(scaled, dw, dh, dw * 3, scaledFaces, &st);
            report.AppendFmt("bench %dpx (%dx%d) faces=%d blaze=%.2f ms landmarker=%.2f ms total=%.2f ms\n", side, dw,
                             dh, scaledFaces.Size(), st.blazeFaceMs, st.landmarkerMs, st.totalMs);
            free(scaled);
        }
        free(rgb);
    }
    TempStr reportPath = str::JoinTemp(ToUtf8Temp(outPath), ".txt");
    file::WriteFile(reportPath, ByteSlice(report.Get()));
    logf("%s", report.Get());
    if (ole == S_OK) {
        CoUninitialize();
    }
    LocalFree(argv);
    return true;
}
