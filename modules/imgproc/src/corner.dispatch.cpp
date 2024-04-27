/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of OpenCV project.
// See opencv/LICENSE for the actual licensing terms.
// Contributed by Giles Payne
//
//M*/

#include "precomp.hpp"

#include <vector>

#include "opencv2/core/hal/intrin.hpp"

#include "corner.simd.hpp"
#include "corner.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

#include "corner.hpp"

namespace cv {

static void doCalcMinEigenValLine(const float* cov, float* dst, int width, int& j)
{
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(doCalcMinEigenValLine, (cov, dst, width, j), CV_CPU_DISPATCH_MODES_ALL);
}

static void doCalcHarrisLine(const float* cov, float* dst, double k, int width, int&j) {
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(doCalcHarrisLine, (cov, dst, k, width, j), CV_CPU_DISPATCH_MODES_ALL);
}

static void doCornerEigenValsVecsLine(const float* dxdata, const float* dydata, float* cov, int width, int& j) {
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(doCornerEigenValsVecsLine, (dxdata, dydata, cov, width, j), CV_CPU_DISPATCH_MODES_ALL);
}

static float float_load(const float* addr) {
    return *addr;
}

static void float_load_deinterleave(const float* addr, float& a, float& b, float& c) {
    a = *addr;
    b = *(addr + 1);
    c = *(addr + 2);
}

static void float_store(float* addr, const float& val) {
    *addr = val;
}

static float float_setall(float val) {
    return val;
}

static float float_sqrt(const float& val) {
    return std::sqrt(val);
}

static void float_store_interleave(float* addr, const float& val1, const float& val2, const float& val3) {
    *addr = val1;
    *(addr + 1) = val2;
    *(addr + 2) = val3;
}

static float float_abs(const float& val) {
    return fabs(val);
}

static float float_select(const float& mask, const float& val1, const float& val2) {
    return mask ? val1 : val2;
}

static void float_zip(const float& a0, const float& a1, float& b0, float& b1) {
    b0 = a0;
    b1 = a1;
}

static void doCalcMinEigenValLine_NOSIMD(const float* cov, float* dst, int width, int& j) {
    calcMinEigenValLine<float, float_load_deinterleave, float_store, float_setall, float_sqrt>(cov, dst, width, 1, j);
}

static void doCalcHarrisLine_NOSIMD(const float* cov, float* dst, double k, int width, int& j) {
    calcHarrisLine<float, float_load_deinterleave, float_store, float_setall>(cov, dst, k, width, 1, j);
}

static void doCornerEigenValsVecsLine_NOSIMD(const float* dxdata, const float* dydata, float* cov, int width, int& j) {
    cornerEigenValsVecsLine<float, float_load, float_store_interleave>(dxdata, dydata, cov, width, 1, j);
}

void calcMinEigenValLine(const float* cov, float* dst, int width) {
    int j = 0;
    doCalcMinEigenValLine(cov, dst, width, j);
    doCalcMinEigenValLine_NOSIMD(cov, dst, width, j);
}

void calcHarrisLine(const float* cov, float* dst, double k, int width) {
    int j = 0;
    doCalcHarrisLine(cov, dst, k, width, j);
    doCalcHarrisLine_NOSIMD(cov, dst, k, width, j);
}

void cornerEigenValsVecsLine(const float* dxdata, const float* dydata, float* cov, int width) {
    int j = 0;
    doCornerEigenValsVecsLine(dxdata, dydata, cov, width, j);
    doCornerEigenValsVecsLine_NOSIMD(dxdata, dydata, cov, width, j);
}

} // namespace
