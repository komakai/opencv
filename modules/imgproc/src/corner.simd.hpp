/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of OpenCV project.
// See opencv/LICENSE for the actual licensing terms.
// Contributed by Giles Payne
//
//M*/

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "corner.templates.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void doCalcMinEigenValLine(const float* cov, float* dst, int width, int& j);

void doCalcHarrisLine(const float* cov, float* dst, double k, int width, int& j);

void doCornerEigenValsVecsLine(const float* dxdata, const float* dydata, float* cov, int width, int& j);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// currently there is no universal intrinsic implementation for AVX - so handle AVX separately to make optimal use of available instructions
#if CV_AVX && !CV_AVX2

struct v_float32x8 final
{
    enum { nlanes = 8 };
    __m256 val;
    explicit v_float32x8(__m256 v) : val(v) {}
    v_float32x8() {}
};

static v_float32x8 operator+(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_add_ps(lhs.val, rhs.val));
}

static v_float32x8 operator-(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_sub_ps(lhs.val, rhs.val));
}

static v_float32x8 operator*(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_mul_ps(lhs.val, rhs.val));
}

static v_float32x8 operator/(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_div_ps(lhs.val, rhs.val));
}

static v_float32x8 operator>(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_cmp_ps(lhs.val, rhs.val, _CMP_GT_OQ));
}

static v_float32x8 operator==(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_cmp_ps(lhs.val, rhs.val, _CMP_EQ_OQ));
}

static v_float32x8 v_load(const float* ptr) {
    return v_float32x8(_mm256_loadu_ps(ptr));
}

static void v_store(float* ptr, const v_float32x8& vec) {
    _mm256_storeu_ps(ptr, vec.val);
}

// load three 8-packed float vector and deinterleave
// probably it's better to write down somewhere else
static void v_load_deinterleave(const float* ptr, v_float32x8& a, v_float32x8& b, v_float32x8& c)
{
    __m256 s0 = _mm256_loadu_ps(ptr);                    // a0, b0, c0, a1, b1, c1, a2, b2,
    __m256 s1 = _mm256_loadu_ps(ptr + 8);                // c2, a3, b3, c3, a4, b4, c4, a5,
    __m256 s2 = _mm256_loadu_ps(ptr + 16);               // b5, c5, a6, b6, c6, a7, b7, c7,
    __m256 s3 = _mm256_permute2f128_ps(s1, s2, 0x21);    // a4, b4, c4, a5, b5, c5, a6, b6,
    __m256 s4 = _mm256_permute2f128_ps(s2, s2, 0x33);    // c6, a7, b7, c7, c6, a7, b7, c7,

    __m256 v00 = _mm256_unpacklo_ps(s0, s3);             // a0, a4, b0, b4, b1, b5, c1, c5,
    __m256 v01 = _mm256_unpackhi_ps(s0, s3);             // c0, c4, a1, a5, a2, a6, b2, b6,
    __m256 v02 = _mm256_unpacklo_ps(s1, s4);             // c2, c6, a3, a7, x,  x,  x,  x,
    __m256 v03 = _mm256_unpackhi_ps(s1, s4);             // b3, b7, c3, c7, x,  x,  x,  x,
    __m256 v04 = _mm256_permute2f128_ps(v02, v03, 0x20); // c2, c6, a3, a7, b3, b7, c3, c7,
    __m256 v05 = _mm256_permute2f128_ps(v01, v03, 0x21); // a2, a6, b2, b6, b3, b7, c3, c7,

    __m256 v10 = _mm256_unpacklo_ps(v00, v05);           // a0, a2, a4, a6, b1, b3, b5, b7,
    __m256 v11 = _mm256_unpackhi_ps(v00, v05);           // b0, b2, b4, b6, c1, c3, c5, c7,
    __m256 v12 = _mm256_unpacklo_ps(v01, v04);           // c0, c2, c4, c6, x,  x,  x,  x,
    __m256 v13 = _mm256_unpackhi_ps(v01, v04);           // a1, a3, a5, a7, x,  x,  x,  x,
    __m256 v14 = _mm256_permute2f128_ps(v11, v12, 0x20); // b0, b2, b4, b6, c0, c2, c4, c6,
    __m256 v15 = _mm256_permute2f128_ps(v10, v11, 0x31); // b1, b3, b5, b7, c1, c3, c5, c7,

    __m256 v20 = _mm256_unpacklo_ps(v14, v15);           // b0, b1, b2, b3, c0, c1, c2, c3,
    __m256 v21 = _mm256_unpackhi_ps(v14, v15);           // b4, b5, b6, b7, c4, c5, c6, c7,
    __m256 v22 = _mm256_unpacklo_ps(v10, v13);           // a0, a1, a2, a3, x,  x,  x,  x,
    __m256 v23 = _mm256_unpackhi_ps(v10, v13);           // a4, a5, a6, a7, x,  x,  x,  x,

    a = v_float32x8(_mm256_permute2f128_ps(v22, v23, 0x20));     // a0, a1, a2, a3, a4, a5, a6, a7,
    b = v_float32x8(_mm256_permute2f128_ps(v20, v21, 0x20));     // b0, b1, b2, b3, b4, b5, b6, b7,
    c = v_float32x8(_mm256_permute2f128_ps(v20, v21, 0x31));     // c0, c1, c2, c3, c4, c5, c6, c7,
}

static void v_store_interleave(float* ptr, const v_float32x8& a, const v_float32x8& b, const v_float32x8& c)
{
    __m256 v00 = _mm256_permute2f128_ps(a.val, b.val, 0x20);      // a0, a1, a2, a3, b0, b1, b2, b3,
    __m256 v01 = _mm256_permute2f128_ps(a.val, b.val, 0x31);      // a4, a5, a6, a7, b4, b5, b6, b7,
    __m256 v02 = _mm256_permute2f128_ps(b.val, c.val, 0x20);      // b0, b1, b2, b3, c0, c1, c2, c3,
    __m256 v03 = _mm256_permute2f128_ps(b.val, c.val, 0x31);      // b4, b5, b6, b7, c4, c5, c6, c7,
    __m256 v04 = _mm256_shuffle_ps(v02, v03, 0x88);       // b0, b2, b4, b6, c0, c2, c4, c6,
    __m256 v05 = _mm256_shuffle_ps(v02, v03, 0xDD);       // b1, b3, b5, b7, c1, c3, c5, c7,
    __m256 v06 = _mm256_shuffle_ps(v00, v01, 0x88);       // a0, a2, a4, a6, b0, b2, b4, b6,
    __m256 v07 = _mm256_shuffle_ps(v00, v01, 0xDD);       // a1, a3, a5, a7, b1, b3, b5, b7,
    __m256 v08 = _mm256_permute2f128_ps(v04, v04, 0x83);  // c0, c2, c4, c6, x,  x,  x,  x,
    __m256 v09 = _mm256_permute2f128_ps(v06, v07, 0x30);  // a0, a2, a4, a6, b1, b3, b5, b7,
    __m256 v10 = _mm256_permute2f128_ps(v04, v05, 0x30);  // b0, b2, b4, b6, c1, c3, c5, c7,
    __m256 v11 = _mm256_shuffle_ps(v09, v10, 0x88);       // a0, a4, b0, b4, b1, b5, c1, c5,
    __m256 v12 = _mm256_shuffle_ps(v09, v10, 0xDD);       // a2, a6, b2, b6, b3, b7, c3, c7,
    __m256 v13 = _mm256_shuffle_ps(v08, v07, 0x88);       // c0, c4, a1, a5, x,  x,  x,  x,
    __m256 v14 = _mm256_shuffle_ps(v08, v07, 0xDD);       // c2, c6, a3, a7, x,  x,  x,  x,
    __m256 v15 = _mm256_permute2f128_ps(v12, v12, 0x83);  // b3, b7, c3, c7, x,  x,  x,  x,
    __m256 v16 = _mm256_permute2f128_ps(v12, v13, 0x02);  // c0, c4, a1, a5, a2, a6, b2, b6,
    __m256 s0 = _mm256_shuffle_ps(v11, v16, 0x88);        // a0, b0, c0, a1, b1, c1, a2, b2,
    __m256 v18 = _mm256_shuffle_ps(v11, v16, 0xDD);       // a4, b4, c4, a5, b5, c5, a6, b6,
    __m256 v19 = _mm256_shuffle_ps(v14, v15, 0x88);       // c2, a3, b3, c3, x,  x,  x,  x,
    __m256 v20 = _mm256_shuffle_ps(v14, v15, 0xDD);       // c6, a7, b7, c7, x,  x,  x,  x,
    __m256 s1 = _mm256_permute2f128_ps(v18, v19, 0x02);   // c2, a3, b3, c3, a4, b4, c4, a5,
    __m256 s2 = _mm256_permute2f128_ps(v18, v20, 0x21);   // b5, c5, a6, b6, c6, a7, b7, c7,

    _mm256_storeu_ps(ptr, s0);
    _mm256_storeu_ps(ptr + 8, s1);
    _mm256_storeu_ps(ptr + 16, s2);
}

static v_float32x8 v_setall(float val) {
    return v_float32x8(_mm256_set1_ps(val));
}

static v_float32x8 v_sqrt(const v_float32x8& vec) {
    return v_float32x8(_mm256_sqrt_ps(vec.val));
}

static v_float32x8 v_muladd(const v_float32x8& v1, const v_float32x8& v2, const v_float32x8& v3) {
    return (v1 * v2) + v3;
}

static v_float32x8 v_abs(const v_float32x8& vec) {
    return v_float32x8(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), vec.val));
}

static v_float32x8 v_select(const v_float32x8& mask, const v_float32x8& vec1, const v_float32x8& vec2) {
    return v_float32x8(_mm256_blendv_ps(vec2.val, vec1.val, mask.val));
}

static void v_zip(const v_float32x8& a0, const v_float32x8& a1, v_float32x8& b0, v_float32x8& b1) {
    b0 = v_float32x8(_mm256_unpacklo_ps(a0.val, a1.val));
    b1 = v_float32x8(_mm256_unpackhi_ps(a0.val, a1.val));
}

void doCalcMinEigenValLine(const float* cov, float* dst, int width, int& j) {
    calcMinEigenValLine<v_float32x8, v_load_deinterleave, v_store, v_setall, v_sqrt>(cov, dst, width, v_float32x8::nlanes, j);
}

void doCalcHarrisLine(const float* cov, float* dst, double k, int width, int& j) {
    calcHarrisLine<v_float32x8, v_load_deinterleave, v_store, v_setall>(cov, dst, k, width, v_float32x8::nlanes, j);
}

void doCornerEigenValsVecsLine(const float* dxdata, const float* dydata, float* cov, int width, int& j) {
    cornerEigenValsVecsLine<v_float32x8, v_load, v_store_interleave>(dxdata, dydata, cov, width, v_float32x8::nlanes, j);
}

#else  // universal intrinsics

static void v_store_interleave3(float* ptr, const v_float32& a, const v_float32& b, const v_float32& c) {
    v_store_interleave(ptr, a, b, c);
}

void doCalcMinEigenValLine(const float* cov, float* dst, int width, int& j) {
    calcMinEigenValLine<v_float32, v_load_deinterleave, v_store, vx_setall_f32, v_sqrt, v_muladd>(cov, dst, width, VTraits<v_float32>::vlanes(), j);
}

void doCalcHarrisLine(const float* cov, float* dst, double k, int width, int& j) {
    calcHarrisLine<v_float32, v_load_deinterleave, v_store, vx_setall_f32>(cov, dst, k, width, VTraits<v_float32>::vlanes(), j);
}

void doCornerEigenValsVecsLine(const float* dxdata, const float* dydata, float* cov, int width, int& j) {
    cornerEigenValsVecsLine<v_float32, vx_load, v_store_interleave3>(dxdata, dydata, cov, width, VTraits<v_float32>::vlanes(), j);
}

#endif

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
