/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#undef CV_FORCE_SIMD128_CPP  // expected AVX implementation only
#include "opencv2/core/hal/intrin.hpp"
#include "corner.hpp"

namespace cv
{

int calcMinEigenValLine_AVX(const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, int width)
{
    int j = 0;
#if CV_AVX_512F
    __m512 half = _mm512_set1_ps(0.5f);
    for (; j <= width - 16; j += 16)
    {
        __m512 v_a, v_b, v_c, v_t;
        v_a = _mm512_loadu_ps(cov_x2 + j);
        v_b = _mm512_loadu_ps(cov_xy + j);
        v_c = _mm512_loadu_ps(cov_y2 + j);

        v_a = _mm512_mul_ps(v_a, half);
        v_c = _mm512_mul_ps(v_c, half);
        v_t = _mm512_sub_ps(v_a, v_c);
        v_t = _mm512_add_ps(_mm512_mul_ps(v_b, v_b), _mm512_mul_ps(v_t, v_t));
        _mm512_storeu_ps(dst + j, _mm512_sub_ps(_mm512_add_ps(v_a, v_c), _mm512_sqrt_ps(v_t)));
    }
#else
    __m256 half = _mm256_set1_ps(0.5f);
    for (; j <= width - 8; j += 8)
    {
        __m256 v_a, v_b, v_c, v_t;
        v_a = _mm256_loadu_ps(cov_x2 + j);
        v_b = _mm256_loadu_ps(cov_xy + j);
        v_c = _mm256_loadu_ps(cov_y2 + j);

        v_a = _mm256_mul_ps(v_a, half);
        v_c = _mm256_mul_ps(v_c, half);
        v_t = _mm256_sub_ps(v_a, v_c);
        v_t = _mm256_add_ps(_mm256_mul_ps(v_b, v_b), _mm256_mul_ps(v_t, v_t));
        _mm256_storeu_ps(dst + j, _mm256_sub_ps(_mm256_add_ps(v_a, v_c), _mm256_sqrt_ps(v_t)));
    }
#endif
    return j;
}

int calcHarrisLine_AVX(const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k, int width)
{
    int j = 0;
#if CV_AVX_512F
    __m512 v_k = _mm512_set1_ps((float)k);

    for (; j <= width - 16; j += 16)
    {
        __m512 v_a, v_b, v_c;
        v_a = _mm512_loadu_ps(cov_x2 + j);
        v_b = _mm512_loadu_ps(cov_xy + j);
        v_c = _mm512_loadu_ps(cov_y2 + j);

        __m512 v_ac_bb = _mm512_sub_ps(_mm512_mul_ps(v_a, v_c), _mm512_mul_ps(v_b, v_b));
        __m512 v_ac = _mm512_add_ps(v_a, v_c);
        __m512 v_dst = _mm512_sub_ps(v_ac_bb, _mm512_mul_ps(v_k, _mm512_mul_ps(v_ac, v_ac)));
        _mm512_storeu_ps(dst + j, v_dst);
    }
#else
    __m256 v_k = _mm256_set1_ps((float)k);

    for (; j <= width - 8; j += 8)
    {
        __m256 v_a, v_b, v_c;
        v_a = _mm256_loadu_ps(cov_x2);
        v_b = _mm256_loadu_ps(cov_xy);
        v_c = _mm256_loadu_ps(cov_y2);

        __m256 v_ac_bb = _mm256_sub_ps(_mm256_mul_ps(v_a, v_c), _mm256_mul_ps(v_b, v_b));
        __m256 v_ac = _mm256_add_ps(v_a, v_c);
        __m256 v_dst = _mm256_sub_ps(v_ac_bb, _mm256_mul_ps(v_k, _mm256_mul_ps(v_ac, v_ac)));
        _mm256_storeu_ps(dst + j, v_dst);
    }
#endif
    return j;
}

int cornerEigenValsVecsLine_AVX(const float* dxdata, const float* dydata, float* cov_data_x2, float* cov_data_xy, float* cov_data_y2, int width)
{
    int j = 0;
#if CV_AVX_512F
    for (; j <= width - 16; j += 16)
    {
        __m512 v_dx = _mm512_loadu_ps(dxdata + j);
        __m512 v_dy = _mm512_loadu_ps(dydata + j);

        __m512 v_dst;
        v_dst = _mm512_mul_ps(v_dx, v_dx);
        _mm512_storeu_ps(cov_data_x2 + j, v_dst);
        v_dst = _mm512_mul_ps(v_dx, v_dy);
        _mm512_storeu_ps(cov_data_xy + j, v_dst);
        v_dst = _mm512_mul_ps(v_dy, v_dy);
        _mm512_storeu_ps(cov_data_y2 + j, v_dst);
    }
#else
    for (; j <= width - 8; j += 8)
    {
        __m256 v_dx = _mm256_loadu_ps(dxdata + j);
        __m256 v_dy = _mm256_loadu_ps(dydata + j);

        __m256 v_dst;
        v_dst = _mm256_mul_ps(v_dx, v_dx);
        _mm256_storeu_ps(cov_data_x2 + j, v_dst);
        v_dst = _mm256_mul_ps(v_dx, v_dy);
        _mm256_storeu_ps(cov_data_xy + j, v_dst);
        v_dst = _mm256_mul_ps(v_dy, v_dy);
        _mm256_storeu_ps(cov_data_y2 + j, v_dst);
    }
#endif
    return j;
}
    
}
/* End of file */
