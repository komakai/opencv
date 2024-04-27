/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of OpenCV project.
// See opencv/LICENSE for the actual licensing terms.
// Contributed by Giles Payne
//
//M*/

#ifndef OPENCV_CORNER_TEMPLATES_HPP
#define OPENCV_CORNER_TEMPLATES_HPP

namespace cv {

template<class T> T muladd(const T& t1, const T& t2, const T& t3) { return (t1 * t2) + t3; }

template<class T, void(*load_deinterleave)(const float*, T&, T&, T&), void(*store)(float*, const T&), T(*setall)(float), T(*sqrt)(const T&), T(*muladd)(const T&, const T&, const T&) = muladd> void calcMinEigenValLine(const float* cov, float* dst, int width, int lanes, int& jRef) {
    int j = jRef;
    T half = setall(0.5f);
    for (; j <= width - lanes; j += lanes) {
        T a, b, c;
        load_deinterleave(cov + j * 3, a, b, c);
        a = a * half;
        c = c * half;
        T t = a - c;
        t = muladd(b, b , (t * t));
        store(dst + j, (a + c) - sqrt(t));
    }
    jRef = j;
}

template<class T, void(*load_deinterleave)(const float*, T&, T&, T&), void(*store)(float*, const T&), T(*setall)(float)> void calcHarrisLine(const float* cov, float* dst, double k, int width, int lanes, int& jRef) {
    int j = jRef;
    T tk = setall((float)k);
    for (; j <= width - lanes; j += lanes) {
        T a, b, c;
        load_deinterleave(cov + j * 3, a, b, c);
        store(dst + j, a*c - b*b - tk*(a + c)*(a + c));
    }
    jRef = j;
}

template<class T, T(*load)(const float*), void(*store_interleave)(float*, const T&, const T&, const T&)> void cornerEigenValsVecsLine(const float* dxdata, const float* dydata, float* cov_data, int width, int lanes, int& jRef) {
    int j = jRef;
    for (; j <= width - lanes; j += lanes) {
        T dx = load(dxdata + j);
        T dy = load(dydata + j);

        store_interleave(cov_data + j * 3, dx * dx, dx * dy, dy * dy);
    }
    jRef = j;
}

} // namespace

#endif
