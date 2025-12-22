#pragma once

#if defined(_MSC_VER)
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#include <cmath>
#include <cstdint>

#if defined(__AVX__)
    #define HELIX_USE_AVX 1
#elif defined(__SSE4_1__)
    #define HELIX_USE_SSE4 1
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
    #define HELIX_USE_SSE2 1
#endif

namespace simd {

#if defined(HELIX_USE_AVX)

struct alignas(32) Vec3Pack {
    __m256d data;

    static inline Vec3Pack load(double x, double y, double z) noexcept {
        Vec3Pack v;
        v.data = _mm256_set_pd(0.0, z, y, x);
        return v;
    }

    static inline Vec3Pack load_ptr(const double* ptr) noexcept {
        Vec3Pack v;
        v.data = _mm256_loadu_pd(ptr);
        return v;
    }

    inline void store(double& x, double& y, double& z) const noexcept {
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, data);
        x = tmp[0];
        y = tmp[1];
        z = tmp[2];
    }

    inline Vec3Pack operator+(const Vec3Pack& other) const noexcept {
        Vec3Pack r;
        r.data = _mm256_add_pd(data, other.data);
        return r;
    }

    inline Vec3Pack operator-(const Vec3Pack& other) const noexcept {
        Vec3Pack r;
        r.data = _mm256_sub_pd(data, other.data);
        return r;
    }

    inline Vec3Pack operator*(const Vec3Pack& other) const noexcept {
        Vec3Pack r;
        r.data = _mm256_mul_pd(data, other.data);
        return r;
    }

    inline Vec3Pack operator*(double s) const noexcept {
        Vec3Pack r;
        r.data = _mm256_mul_pd(data, _mm256_set1_pd(s));
        return r;
    }

    inline Vec3Pack operator/(double s) const noexcept {
        Vec3Pack r;
        r.data = _mm256_mul_pd(data, _mm256_set1_pd(1.0 / s));
        return r;
    }

    inline double dot(const Vec3Pack& other) const noexcept {
        __m256d mul = _mm256_mul_pd(data, other.data);
        __m128d low = _mm256_castpd256_pd128(mul);
        __m128d high = _mm256_extractf128_pd(mul, 1);
        __m128d sum = _mm_add_pd(low, high);
        sum = _mm_hadd_pd(sum, sum);
        return _mm_cvtsd_f64(sum);
    }

    inline double normSquared() const noexcept {
        return dot(*this);
    }

    inline double norm() const noexcept {
        return std::sqrt(normSquared());
    }

    inline Vec3Pack cross(const Vec3Pack& other) const noexcept {
        alignas(32) double a[4], b[4];
        _mm256_store_pd(a, data);
        _mm256_store_pd(b, other.data);
        return load(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        );
    }

    inline Vec3Pack normalized() const noexcept {
        double n = norm();
        if (n > 1e-12) {
            return *this / n;
        }
        return load(0, 0, 0);
    }
};

struct alignas(32) QuatPack {
    __m256d data;

    static inline QuatPack load(double w, double x, double y, double z) noexcept {
        QuatPack q;
        q.data = _mm256_set_pd(z, y, x, w);
        return q;
    }

    inline void store(double& w, double& x, double& y, double& z) const noexcept {
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, data);
        w = tmp[0];
        x = tmp[1];
        y = tmp[2];
        z = tmp[3];
    }

    inline QuatPack operator*(const QuatPack& other) const noexcept {
        alignas(32) double a[4], b[4];
        _mm256_store_pd(a, data);
        _mm256_store_pd(b, other.data);

        double rw = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
        double rx = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
        double ry = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
        double rz = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];

        return load(rw, rx, ry, rz);
    }

    inline QuatPack operator+(const QuatPack& other) const noexcept {
        QuatPack r;
        r.data = _mm256_add_pd(data, other.data);
        return r;
    }

    inline QuatPack operator*(double s) const noexcept {
        QuatPack r;
        r.data = _mm256_mul_pd(data, _mm256_set1_pd(s));
        return r;
    }

    inline double normSquared() const noexcept {
        __m256d sq = _mm256_mul_pd(data, data);
        __m128d low = _mm256_castpd256_pd128(sq);
        __m128d high = _mm256_extractf128_pd(sq, 1);
        __m128d sum = _mm_add_pd(low, high);
        sum = _mm_hadd_pd(sum, sum);
        return _mm_cvtsd_f64(sum);
    }

    inline double norm() const noexcept {
        return std::sqrt(normSquared());
    }

    inline QuatPack normalized() const noexcept {
        double n = norm();
        if (n > 1e-12) {
            QuatPack r;
            r.data = _mm256_mul_pd(data, _mm256_set1_pd(1.0 / n));
            return r;
        }
        return load(1, 0, 0, 0);
    }

    inline QuatPack conjugate() const noexcept {
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, data);
        return load(tmp[0], -tmp[1], -tmp[2], -tmp[3]);
    }

    inline Vec3Pack rotate(const Vec3Pack& v) const noexcept {
        alignas(32) double q[4], vec[4];
        _mm256_store_pd(q, data);
        _mm256_store_pd(vec, v.data);

        double qx = q[1], qy = q[2], qz = q[3], qw = q[0];
        double vx = vec[0], vy = vec[1], vz = vec[2];

        double uvx = qy * vz - qz * vy;
        double uvy = qz * vx - qx * vz;
        double uvz = qx * vy - qy * vx;

        double uuvx = qy * uvz - qz * uvy;
        double uuvy = qz * uvx - qx * uvz;
        double uuvz = qx * uvy - qy * uvx;

        return Vec3Pack::load(
            vx + 2.0 * (qw * uvx + uuvx),
            vy + 2.0 * (qw * uvy + uuvy),
            vz + 2.0 * (qw * uvz + uuvz)
        );
    }
};

inline void mat3_mul_vec3(const double m[3][3], const Vec3Pack& v, Vec3Pack& out) noexcept {
    alignas(32) double vec[4];
    _mm256_store_pd(vec, v.data);

    __m256d row0 = _mm256_set_pd(0, m[0][2], m[0][1], m[0][0]);
    __m256d row1 = _mm256_set_pd(0, m[1][2], m[1][1], m[1][0]);
    __m256d row2 = _mm256_set_pd(0, m[2][2], m[2][1], m[2][0]);

    __m256d vv = _mm256_set_pd(0, vec[2], vec[1], vec[0]);

    __m256d mul0 = _mm256_mul_pd(row0, vv);
    __m256d mul1 = _mm256_mul_pd(row1, vv);
    __m256d mul2 = _mm256_mul_pd(row2, vv);

    __m128d low0 = _mm256_castpd256_pd128(mul0);
    __m128d high0 = _mm256_extractf128_pd(mul0, 1);
    __m128d sum0 = _mm_add_pd(low0, high0);
    sum0 = _mm_hadd_pd(sum0, sum0);

    __m128d low1 = _mm256_castpd256_pd128(mul1);
    __m128d high1 = _mm256_extractf128_pd(mul1, 1);
    __m128d sum1 = _mm_add_pd(low1, high1);
    sum1 = _mm_hadd_pd(sum1, sum1);

    __m128d low2 = _mm256_castpd256_pd128(mul2);
    __m128d high2 = _mm256_extractf128_pd(mul2, 1);
    __m128d sum2 = _mm_add_pd(low2, high2);
    sum2 = _mm_hadd_pd(sum2, sum2);

    out = Vec3Pack::load(_mm_cvtsd_f64(sum0), _mm_cvtsd_f64(sum1), _mm_cvtsd_f64(sum2));
}

inline void array_scale_simd(const double* arr, double scale, double* out, size_t n) noexcept {
    __m256d s = _mm256_set1_pd(scale);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(arr + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(v, s));
    }
    for (; i < n; ++i) {
        out[i] = arr[i] * scale;
    }
}

inline void array_add_simd(const double* a, const double* b, double* out, size_t n) noexcept {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void array_fma_simd(const double* a, const double* b, double scale, double* out, size_t n) noexcept {
    __m256d s = _mm256_set1_pd(scale);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
#ifdef __FMA__
        _mm256_storeu_pd(out + i, _mm256_fmadd_pd(vb, s, va));
#else
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, _mm256_mul_pd(vb, s)));
#endif
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i] * scale;
    }
}

inline double array_max_error_simd(const double* y, const double* yerr, size_t n) noexcept {
    __m256d ones = _mm256_set1_pd(1.0);
    __m256d maxVec = _mm256_setzero_pd();
    __m256d signMask = _mm256_set1_pd(-0.0);
    
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d yv = _mm256_loadu_pd(y + i);
        __m256d errv = _mm256_loadu_pd(yerr + i);

        __m256d absY = _mm256_andnot_pd(signMask, yv);
        __m256d scale = _mm256_max_pd(absY, ones);
        
        __m256d absErr = _mm256_andnot_pd(signMask, errv);
        
        __m256d rcpScale = _mm256_div_pd(ones, scale);
        __m256d ratio = _mm256_mul_pd(absErr, rcpScale);
        
        maxVec = _mm256_max_pd(maxVec, ratio);
    }

    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, maxVec);
    double maxErr = tmp[0];
    maxErr = maxErr > tmp[1] ? maxErr : tmp[1];
    maxErr = maxErr > tmp[2] ? maxErr : tmp[2];
    maxErr = maxErr > tmp[3] ? maxErr : tmp[3];

    for (; i < n; ++i) {
        double scale = std::abs(y[i]);
        scale = scale > 1.0 ? scale : 1.0;
        double err = std::abs(yerr[i]) / scale;
        maxErr = maxErr > err ? maxErr : err;
    }

    return maxErr;
}

inline double fast_reciprocal(double x) noexcept {
    __m128d v = _mm_set_sd(x);
    __m128d r = _mm_div_sd(_mm_set_sd(1.0), v);
    return _mm_cvtsd_f64(r);
}

inline void batch_reciprocal_mul(const double* values, double divisor, double* out, size_t n) noexcept {
    double rcp = 1.0 / divisor;
    __m256d rcpVec = _mm256_set1_pd(rcp);
    
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(values + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(v, rcpVec));
    }
    for (; i < n; ++i) {
        out[i] = values[i] * rcp;
    }
}

#else

struct Vec3Pack {
    double x, y, z;

    static inline Vec3Pack load(double x_, double y_, double z_) noexcept {
        return {x_, y_, z_};
    }

    static inline Vec3Pack load_ptr(const double* ptr) noexcept {
        return {ptr[0], ptr[1], ptr[2]};
    }

    inline void store(double& ox, double& oy, double& oz) const noexcept {
        ox = x; oy = y; oz = z;
    }

    inline Vec3Pack operator+(const Vec3Pack& o) const noexcept { return {x+o.x, y+o.y, z+o.z}; }
    inline Vec3Pack operator-(const Vec3Pack& o) const noexcept { return {x-o.x, y-o.y, z-o.z}; }
    inline Vec3Pack operator*(const Vec3Pack& o) const noexcept { return {x*o.x, y*o.y, z*o.z}; }
    inline Vec3Pack operator*(double s) const noexcept { return {x*s, y*s, z*s}; }
    inline Vec3Pack operator/(double s) const noexcept { double r = 1.0/s; return {x*r, y*r, z*r}; }

    inline double dot(const Vec3Pack& o) const noexcept { return x*o.x + y*o.y + z*o.z; }
    inline double normSquared() const noexcept { return x*x + y*y + z*z; }
    inline double norm() const noexcept { return std::sqrt(normSquared()); }

    inline Vec3Pack cross(const Vec3Pack& o) const noexcept {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }

    inline Vec3Pack normalized() const noexcept {
        double n = norm();
        return n > 1e-12 ? *this / n : Vec3Pack{0,0,0};
    }
};

struct QuatPack {
    double w, x, y, z;

    static inline QuatPack load(double w_, double x_, double y_, double z_) noexcept {
        return {w_, x_, y_, z_};
    }

    inline void store(double& ow, double& ox, double& oy, double& oz) const noexcept {
        ow = w; ox = x; oy = y; oz = z;
    }

    inline QuatPack operator*(const QuatPack& q) const noexcept {
        return {
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        };
    }

    inline QuatPack operator+(const QuatPack& q) const noexcept {
        return {w+q.w, x+q.x, y+q.y, z+q.z};
    }

    inline QuatPack operator*(double s) const noexcept {
        return {w*s, x*s, y*s, z*s};
    }

    inline double normSquared() const noexcept { return w*w + x*x + y*y + z*z; }
    inline double norm() const noexcept { return std::sqrt(normSquared()); }

    inline QuatPack normalized() const noexcept {
        double n = norm();
        double r = 1.0 / n;
        return n > 1e-12 ? QuatPack{w*r, x*r, y*r, z*r} : QuatPack{1,0,0,0};
    }

    inline QuatPack conjugate() const noexcept { return {w, -x, -y, -z}; }

    inline Vec3Pack rotate(const Vec3Pack& v) const noexcept {
        double uvx = y*v.z - z*v.y;
        double uvy = z*v.x - x*v.z;
        double uvz = x*v.y - y*v.x;
        double uuvx = y*uvz - z*uvy;
        double uuvy = z*uvx - x*uvz;
        double uuvz = x*uvy - y*uvx;
        return {v.x + 2*(w*uvx + uuvx), v.y + 2*(w*uvy + uuvy), v.z + 2*(w*uvz + uuvz)};
    }
};

inline void mat3_mul_vec3(const double m[3][3], const Vec3Pack& v, Vec3Pack& out) noexcept {
    out.x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z;
    out.y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z;
    out.z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z;
}

inline void array_scale_simd(const double* arr, double scale, double* out, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) out[i] = arr[i] * scale;
}

inline void array_add_simd(const double* a, const double* b, double* out, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

inline void array_fma_simd(const double* a, const double* b, double scale, double* out, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i] * scale;
}

inline double array_max_error_simd(const double* y, const double* yerr, size_t n) noexcept {
    double maxErr = 0;
    for (size_t i = 0; i < n; ++i) {
        double absY = y[i] < 0 ? -y[i] : y[i];
        double scale = absY > 1.0 ? absY : 1.0;
        double absErr = yerr[i] < 0 ? -yerr[i] : yerr[i];
        double err = absErr / scale;
        maxErr = maxErr > err ? maxErr : err;
    }
    return maxErr;
}

inline double fast_reciprocal(double x) noexcept {
    return 1.0 / x;
}

inline void batch_reciprocal_mul(const double* values, double divisor, double* out, size_t n) noexcept {
    double rcp = 1.0 / divisor;
    for (size_t i = 0; i < n; ++i) out[i] = values[i] * rcp;
}

#endif

}
