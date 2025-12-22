#pragma once
#include <array>
#include <cmath>
#include <limits>
#include "SIMDMath.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct alignas(32) Vec3 {
    double x, y, z;
    
    constexpr Vec3() noexcept : x(0), y(0), z(0) {}
    constexpr Vec3(double x_, double y_, double z_) noexcept : x(x_), y(y_), z(z_) {}
    
    inline Vec3 operator+(const Vec3& v) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = a + b;
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    inline Vec3 operator-(const Vec3& v) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = a - b;
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    constexpr Vec3 operator-() const noexcept { return Vec3(-x, -y, -z); }
    
    inline Vec3 operator*(double s) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack r = a * s;
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    inline Vec3 operator*(const Vec3& v) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = a * b;
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    inline Vec3 operator/(double s) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack r = a / s;
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    inline Vec3& operator+=(const Vec3& v) noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = a + b;
        r.store(x, y, z);
        return *this;
    }
    
    inline Vec3& operator-=(const Vec3& v) noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = a - b;
        r.store(x, y, z);
        return *this;
    }
    
    inline Vec3& operator*=(double s) noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack r = a * s;
        r.store(x, y, z);
        return *this;
    }
    
    inline double dot(const Vec3& v) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        return a.dot(b);
    }
    
    inline Vec3 cross(const Vec3& v) const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        simd::Vec3Pack b = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = a.cross(b);
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    inline double normSquared() const noexcept {
        simd::Vec3Pack a = simd::Vec3Pack::load(x, y, z);
        return a.normSquared();
    }
    
    inline double norm() const noexcept {
        return std::sqrt(normSquared());
    }
    
    inline Vec3 normalized() const noexcept { 
        double n = norm();
        return n > 1e-12 ? *this / n : Vec3();
    }
    
    constexpr Vec3 abs() const noexcept { 
        return Vec3(x < 0 ? -x : x, y < 0 ? -y : y, z < 0 ? -z : z); 
    }
    
    constexpr double max() const noexcept { return x > y ? (x > z ? x : z) : (y > z ? y : z); }
    constexpr double min() const noexcept { return x < y ? (x < z ? x : z) : (y < z ? y : z); }
    
    static constexpr Vec3 zero() noexcept { return Vec3(0, 0, 0); }
    static constexpr Vec3 unitX() noexcept { return Vec3(1, 0, 0); }
    static constexpr Vec3 unitY() noexcept { return Vec3(0, 1, 0); }
    static constexpr Vec3 unitZ() noexcept { return Vec3(0, 0, 1); }
};

inline Vec3 operator*(double s, const Vec3& v) noexcept { return v * s; }

struct alignas(32) Quaternion {
    double w, x, y, z;
    
    constexpr Quaternion() noexcept : w(1), x(0), y(0), z(0) {}
    constexpr Quaternion(double w_, double x_, double y_, double z_) noexcept : w(w_), x(x_), y(y_), z(z_) {}
    
    inline Quaternion operator*(const Quaternion& q) const noexcept {
        simd::QuatPack a = simd::QuatPack::load(w, x, y, z);
        simd::QuatPack b = simd::QuatPack::load(q.w, q.x, q.y, q.z);
        simd::QuatPack r = a * b;
        Quaternion result;
        r.store(result.w, result.x, result.y, result.z);
        return result;
    }
    
    inline Quaternion operator+(const Quaternion& q) const noexcept {
        simd::QuatPack a = simd::QuatPack::load(w, x, y, z);
        simd::QuatPack b = simd::QuatPack::load(q.w, q.x, q.y, q.z);
        simd::QuatPack r = a + b;
        Quaternion result;
        r.store(result.w, result.x, result.y, result.z);
        return result;
    }
    
    inline Quaternion operator*(double s) const noexcept {
        simd::QuatPack a = simd::QuatPack::load(w, x, y, z);
        simd::QuatPack r = a * s;
        Quaternion result;
        r.store(result.w, result.x, result.y, result.z);
        return result;
    }
    
    inline double normSquared() const noexcept {
        simd::QuatPack a = simd::QuatPack::load(w, x, y, z);
        return a.normSquared();
    }
    
    inline double norm() const noexcept { return std::sqrt(normSquared()); }
    
    inline Quaternion normalized() const noexcept {
        simd::QuatPack a = simd::QuatPack::load(w, x, y, z);
        simd::QuatPack r = a.normalized();
        Quaternion result;
        r.store(result.w, result.x, result.y, result.z);
        return result;
    }
    
    inline Quaternion conjugate() const noexcept { return Quaternion(w, -x, -y, -z); }
    
    inline Quaternion inverse() const noexcept {
        double n2 = normSquared();
        return n2 > 1e-12 ? Quaternion(w / n2, -x / n2, -y / n2, -z / n2) : Quaternion();
    }
    
    inline Vec3 rotate(const Vec3& v) const noexcept {
        simd::QuatPack q = simd::QuatPack::load(w, x, y, z);
        simd::Vec3Pack vp = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = q.rotate(vp);
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    inline Vec3 inverseRotate(const Vec3& v) const noexcept {
        simd::QuatPack q = simd::QuatPack::load(w, -x, -y, -z);
        simd::Vec3Pack vp = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r = q.rotate(vp);
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    static Quaternion fromAxisAngle(const Vec3& axis, double angle) noexcept {
        double halfAngle = angle * 0.5;
        double s = std::sin(halfAngle);
        Vec3 n = axis.normalized();
        return Quaternion(std::cos(halfAngle), n.x * s, n.y * s, n.z * s);
    }
    
    static Quaternion fromEulerZYX(double roll, double pitch, double yaw) noexcept {
        double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
        double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
        double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }
    
    Vec3 toEulerZYX() const noexcept {
        Vec3 euler;
        double sinr_cosp = 2.0 * (w * x + y * z);
        double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        euler.x = std::atan2(sinr_cosp, cosr_cosp);
        
        double sinp = 2.0 * (w * y - z * x);
        euler.y = std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
        
        double siny_cosp = 2.0 * (w * z + x * y);
        double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        euler.z = std::atan2(siny_cosp, cosy_cosp);
        
        return euler;
    }
    
    static Quaternion slerp(const Quaternion& a, const Quaternion& b, double t) noexcept {
        double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
        Quaternion b2 = dot < 0 ? Quaternion(-b.w, -b.x, -b.y, -b.z) : b;
        dot = dot < 0 ? -dot : dot;
        
        if (dot > 0.9995) {
            return Quaternion(
                a.w + t * (b2.w - a.w),
                a.x + t * (b2.x - a.x),
                a.y + t * (b2.y - a.y),
                a.z + t * (b2.z - a.z)
            ).normalized();
        }
        
        double theta0 = std::acos(dot);
        double theta = theta0 * t;
        double sinTheta = std::sin(theta);
        double sinTheta0 = std::sin(theta0);
        
        double s0 = std::cos(theta) - dot * sinTheta / sinTheta0;
        double s1 = sinTheta / sinTheta0;
        
        return Quaternion(
            s0 * a.w + s1 * b2.w,
            s0 * a.x + s1 * b2.x,
            s0 * a.y + s1 * b2.y,
            s0 * a.z + s1 * b2.z
        );
    }
    
    Quaternion derivative(const Vec3& omega) const noexcept {
        return Quaternion(0, omega.x, omega.y, omega.z) * (*this) * 0.5;
    }
};

struct alignas(64) Mat3 {
    double data[3][3];
    
    constexpr Mat3() noexcept : data{{1,0,0},{0,1,0},{0,0,1}} {}
    
    constexpr Mat3(double d00, double d01, double d02,
                   double d10, double d11, double d12,
                   double d20, double d21, double d22) noexcept
        : data{{d00,d01,d02},{d10,d11,d12},{d20,d21,d22}} {}
    
    static constexpr Mat3 diagonal(double ixx, double iyy, double izz) noexcept {
        return Mat3(ixx, 0, 0, 0, iyy, 0, 0, 0, izz);
    }
    
    static constexpr Mat3 zero() noexcept {
        return Mat3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    
    inline Vec3 operator*(const Vec3& v) const noexcept {
        simd::Vec3Pack vp = simd::Vec3Pack::load(v.x, v.y, v.z);
        simd::Vec3Pack r;
        simd::mat3_mul_vec3(data, vp, r);
        Vec3 result;
        r.store(result.x, result.y, result.z);
        return result;
    }
    
    constexpr Mat3 operator*(const Mat3& m) const noexcept {
        Mat3 r;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                r.data[i][j] = data[i][0] * m.data[0][j] 
                             + data[i][1] * m.data[1][j] 
                             + data[i][2] * m.data[2][j];
            }
        }
        return r;
    }
    
    constexpr Mat3 operator*(double s) const noexcept {
        return Mat3(
            data[0][0]*s, data[0][1]*s, data[0][2]*s,
            data[1][0]*s, data[1][1]*s, data[1][2]*s,
            data[2][0]*s, data[2][1]*s, data[2][2]*s
        );
    }
    
    constexpr Mat3 operator+(const Mat3& m) const noexcept {
        return Mat3(
            data[0][0]+m.data[0][0], data[0][1]+m.data[0][1], data[0][2]+m.data[0][2],
            data[1][0]+m.data[1][0], data[1][1]+m.data[1][1], data[1][2]+m.data[1][2],
            data[2][0]+m.data[2][0], data[2][1]+m.data[2][1], data[2][2]+m.data[2][2]
        );
    }
    
    constexpr Mat3 transpose() const noexcept {
        return Mat3(
            data[0][0], data[1][0], data[2][0],
            data[0][1], data[1][1], data[2][1],
            data[0][2], data[1][2], data[2][2]
        );
    }
    
    constexpr double determinant() const noexcept {
        return data[0][0] * (data[1][1] * data[2][2] - data[2][1] * data[1][2])
             - data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
             + data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
    }
    
    Mat3 inverse() const noexcept {
        double det = determinant();
        if (std::abs(det) < 1e-12) return Mat3();
        double invDet = 1.0 / det;
        
        return Mat3(
            (data[1][1] * data[2][2] - data[2][1] * data[1][2]) * invDet,
            (data[0][2] * data[2][1] - data[0][1] * data[2][2]) * invDet,
            (data[0][1] * data[1][2] - data[0][2] * data[1][1]) * invDet,
            (data[1][2] * data[2][0] - data[1][0] * data[2][2]) * invDet,
            (data[0][0] * data[2][2] - data[0][2] * data[2][0]) * invDet,
            (data[1][0] * data[0][2] - data[0][0] * data[1][2]) * invDet,
            (data[1][0] * data[2][1] - data[2][0] * data[1][1]) * invDet,
            (data[2][0] * data[0][1] - data[0][0] * data[2][1]) * invDet,
            (data[0][0] * data[1][1] - data[1][0] * data[0][1]) * invDet
        );
    }
    
    static Mat3 fromQuaternion(const Quaternion& q) noexcept {
        double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
        
        return Mat3(
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
            2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
            2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)
        );
    }
    
    static constexpr Mat3 skew(const Vec3& v) noexcept {
        return Mat3(0, -v.z, v.y, v.z, 0, -v.x, -v.y, v.x, 0);
    }
};

struct MotorCommand {
    double rpm[4];
    
    constexpr MotorCommand() noexcept : rpm{0, 0, 0, 0} {}
    constexpr MotorCommand(double m0, double m1, double m2, double m3) noexcept : rpm{m0, m1, m2, m3} {}
    
    constexpr double operator[](int i) const noexcept { return rpm[i]; }
    double& operator[](int i) noexcept { return rpm[i]; }
    
    static constexpr MotorCommand hover(double rpmVal) noexcept {
        return MotorCommand(rpmVal, rpmVal, rpmVal, rpmVal);
    }
};

struct State {
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 angularVelocity;
    double motorRPM[4];
    double batteryVoltage;
    double time;
    
    constexpr State() noexcept 
        : position(), velocity(), orientation(), angularVelocity()
        , motorRPM{0,0,0,0}, batteryVoltage(16.8), time(0) {}
    
    Vec3 getLinearAcceleration(double mass, const Vec3& force) const noexcept {
        return force / mass;
    }
};

struct IMUReading {
    Vec3 accelerometer;
    Vec3 gyroscope;
    Vec3 magnetometer;
    double barometer;
    double timestamp;
    
    constexpr IMUReading() noexcept 
        : accelerometer(), gyroscope(), magnetometer()
        , barometer(101325.0), timestamp(0) {}
};

struct WindField {
    Vec3 meanVelocity;
    Vec3 turbulence;
    double gustMagnitude;
    double gustFrequency;
    
    constexpr WindField() noexcept 
        : meanVelocity(), turbulence(), gustMagnitude(0), gustFrequency(0) {}
    
    Vec3 getVelocityAt(const Vec3& position, double time) const noexcept {
        double gust = gustMagnitude * std::sin(gustFrequency * time);
        return meanVelocity + turbulence * gust;
    }
};
