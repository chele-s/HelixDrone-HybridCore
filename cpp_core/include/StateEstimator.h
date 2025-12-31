#pragma once
#include "Types.h"
#include <array>
#include <random>
#include <cmath>

struct SensorNoise {
    double accel_std = 0.5;
    double gyro_std = 0.02;
    double gps_pos_std = 2.0;
    double gps_vel_std = 0.3;
    double baro_std = 0.5;
    double mag_std = 0.05;
    
    double gps_update_rate = 10.0;
    double baro_update_rate = 50.0;
    double mag_update_rate = 100.0;
    
    double accel_bias_stability = 0.001;
    double gyro_bias_stability = 0.0001;
};

struct SensorReading {
    Vec3 accelerometer;
    Vec3 gyroscope;
    Vec3 gps_position;
    Vec3 gps_velocity;
    double barometer;
    Vec3 magnetometer;
    double timestamp;
    
    bool gps_valid = false;
    bool baro_valid = false;
    bool mag_valid = false;
};

struct NominalState {
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 accel_bias;
    Vec3 gyro_bias;
};

struct ErrorState {
    static constexpr int DIM = 15;
    double data[DIM];
    
    ErrorState() noexcept { reset(); }
    
    void reset() noexcept {
        for (int i = 0; i < DIM; ++i) data[i] = 0.0;
    }
    
    Vec3 position() const noexcept { return Vec3(data[0], data[1], data[2]); }
    Vec3 velocity() const noexcept { return Vec3(data[3], data[4], data[5]); }
    Vec3 angle() const noexcept { return Vec3(data[6], data[7], data[8]); }
    Vec3 accel_bias() const noexcept { return Vec3(data[9], data[10], data[11]); }
    Vec3 gyro_bias() const noexcept { return Vec3(data[12], data[13], data[14]); }
    
    void set_position(const Vec3& v) noexcept { data[0] = v.x; data[1] = v.y; data[2] = v.z; }
    void set_velocity(const Vec3& v) noexcept { data[3] = v.x; data[4] = v.y; data[5] = v.z; }
    void set_angle(const Vec3& v) noexcept { data[6] = v.x; data[7] = v.y; data[8] = v.z; }
    void set_accel_bias(const Vec3& v) noexcept { data[9] = v.x; data[10] = v.y; data[11] = v.z; }
    void set_gyro_bias(const Vec3& v) noexcept { data[12] = v.x; data[13] = v.y; data[14] = v.z; }
};

struct EKFState {
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 accel_bias;
    Vec3 gyro_bias;
    
    static constexpr int DIM = 16;
    
    std::array<double, DIM> toArray() const noexcept {
        return {
            position.x, position.y, position.z,
            velocity.x, velocity.y, velocity.z,
            orientation.w, orientation.x, orientation.y, orientation.z,
            accel_bias.x, accel_bias.y, accel_bias.z,
            gyro_bias.x, gyro_bias.y, gyro_bias.z
        };
    }
    
    static EKFState fromNominal(const NominalState& n) noexcept {
        EKFState s;
        s.position = n.position;
        s.velocity = n.velocity;
        s.orientation = n.orientation;
        s.accel_bias = n.accel_bias;
        s.gyro_bias = n.gyro_bias;
        return s;
    }
};

template<int N>
class SquareMatrix {
public:
    double data[N][N];
    
    SquareMatrix() noexcept {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                data[i][j] = 0.0;
    }
    
    static SquareMatrix identity() noexcept {
        SquareMatrix m;
        for (int i = 0; i < N; ++i) m.data[i][i] = 1.0;
        return m;
    }
    
    static SquareMatrix diagonal(double val) noexcept {
        SquareMatrix m;
        for (int i = 0; i < N; ++i) m.data[i][i] = val;
        return m;
    }
    
    SquareMatrix operator+(const SquareMatrix& other) const noexcept {
        SquareMatrix r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.data[i][j] = data[i][j] + other.data[i][j];
        return r;
    }
    
    SquareMatrix operator-(const SquareMatrix& other) const noexcept {
        SquareMatrix r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.data[i][j] = data[i][j] - other.data[i][j];
        return r;
    }
    
    SquareMatrix operator*(const SquareMatrix& other) const noexcept {
        SquareMatrix r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k)
                    r.data[i][j] += data[i][k] * other.data[k][j];
        return r;
    }
    
    SquareMatrix operator*(double s) const noexcept {
        SquareMatrix r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.data[i][j] = data[i][j] * s;
        return r;
    }
    
    SquareMatrix transpose() const noexcept {
        SquareMatrix r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.data[i][j] = data[j][i];
        return r;
    }
    
    void setDiagonal(int start, int count, double val) noexcept {
        for (int i = 0; i < count && (start + i) < N; ++i)
            data[start + i][start + i] = val;
    }
    
    void setBlock3x3(int row, int col, const Mat3& m) noexcept {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (row + i < N && col + j < N)
                    data[row + i][col + j] = m.data[i][j];
    }
    
    double& operator()(int i, int j) noexcept { return data[i][j]; }
    double operator()(int i, int j) const noexcept { return data[i][j]; }
};

class ErrorStateKalmanFilter {
public:
    static constexpr int ERROR_DIM = 15;
    static constexpr int MEAS_GPS = 6;
    static constexpr int MEAS_BARO = 1;
    static constexpr int MEAS_MAG = 3;
    
    ErrorStateKalmanFilter() noexcept;
    explicit ErrorStateKalmanFilter(const SensorNoise& noise) noexcept;
    
    void reset() noexcept;
    void reset(const Vec3& pos, const Vec3& vel, const Quaternion& ori) noexcept;
    
    void predict(const Vec3& accel, const Vec3& gyro, double dt) noexcept;
    
    void updateGPS(const Vec3& gps_pos, const Vec3& gps_vel) noexcept;
    void updateBarometer(double altitude) noexcept;
    void updateMagnetometer(const Vec3& mag) noexcept;
    
    EKFState getState() const noexcept;
    Vec3 getPosition() const noexcept { return nominal_.position; }
    Vec3 getVelocity() const noexcept { return nominal_.velocity; }
    Quaternion getOrientation() const noexcept { return nominal_.orientation; }
    Vec3 getAccelBias() const noexcept { return nominal_.accel_bias; }
    Vec3 getGyroBias() const noexcept { return nominal_.gyro_bias; }
    
    double getPositionUncertainty() const noexcept;
    double getOrientationUncertainty() const noexcept;
    
    void setProcessNoise(double pos, double vel, double ori, double bias) noexcept;
    void setMeasurementNoise(const SensorNoise& noise) noexcept;
    
    const SquareMatrix<ERROR_DIM>& getCovariance() const noexcept { return P_; }
    
private:
    NominalState nominal_;
    ErrorState error_;
    SquareMatrix<ERROR_DIM> P_;
    SquareMatrix<ERROR_DIM> Q_;
    SensorNoise noise_;
    
    Vec3 gravity_;
    Vec3 mag_reference_;
    
    void propagateNominal(const Vec3& accel, const Vec3& gyro, double dt) noexcept;
    SquareMatrix<ERROR_DIM> computeF(const Vec3& accel, double dt) const noexcept;
    void injectError() noexcept;
    void resetError() noexcept;
    
    Mat3 skewSymmetric(const Vec3& v) const noexcept;
    Mat3 quaternionToRotation(const Quaternion& q) const noexcept;
    Quaternion smallAngleQuaternion(const Vec3& dtheta) const noexcept;
    
    template<int M>
    void applyUpdate(
        const std::array<double, M>& innovation,
        const std::array<std::array<double, ERROR_DIM>, M>& H,
        const std::array<double, M>& R_diag
    ) noexcept;
};

class SensorSimulator {
public:
    SensorSimulator() noexcept;
    explicit SensorSimulator(const SensorNoise& noise) noexcept;
    
    void setNoise(const SensorNoise& noise) noexcept { noise_ = noise; }
    
    SensorReading simulate(const State& true_state, const Vec3& true_accel, double dt) noexcept;
    
    void reset() noexcept;
    
private:
    SensorNoise noise_;
    std::mt19937 rng_;
    std::normal_distribution<double> dist_;
    
    Vec3 accel_bias_;
    Vec3 gyro_bias_;
    
    double last_gps_time_ = 0;
    double last_baro_time_ = 0;
    double last_mag_time_ = 0;
    double current_time_ = 0;
    
    Vec3 addNoise(const Vec3& v, double std) noexcept;
    double addNoise(double v, double std) noexcept;
    void updateBiases(double dt) noexcept;
};

class StateEstimator {
public:
    StateEstimator() noexcept;
    explicit StateEstimator(const SensorNoise& noise) noexcept;
    
    void reset() noexcept;
    void reset(const State& initial_state) noexcept;
    
    EKFState update(const State& true_state, const Vec3& true_accel, double dt) noexcept;
    
    EKFState getEstimatedState() const noexcept { return eskf_.getState(); }
    SensorReading getLastSensorReading() const noexcept { return last_reading_; }
    
    void setNoise(const SensorNoise& noise) noexcept;
    SensorNoise getNoise() const noexcept { return noise_; }
    
    double getPositionError(const Vec3& true_pos) const noexcept;
    double getVelocityError(const Vec3& true_vel) const noexcept;
    double getOrientationError(const Quaternion& true_ori) const noexcept;
    
    bool isInitialized() const noexcept { return initialized_; }
    
private:
    ErrorStateKalmanFilter eskf_;
    SensorSimulator sensor_;
    SensorNoise noise_;
    SensorReading last_reading_;
    bool initialized_ = false;
};

using ExtendedKalmanFilter = ErrorStateKalmanFilter;
