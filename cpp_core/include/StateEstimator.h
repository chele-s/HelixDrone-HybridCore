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
    
    double cable_angle_std = 0.05;
    double cable_angle_update_rate = 100.0;
    bool cable_sensor_enabled = false;
};

struct RobustnessConfig {
    bool enable_chi_square_gating = true;
    bool enable_fault_detection = true;
    bool enable_adaptive_noise = true;
    bool enable_state_dependent_noise = true;
    
    double chi_square_threshold_1dof = 6.635;
    double chi_square_threshold_3dof = 11.345;
    
    int max_consecutive_rejections = 10;
    double adaptive_alpha = 0.98;
    
    double gyro_noise_scale_coeff = 0.1;
    double accel_noise_scale_coeff = 0.05;
    
    double min_innovation_variance = 1e-6;
    double max_innovation_variance = 1e6;
    
    bool enable_external_force_estimation = false;
    double external_force_process_noise = 1.0;
};

struct CableSensorReading {
    double theta_x = 0;
    double theta_y = 0;
    double tension = 0;
    double timestamp = 0;
    bool valid = false;
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
    
    CableSensorReading cable;
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

struct ConsistencyMetrics {
    double current_nis = 0.0;
    double avg_nis = 0.0;
    double nees = 0.0;
    int total_updates = 0;
    int rejected_updates = 0;
    double rejection_rate = 0.0;
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
    static constexpr int MEAS_BARO = 1;
    static constexpr int MEAS_MAG = 3;
    static constexpr int NUM_SENSORS = 4;
    
    ErrorStateKalmanFilter() noexcept;
    explicit ErrorStateKalmanFilter(const SensorNoise& noise) noexcept;
    
    void reset() noexcept;
    void reset(const Vec3& pos, const Vec3& vel, const Quaternion& ori) noexcept;
    
    bool predict(const Vec3& accel, const Vec3& gyro, double dt) noexcept;
    
    bool updateGPSPosition(const Vec3& gps_pos) noexcept;
    bool updateGPSVelocity(const Vec3& gps_vel) noexcept;
    bool updateBarometer(double altitude) noexcept;
    bool updateMagnetometer(const Vec3& mag) noexcept;
    
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
    void setRobustnessConfig(const RobustnessConfig& config) noexcept { robustness_config_ = config; }
    
    const SquareMatrix<ERROR_DIM>& getCovariance() const noexcept { return P_; }
    
    bool isSensorHealthy(int sensor_id) const noexcept;
    double getCurrentNIS() const noexcept { return current_nis_; }
    ConsistencyMetrics getConsistencyMetrics() const noexcept;
    
    double computeNEES(const Vec3& true_pos, const Vec3& true_vel, const Quaternion& true_ori) const noexcept;
    bool isYawObservable() const noexcept;
    bool isAccelValid() const noexcept;
    
private:
    NominalState nominal_;
    ErrorState error_;
    SquareMatrix<ERROR_DIM> P_;
    SquareMatrix<ERROR_DIM> Q_;
    SensorNoise noise_;
    RobustnessConfig robustness_config_;
    
    Vec3 gravity_;
    Vec3 mag_reference_;
    
    double current_nis_ = 0.0;
    double nis_sum_ = 0.0;
    int nis_count_ = 0;
    std::array<int, NUM_SENSORS> consecutive_rejections_;
    std::array<bool, NUM_SENSORS> sensor_health_;
    std::array<double, NUM_SENSORS> adaptive_R_;
    int total_updates_ = 0;
    int rejected_updates_ = 0;
    
    Vec3 last_accel_;
    Vec3 last_gyro_;
    
    void propagateNominal(const Vec3& accel, const Vec3& gyro, double dt) noexcept;
    SquareMatrix<ERROR_DIM> computeF(const Vec3& accel, const Vec3& gyro, double dt) const noexcept;
    SquareMatrix<ERROR_DIM> computeResetJacobian(const Vec3& error_theta) const noexcept;
    void injectError() noexcept;
    void resetError() noexcept;
    
    Mat3 skewSymmetric(const Vec3& v) const noexcept;
    Mat3 quaternionToRotation(const Quaternion& q) const noexcept;
    Quaternion smallAngleQuaternion(const Vec3& dtheta) const noexcept;
    
    bool checkInnovation(double nis, double threshold, int sensor_id) noexcept;
    
    template<int M>
    bool applyUpdate(
        const std::array<double, M>& innovation,
        const std::array<std::array<double, ERROR_DIM>, M>& H,
        std::array<double, M>& R_diag,
        int sensor_id,
        double chi_square_threshold
    ) noexcept;
};

class SensorSimulator {
public:
    SensorSimulator() noexcept;
    explicit SensorSimulator(const SensorNoise& noise) noexcept;
    
    void setNoise(const SensorNoise& noise) noexcept { noise_ = noise; }
    
    SensorReading simulate(const State& true_state, const Vec3& true_accel, double dt) noexcept;
    
    CableSensorReading simulateCableAngle(double true_theta_x, double true_theta_y, double true_tension, double dt) noexcept;
    
    void reset() noexcept;
    
    void injectGPSFailure(bool failed) noexcept { gps_failed_ = failed; }
    void injectBaroFailure(bool failed) noexcept { baro_failed_ = failed; }
    void injectMagFailure(bool failed) noexcept { mag_failed_ = failed; }
    void injectCableFailure(bool failed) noexcept { cable_failed_ = failed; }
    void injectGPSSpoof(const Vec3& offset) noexcept { gps_spoof_offset_ = offset; gps_spoofed_ = true; }
    void clearAllFailures() noexcept { gps_failed_ = baro_failed_ = mag_failed_ = gps_spoofed_ = cable_failed_ = false; }
    
private:
    SensorNoise noise_;
    std::mt19937 rng_;
    std::normal_distribution<double> dist_;
    
    Vec3 accel_bias_;
    Vec3 gyro_bias_;
    
    double last_gps_time_ = -1e9;
    double last_baro_time_ = -1e9;
    double last_mag_time_ = -1e9;
    double last_cable_time_ = -1e9;
    double current_time_ = 0;
    
    bool gps_failed_ = false;
    bool baro_failed_ = false;
    bool mag_failed_ = false;
    bool gps_spoofed_ = false;
    bool cable_failed_ = false;
    Vec3 gps_spoof_offset_;
    
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
    void setRobustnessConfig(const RobustnessConfig& config) noexcept { eskf_.setRobustnessConfig(config); }
    
    double getPositionError(const Vec3& true_pos) const noexcept;
    double getVelocityError(const Vec3& true_vel) const noexcept;
    double getOrientationError(const Quaternion& true_ori) const noexcept;
    
    bool isInitialized() const noexcept { return initialized_; }
    
    SensorSimulator& getSensorSimulator() noexcept { return sensor_; }
    ErrorStateKalmanFilter& getFilter() noexcept { return eskf_; }
    
private:
    ErrorStateKalmanFilter eskf_;
    SensorSimulator sensor_;
    SensorNoise noise_;
    SensorReading last_reading_;
    bool initialized_ = false;
};

using ExtendedKalmanFilter = ErrorStateKalmanFilter;
