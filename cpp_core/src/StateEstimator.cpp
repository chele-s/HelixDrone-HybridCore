#include "StateEstimator.h"
#include <cstring>

ErrorStateKalmanFilter::ErrorStateKalmanFilter() noexcept
    : nominal_(), error_(), P_(SquareMatrix<ERROR_DIM>::identity()), Q_(), noise_()
    , gravity_(0, 0, -9.81), mag_reference_(0.22, 0, 0.42) {
    reset();
}

ErrorStateKalmanFilter::ErrorStateKalmanFilter(const SensorNoise& noise) noexcept
    : nominal_(), error_(), P_(SquareMatrix<ERROR_DIM>::identity()), Q_(), noise_(noise)
    , gravity_(0, 0, -9.81), mag_reference_(0.22, 0, 0.42) {
    reset();
}

void ErrorStateKalmanFilter::reset() noexcept {
    nominal_.position = Vec3();
    nominal_.velocity = Vec3();
    nominal_.orientation = Quaternion();
    nominal_.accel_bias = Vec3();
    nominal_.gyro_bias = Vec3();
    
    error_.reset();
    
    P_ = SquareMatrix<ERROR_DIM>::identity();
    P_.setDiagonal(0, 3, 1.0);
    P_.setDiagonal(3, 3, 0.1);
    P_.setDiagonal(6, 3, 0.01);
    P_.setDiagonal(9, 3, 0.01);
    P_.setDiagonal(12, 3, 0.001);
    
    Q_ = SquareMatrix<ERROR_DIM>::identity();
    Q_.setDiagonal(0, 3, 0.001);
    Q_.setDiagonal(3, 3, 0.01);
    Q_.setDiagonal(6, 3, 0.0001);
    Q_.setDiagonal(9, 3, 0.00001);
    Q_.setDiagonal(12, 3, 0.000001);
}

void ErrorStateKalmanFilter::reset(const Vec3& pos, const Vec3& vel, const Quaternion& ori) noexcept {
    reset();
    nominal_.position = pos;
    nominal_.velocity = vel;
    nominal_.orientation = ori.normalized();
}

Mat3 ErrorStateKalmanFilter::skewSymmetric(const Vec3& v) const noexcept {
    Mat3 m;
    m.data[0][0] = 0;      m.data[0][1] = -v.z;   m.data[0][2] = v.y;
    m.data[1][0] = v.z;    m.data[1][1] = 0;      m.data[1][2] = -v.x;
    m.data[2][0] = -v.y;   m.data[2][1] = v.x;    m.data[2][2] = 0;
    return m;
}

Mat3 ErrorStateKalmanFilter::quaternionToRotation(const Quaternion& q) const noexcept {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    Mat3 R;
    R.data[0][0] = 1 - 2*(y*y + z*z);  R.data[0][1] = 2*(x*y - w*z);      R.data[0][2] = 2*(x*z + w*y);
    R.data[1][0] = 2*(x*y + w*z);      R.data[1][1] = 1 - 2*(x*x + z*z);  R.data[1][2] = 2*(y*z - w*x);
    R.data[2][0] = 2*(x*z - w*y);      R.data[2][1] = 2*(y*z + w*x);      R.data[2][2] = 1 - 2*(x*x + y*y);
    return R;
}

Quaternion ErrorStateKalmanFilter::smallAngleQuaternion(const Vec3& dtheta) const noexcept {
    double angle = std::sqrt(dtheta.x*dtheta.x + dtheta.y*dtheta.y + dtheta.z*dtheta.z);
    
    if (angle < 1e-12) {
        return Quaternion(1.0, dtheta.x * 0.5, dtheta.y * 0.5, dtheta.z * 0.5).normalized();
    }
    
    double half_angle = angle * 0.5;
    double s = std::sin(half_angle) / angle;
    
    return Quaternion(std::cos(half_angle), dtheta.x * s, dtheta.y * s, dtheta.z * s);
}

void ErrorStateKalmanFilter::propagateNominal(const Vec3& accel, const Vec3& gyro, double dt) noexcept {
    Vec3 accel_corrected = accel - nominal_.accel_bias;
    Vec3 gyro_corrected = gyro - nominal_.gyro_bias;
    
    Mat3 R = quaternionToRotation(nominal_.orientation);
    Vec3 accel_world;
    accel_world.x = R.data[0][0]*accel_corrected.x + R.data[0][1]*accel_corrected.y + R.data[0][2]*accel_corrected.z;
    accel_world.y = R.data[1][0]*accel_corrected.x + R.data[1][1]*accel_corrected.y + R.data[1][2]*accel_corrected.z;
    accel_world.z = R.data[2][0]*accel_corrected.x + R.data[2][1]*accel_corrected.y + R.data[2][2]*accel_corrected.z;
    accel_world = accel_world + gravity_;
    
    nominal_.position = nominal_.position + nominal_.velocity * dt + accel_world * (0.5 * dt * dt);
    nominal_.velocity = nominal_.velocity + accel_world * dt;
    
    double omega_mag = gyro_corrected.norm();
    if (omega_mag > 1e-10) {
        double half_angle = omega_mag * dt * 0.5;
        double s = std::sin(half_angle) / omega_mag;
        Quaternion dq(std::cos(half_angle), gyro_corrected.x * s, gyro_corrected.y * s, gyro_corrected.z * s);
        nominal_.orientation = (nominal_.orientation * dq).normalized();
    }
}

SquareMatrix<ErrorStateKalmanFilter::ERROR_DIM> ErrorStateKalmanFilter::computeF(
    const Vec3& accel, double dt
) const noexcept {
    SquareMatrix<ERROR_DIM> F = SquareMatrix<ERROR_DIM>::identity();
    
    for (int i = 0; i < 3; ++i) {
        F(i, i + 3) = dt;
    }
    
    Vec3 accel_corrected = accel - nominal_.accel_bias;
    Mat3 R = quaternionToRotation(nominal_.orientation);
    
    Vec3 Ra;
    Ra.x = R.data[0][0]*accel_corrected.x + R.data[0][1]*accel_corrected.y + R.data[0][2]*accel_corrected.z;
    Ra.y = R.data[1][0]*accel_corrected.x + R.data[1][1]*accel_corrected.y + R.data[1][2]*accel_corrected.z;
    Ra.z = R.data[2][0]*accel_corrected.x + R.data[2][1]*accel_corrected.y + R.data[2][2]*accel_corrected.z;
    
    Mat3 Ra_skew = skewSymmetric(Ra);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F(3 + i, 6 + j) = -Ra_skew.data[i][j] * dt;
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F(3 + i, 9 + j) = -R.data[i][j] * dt;
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        F(6 + i, 12 + i) = -dt;
    }
    
    return F;
}

void ErrorStateKalmanFilter::injectError() noexcept {
    nominal_.position = nominal_.position + error_.position();
    nominal_.velocity = nominal_.velocity + error_.velocity();
    
    Quaternion dq = smallAngleQuaternion(error_.angle());
    nominal_.orientation = (nominal_.orientation * dq).normalized();
    
    nominal_.accel_bias = nominal_.accel_bias + error_.accel_bias();
    nominal_.gyro_bias = nominal_.gyro_bias + error_.gyro_bias();
}

void ErrorStateKalmanFilter::resetError() noexcept {
    error_.reset();
}

void ErrorStateKalmanFilter::predict(const Vec3& accel, const Vec3& gyro, double dt) noexcept {
    propagateNominal(accel, gyro, dt);
    
    SquareMatrix<ERROR_DIM> F = computeF(accel, dt);
    SquareMatrix<ERROR_DIM> Ft = F.transpose();
    SquareMatrix<ERROR_DIM> FP = F * P_;
    P_ = FP * Ft + Q_ * (dt * dt);
    
    for (int i = 0; i < ERROR_DIM; ++i) {
        P_(i, i) = std::max(P_(i, i), 1e-12);
    }
}

template<int M>
void ErrorStateKalmanFilter::applyUpdate(
    const std::array<double, M>& innovation,
    const std::array<std::array<double, ERROR_DIM>, M>& H,
    const std::array<double, M>& R_diag
) noexcept {
    std::array<std::array<double, M>, ERROR_DIM> PHt;
    for (int i = 0; i < ERROR_DIM; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;
            for (int k = 0; k < ERROR_DIM; ++k) {
                sum += P_(i, k) * H[j][k];
            }
            PHt[i][j] = sum;
        }
    }
    
    std::array<std::array<double, M>, M> S;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;
            for (int k = 0; k < ERROR_DIM; ++k) {
                sum += H[i][k] * PHt[k][j];
            }
            S[i][j] = sum + (i == j ? R_diag[i] : 0);
        }
    }
    
    std::array<std::array<double, M>, M> S_inv;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            S_inv[i][j] = 0;
    
    if constexpr (M == 1) {
        S_inv[0][0] = 1.0 / (S[0][0] + 1e-12);
    } else if constexpr (M == 3) {
        double det = S[0][0]*(S[1][1]*S[2][2] - S[1][2]*S[2][1])
                   - S[0][1]*(S[1][0]*S[2][2] - S[1][2]*S[2][0])
                   + S[0][2]*(S[1][0]*S[2][1] - S[1][1]*S[2][0]);
        double inv_det = 1.0 / (det + 1e-12);
        S_inv[0][0] = (S[1][1]*S[2][2] - S[1][2]*S[2][1]) * inv_det;
        S_inv[0][1] = (S[0][2]*S[2][1] - S[0][1]*S[2][2]) * inv_det;
        S_inv[0][2] = (S[0][1]*S[1][2] - S[0][2]*S[1][1]) * inv_det;
        S_inv[1][0] = (S[1][2]*S[2][0] - S[1][0]*S[2][2]) * inv_det;
        S_inv[1][1] = (S[0][0]*S[2][2] - S[0][2]*S[2][0]) * inv_det;
        S_inv[1][2] = (S[0][2]*S[1][0] - S[0][0]*S[1][2]) * inv_det;
        S_inv[2][0] = (S[1][0]*S[2][1] - S[1][1]*S[2][0]) * inv_det;
        S_inv[2][1] = (S[0][1]*S[2][0] - S[0][0]*S[2][1]) * inv_det;
        S_inv[2][2] = (S[0][0]*S[1][1] - S[0][1]*S[1][0]) * inv_det;
    } else if constexpr (M == 6) {
        for (int i = 0; i < 6; ++i) {
            S_inv[i][i] = 1.0 / (S[i][i] + 1e-12);
        }
    }
    
    std::array<std::array<double, M>, ERROR_DIM> K;
    for (int i = 0; i < ERROR_DIM; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += PHt[i][k] * S_inv[k][j];
            }
            K[i][j] = sum;
        }
    }
    
    for (int i = 0; i < ERROR_DIM; ++i) {
        double dx = 0;
        for (int j = 0; j < M; ++j) {
            dx += K[i][j] * innovation[j];
        }
        error_.data[i] += dx;
    }
    
    SquareMatrix<ERROR_DIM> KH;
    for (int i = 0; i < ERROR_DIM; ++i) {
        for (int j = 0; j < ERROR_DIM; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += K[i][k] * H[k][j];
            }
            KH(i, j) = sum;
        }
    }
    
    SquareMatrix<ERROR_DIM> I_KH = SquareMatrix<ERROR_DIM>::identity() - KH;
    P_ = I_KH * P_;
    
    injectError();
    resetError();
}

void ErrorStateKalmanFilter::updateGPS(const Vec3& gps_pos, const Vec3& gps_vel) noexcept {
    std::array<double, MEAS_GPS> innovation = {
        gps_pos.x - nominal_.position.x,
        gps_pos.y - nominal_.position.y,
        gps_pos.z - nominal_.position.z,
        gps_vel.x - nominal_.velocity.x,
        gps_vel.y - nominal_.velocity.y,
        gps_vel.z - nominal_.velocity.z
    };
    
    std::array<std::array<double, ERROR_DIM>, MEAS_GPS> H = {};
    H[0][0] = 1; H[1][1] = 1; H[2][2] = 1;
    H[3][3] = 1; H[4][4] = 1; H[5][5] = 1;
    
    std::array<double, MEAS_GPS> R = {
        noise_.gps_pos_std * noise_.gps_pos_std,
        noise_.gps_pos_std * noise_.gps_pos_std,
        noise_.gps_pos_std * noise_.gps_pos_std,
        noise_.gps_vel_std * noise_.gps_vel_std,
        noise_.gps_vel_std * noise_.gps_vel_std,
        noise_.gps_vel_std * noise_.gps_vel_std
    };
    
    applyUpdate<MEAS_GPS>(innovation, H, R);
}

void ErrorStateKalmanFilter::updateBarometer(double altitude) noexcept {
    std::array<double, MEAS_BARO> innovation = {
        altitude - nominal_.position.z
    };
    
    std::array<std::array<double, ERROR_DIM>, MEAS_BARO> H = {};
    H[0][2] = 1;
    
    std::array<double, MEAS_BARO> R = {
        noise_.baro_std * noise_.baro_std
    };
    
    applyUpdate<MEAS_BARO>(innovation, H, R);
}

void ErrorStateKalmanFilter::updateMagnetometer(const Vec3& mag) noexcept {
    Mat3 R_mat = quaternionToRotation(nominal_.orientation);
    Vec3 mag_pred;
    mag_pred.x = R_mat.data[0][0]*mag_reference_.x + R_mat.data[0][1]*mag_reference_.y + R_mat.data[0][2]*mag_reference_.z;
    mag_pred.y = R_mat.data[1][0]*mag_reference_.x + R_mat.data[1][1]*mag_reference_.y + R_mat.data[1][2]*mag_reference_.z;
    mag_pred.z = R_mat.data[2][0]*mag_reference_.x + R_mat.data[2][1]*mag_reference_.y + R_mat.data[2][2]*mag_reference_.z;
    
    std::array<double, MEAS_MAG> innovation = {
        mag.x - mag_pred.x,
        mag.y - mag_pred.y,
        mag.z - mag_pred.z
    };
    
    Mat3 mag_skew = skewSymmetric(mag_pred);
    std::array<std::array<double, ERROR_DIM>, MEAS_MAG> H = {};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            H[i][6 + j] = mag_skew.data[i][j];
        }
    }
    
    std::array<double, MEAS_MAG> R = {
        noise_.mag_std * noise_.mag_std,
        noise_.mag_std * noise_.mag_std,
        noise_.mag_std * noise_.mag_std
    };
    
    applyUpdate<MEAS_MAG>(innovation, H, R);
}

EKFState ErrorStateKalmanFilter::getState() const noexcept {
    return EKFState::fromNominal(nominal_);
}

double ErrorStateKalmanFilter::getPositionUncertainty() const noexcept {
    return std::sqrt(P_(0,0) + P_(1,1) + P_(2,2));
}

double ErrorStateKalmanFilter::getOrientationUncertainty() const noexcept {
    return std::sqrt(P_(6,6) + P_(7,7) + P_(8,8));
}

void ErrorStateKalmanFilter::setProcessNoise(double pos, double vel, double ori, double bias) noexcept {
    Q_.setDiagonal(0, 3, pos);
    Q_.setDiagonal(3, 3, vel);
    Q_.setDiagonal(6, 3, ori);
    Q_.setDiagonal(9, 6, bias);
}

void ErrorStateKalmanFilter::setMeasurementNoise(const SensorNoise& noise) noexcept {
    noise_ = noise;
}

SensorSimulator::SensorSimulator() noexcept
    : noise_(), rng_(std::random_device{}()), dist_(0.0, 1.0)
    , accel_bias_(), gyro_bias_() {}

SensorSimulator::SensorSimulator(const SensorNoise& noise) noexcept
    : noise_(noise), rng_(std::random_device{}()), dist_(0.0, 1.0)
    , accel_bias_(), gyro_bias_() {}

void SensorSimulator::reset() noexcept {
    accel_bias_ = Vec3();
    gyro_bias_ = Vec3();
    last_gps_time_ = 0;
    last_baro_time_ = 0;
    last_mag_time_ = 0;
    current_time_ = 0;
}

Vec3 SensorSimulator::addNoise(const Vec3& v, double std) noexcept {
    return Vec3(
        v.x + dist_(rng_) * std,
        v.y + dist_(rng_) * std,
        v.z + dist_(rng_) * std
    );
}

double SensorSimulator::addNoise(double v, double std) noexcept {
    return v + dist_(rng_) * std;
}

void SensorSimulator::updateBiases(double dt) noexcept {
    double accel_walk = noise_.accel_bias_stability * std::sqrt(dt);
    double gyro_walk = noise_.gyro_bias_stability * std::sqrt(dt);
    
    accel_bias_.x += dist_(rng_) * accel_walk;
    accel_bias_.y += dist_(rng_) * accel_walk;
    accel_bias_.z += dist_(rng_) * accel_walk;
    
    gyro_bias_.x += dist_(rng_) * gyro_walk;
    gyro_bias_.y += dist_(rng_) * gyro_walk;
    gyro_bias_.z += dist_(rng_) * gyro_walk;
}

SensorReading SensorSimulator::simulate(const State& true_state, const Vec3& true_accel, double dt) noexcept {
    current_time_ += dt;
    updateBiases(dt);
    
    SensorReading reading;
    reading.timestamp = current_time_;
    
    Vec3 gravity(0, 0, 9.81);
    Vec3 specific_force = true_state.orientation.inverseRotate(true_accel + gravity);
    reading.accelerometer = addNoise(specific_force + accel_bias_, noise_.accel_std);
    
    reading.gyroscope = addNoise(true_state.angularVelocity + gyro_bias_, noise_.gyro_std);
    
    double gps_period = 1.0 / noise_.gps_update_rate;
    if (current_time_ - last_gps_time_ >= gps_period) {
        reading.gps_position = addNoise(true_state.position, noise_.gps_pos_std);
        reading.gps_velocity = addNoise(true_state.velocity, noise_.gps_vel_std);
        reading.gps_valid = true;
        last_gps_time_ = current_time_;
    } else {
        reading.gps_valid = false;
    }
    
    double baro_period = 1.0 / noise_.baro_update_rate;
    if (current_time_ - last_baro_time_ >= baro_period) {
        reading.barometer = addNoise(true_state.position.z, noise_.baro_std);
        reading.baro_valid = true;
        last_baro_time_ = current_time_;
    } else {
        reading.baro_valid = false;
    }
    
    double mag_period = 1.0 / noise_.mag_update_rate;
    if (current_time_ - last_mag_time_ >= mag_period) {
        Vec3 earth_mag(0.22, 0, 0.42);
        reading.magnetometer = addNoise(true_state.orientation.inverseRotate(earth_mag), noise_.mag_std);
        reading.mag_valid = true;
        last_mag_time_ = current_time_;
    } else {
        reading.mag_valid = false;
    }
    
    return reading;
}

StateEstimator::StateEstimator() noexcept
    : eskf_(), sensor_(), noise_(), last_reading_(), initialized_(false) {}

StateEstimator::StateEstimator(const SensorNoise& noise) noexcept
    : eskf_(noise), sensor_(noise), noise_(noise), last_reading_(), initialized_(false) {}

void StateEstimator::reset() noexcept {
    eskf_.reset();
    sensor_.reset();
    initialized_ = false;
}

void StateEstimator::reset(const State& initial_state) noexcept {
    eskf_.reset(initial_state.position, initial_state.velocity, initial_state.orientation);
    sensor_.reset();
    initialized_ = true;
}

void StateEstimator::setNoise(const SensorNoise& noise) noexcept {
    noise_ = noise;
    eskf_.setMeasurementNoise(noise);
    sensor_.setNoise(noise);
}

EKFState StateEstimator::update(const State& true_state, const Vec3& true_accel, double dt) noexcept {
    last_reading_ = sensor_.simulate(true_state, true_accel, dt);
    
    if (!initialized_) {
        if (last_reading_.gps_valid) {
            eskf_.reset(last_reading_.gps_position, last_reading_.gps_velocity, true_state.orientation);
            initialized_ = true;
        }
        return eskf_.getState();
    }
    
    eskf_.predict(last_reading_.accelerometer, last_reading_.gyroscope, dt);
    
    if (last_reading_.gps_valid) {
        eskf_.updateGPS(last_reading_.gps_position, last_reading_.gps_velocity);
    }
    
    if (last_reading_.baro_valid) {
        eskf_.updateBarometer(last_reading_.barometer);
    }
    
    if (last_reading_.mag_valid) {
        eskf_.updateMagnetometer(last_reading_.magnetometer);
    }
    
    return eskf_.getState();
}

double StateEstimator::getPositionError(const Vec3& true_pos) const noexcept {
    Vec3 est = eskf_.getPosition();
    return std::sqrt(
        (true_pos.x - est.x) * (true_pos.x - est.x) +
        (true_pos.y - est.y) * (true_pos.y - est.y) +
        (true_pos.z - est.z) * (true_pos.z - est.z)
    );
}

double StateEstimator::getVelocityError(const Vec3& true_vel) const noexcept {
    Vec3 est = eskf_.getVelocity();
    return std::sqrt(
        (true_vel.x - est.x) * (true_vel.x - est.x) +
        (true_vel.y - est.y) * (true_vel.y - est.y) +
        (true_vel.z - est.z) * (true_vel.z - est.z)
    );
}

double StateEstimator::getOrientationError(const Quaternion& true_ori) const noexcept {
    Quaternion est = eskf_.getOrientation();
    Quaternion diff = true_ori.inverse() * est;
    return 2.0 * std::acos(std::min(1.0, std::abs(diff.w)));
}

template void ErrorStateKalmanFilter::applyUpdate<1>(
    const std::array<double, 1>&,
    const std::array<std::array<double, ERROR_DIM>, 1>&,
    const std::array<double, 1>&
) noexcept;

template void ErrorStateKalmanFilter::applyUpdate<3>(
    const std::array<double, 3>&,
    const std::array<std::array<double, ERROR_DIM>, 3>&,
    const std::array<double, 3>&
) noexcept;

template void ErrorStateKalmanFilter::applyUpdate<6>(
    const std::array<double, 6>&,
    const std::array<std::array<double, ERROR_DIM>, 6>&,
    const std::array<double, 6>&
) noexcept;
