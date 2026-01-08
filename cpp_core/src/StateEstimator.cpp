#include "StateEstimator.h"
#include <cstring>

ErrorStateKalmanFilter::ErrorStateKalmanFilter() noexcept
    : nominal_(), error_(), P_(SquareMatrix<ERROR_DIM>::identity()), Q_(), noise_()
    , robustness_config_()
    , gravity_(0, 0, -9.81), mag_reference_(0.22, 0, 0.42)
    , current_nis_(0), nis_sum_(0), nis_count_(0)
    , consecutive_rejections_(), sensor_health_(), adaptive_R_()
    , total_updates_(0), rejected_updates_(0)
    , last_accel_(), last_gyro_() {
    reset();
}

ErrorStateKalmanFilter::ErrorStateKalmanFilter(const SensorNoise& noise) noexcept
    : nominal_(), error_(), P_(SquareMatrix<ERROR_DIM>::identity()), Q_(), noise_(noise)
    , robustness_config_()
    , gravity_(0, 0, -9.81), mag_reference_(0.22, 0, 0.42)
    , current_nis_(0), nis_sum_(0), nis_count_(0)
    , consecutive_rejections_(), sensor_health_(), adaptive_R_()
    , total_updates_(0), rejected_updates_(0)
    , last_accel_(), last_gyro_() {
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
    
    current_nis_ = 0;
    nis_sum_ = 0;
    nis_count_ = 0;
    total_updates_ = 0;
    rejected_updates_ = 0;
    
    last_accel_ = Vec3(0, 0, 9.81);
    last_gyro_ = Vec3();
    
    for (int i = 0; i < NUM_SENSORS; ++i) {
        consecutive_rejections_[i] = 0;
        sensor_health_[i] = true;
        adaptive_R_[i] = 1.0;
    }
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
    double angle_sq = dtheta.x*dtheta.x + dtheta.y*dtheta.y + dtheta.z*dtheta.z;
    
    if (angle_sq < 1e-12) {
        double w = 1.0 - angle_sq / 8.0;
        double s = 0.5 - angle_sq / 48.0;
        return Quaternion(w, dtheta.x * s, dtheta.y * s, dtheta.z * s).normalized();
    }
    
    double angle = std::sqrt(angle_sq);
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
        nominal_.orientation = (dq * nominal_.orientation).normalized();
    }
    
    last_accel_ = accel;
    last_gyro_ = gyro;
}

SquareMatrix<ErrorStateKalmanFilter::ERROR_DIM> ErrorStateKalmanFilter::computeF(
    const Vec3& accel, const Vec3& gyro, double dt
) const noexcept {
    SquareMatrix<ERROR_DIM> F = SquareMatrix<ERROR_DIM>::identity();
    
    for (int i = 0; i < 3; ++i) {
        F(i, i + 3) = dt;
    }
    
    Vec3 accel_corrected = accel - nominal_.accel_bias;
    Vec3 gyro_corrected = gyro - nominal_.gyro_bias;
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
    
    Mat3 omega_skew = skewSymmetric(gyro_corrected);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F(6 + i, 6 + j) += -omega_skew.data[i][j] * dt;
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        F(6 + i, 12 + i) = -dt;
    }
    
    return F;
}

SquareMatrix<ErrorStateKalmanFilter::ERROR_DIM> ErrorStateKalmanFilter::computeResetJacobian(
    const Vec3& error_theta
) const noexcept {
    SquareMatrix<ERROR_DIM> G = SquareMatrix<ERROR_DIM>::identity();
    
    Mat3 skew = skewSymmetric(error_theta);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            G(6 + i, 6 + j) = (i == j ? 1.0 : 0.0) - 0.5 * skew.data[i][j];
        }
    }
    
    return G;
}

void ErrorStateKalmanFilter::injectError() noexcept {
    nominal_.position = nominal_.position + error_.position();
    nominal_.velocity = nominal_.velocity + error_.velocity();
    
    Quaternion dq = smallAngleQuaternion(error_.angle());
    nominal_.orientation = (dq * nominal_.orientation).normalized();
    
    nominal_.accel_bias = nominal_.accel_bias + error_.accel_bias();
    nominal_.gyro_bias = nominal_.gyro_bias + error_.gyro_bias();
}

void ErrorStateKalmanFilter::resetError() noexcept {
    SquareMatrix<ERROR_DIM> G = computeResetJacobian(error_.angle());
    SquareMatrix<ERROR_DIM> Gt = G.transpose();
    P_ = G * P_ * Gt;
    
    error_.reset();
}

bool ErrorStateKalmanFilter::isYawObservable() const noexcept {
    double vel_horiz_sq = nominal_.velocity.x * nominal_.velocity.x + 
                          nominal_.velocity.y * nominal_.velocity.y;
    return vel_horiz_sq > 0.25;
}

bool ErrorStateKalmanFilter::isAccelValid() const noexcept {
    double accel_mag = last_accel_.norm();
    return std::abs(accel_mag - 9.81) < 2.0;
}

double ErrorStateKalmanFilter::computeNEES(const Vec3& true_pos, const Vec3& true_vel, const Quaternion& true_ori) const noexcept {
    std::array<double, 9> error;
    error[0] = true_pos.x - nominal_.position.x;
    error[1] = true_pos.y - nominal_.position.y;
    error[2] = true_pos.z - nominal_.position.z;
    error[3] = true_vel.x - nominal_.velocity.x;
    error[4] = true_vel.y - nominal_.velocity.y;
    error[5] = true_vel.z - nominal_.velocity.z;
    
    Quaternion q_diff = true_ori * nominal_.orientation.inverse();
    if (q_diff.w < 0) {
        q_diff.w = -q_diff.w;
        q_diff.x = -q_diff.x;
        q_diff.y = -q_diff.y;
        q_diff.z = -q_diff.z;
    }
    error[6] = 2.0 * q_diff.x;
    error[7] = 2.0 * q_diff.y;
    error[8] = 2.0 * q_diff.z;
    
    double nees = 0;
    for (int i = 0; i < 9; ++i) {
        double p_inv = 1.0 / std::max(P_(i, i), 1e-12);
        nees += error[i] * error[i] * p_inv;
    }
    
    return nees;
}

bool ErrorStateKalmanFilter::predict(const Vec3& accel, const Vec3& gyro, double dt) noexcept {
    propagateNominal(accel, gyro, dt);
    
    SquareMatrix<ERROR_DIM> F = computeF(accel, gyro, dt);
    SquareMatrix<ERROR_DIM> Ft = F.transpose();
    
    SquareMatrix<ERROR_DIM> Q_scaled = Q_;
    
    if (robustness_config_.enable_state_dependent_noise) {
        double gyro_norm = gyro.norm();
        double accel_norm = accel.norm();
        double accel_dev = std::abs(accel_norm - 9.81);
        
        double q_vel_scale = 1.0 + robustness_config_.accel_noise_scale_coeff * accel_dev;
        double q_angle_scale = 1.0 + robustness_config_.gyro_noise_scale_coeff * gyro_norm;
        
        for (int i = 3; i < 6; ++i) Q_scaled(i, i) *= q_vel_scale;
        for (int i = 6; i < 9; ++i) Q_scaled(i, i) *= q_angle_scale;
    }
    
    SquareMatrix<ERROR_DIM> FP = F * P_;
    P_ = FP * Ft + Q_scaled * (dt * dt);
    
    SquareMatrix<ERROR_DIM> Pt = P_.transpose();
    P_ = (P_ + Pt) * 0.5;
    
    for (int i = 0; i < ERROR_DIM; ++i) {
        P_(i, i) = std::max(P_(i, i), 1e-12);
    }
    
    return true;
}

bool ErrorStateKalmanFilter::checkInnovation(double nis, double threshold, int sensor_id) noexcept {
    ++total_updates_;
    current_nis_ = nis;
    nis_sum_ += nis;
    ++nis_count_;
    
    if (!robustness_config_.enable_chi_square_gating) {
        return true;
    }
    
    bool valid = nis < threshold;
    
    if (!robustness_config_.enable_fault_detection) {
        if (!valid) ++rejected_updates_;
        return valid;
    }
    
    if (!valid) {
        ++rejected_updates_;
        ++consecutive_rejections_[sensor_id];
        if (consecutive_rejections_[sensor_id] > robustness_config_.max_consecutive_rejections) {
            sensor_health_[sensor_id] = false;
        }
        return false;
    }
    
    if (consecutive_rejections_[sensor_id] > 0) {
        --consecutive_rejections_[sensor_id];
    }
    if (consecutive_rejections_[sensor_id] == 0) {
        sensor_health_[sensor_id] = true;
    }
    
    return true;
}

template<int M>
bool ErrorStateKalmanFilter::applyUpdate(
    const std::array<double, M>& innovation,
    const std::array<std::array<double, ERROR_DIM>, M>& H,
    std::array<double, M>& R_diag,
    int sensor_id,
    double chi_square_threshold
) noexcept {
    if (robustness_config_.enable_fault_detection && !sensor_health_[sensor_id]) {
        return false;
    }
    
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
    
    std::array<std::array<double, M>, M> S_inv = {};
    
    if constexpr (M == 1) {
        double s_val = std::max(S[0][0], robustness_config_.min_innovation_variance);
        s_val = std::min(s_val, robustness_config_.max_innovation_variance);
        S_inv[0][0] = 1.0 / s_val;
    } else if constexpr (M == 3) {
        double det = S[0][0]*(S[1][1]*S[2][2] - S[1][2]*S[2][1])
                   - S[0][1]*(S[1][0]*S[2][2] - S[1][2]*S[2][0])
                   + S[0][2]*(S[1][0]*S[2][1] - S[1][1]*S[2][0]);
        
        double det_abs = std::abs(det);
        if (det_abs < robustness_config_.min_innovation_variance) {
            return false;
        }
        
        double inv_det = 1.0 / det;
        S_inv[0][0] = (S[1][1]*S[2][2] - S[1][2]*S[2][1]) * inv_det;
        S_inv[0][1] = (S[0][2]*S[2][1] - S[0][1]*S[2][2]) * inv_det;
        S_inv[0][2] = (S[0][1]*S[1][2] - S[0][2]*S[1][1]) * inv_det;
        S_inv[1][0] = (S[1][2]*S[2][0] - S[1][0]*S[2][2]) * inv_det;
        S_inv[1][1] = (S[0][0]*S[2][2] - S[0][2]*S[2][0]) * inv_det;
        S_inv[1][2] = (S[0][2]*S[1][0] - S[0][0]*S[1][2]) * inv_det;
        S_inv[2][0] = (S[1][0]*S[2][1] - S[1][1]*S[2][0]) * inv_det;
        S_inv[2][1] = (S[0][1]*S[2][0] - S[0][0]*S[2][1]) * inv_det;
        S_inv[2][2] = (S[0][0]*S[1][1] - S[0][1]*S[1][0]) * inv_det;
    }
    
    double nis = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            nis += innovation[i] * S_inv[i][j] * innovation[j];
        }
    }
    
    if (!checkInnovation(nis, chi_square_threshold, sensor_id)) {
        return false;
    }
    
    if (robustness_config_.enable_adaptive_noise) {
        double alpha = robustness_config_.adaptive_alpha;
        for (int i = 0; i < M; ++i) {
            double inn_sq = innovation[i] * innovation[i];
            R_diag[i] = alpha * R_diag[i] + (1.0 - alpha) * inn_sq;
            R_diag[i] = std::max(R_diag[i], robustness_config_.min_innovation_variance);
            R_diag[i] = std::min(R_diag[i], robustness_config_.max_innovation_variance);
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
    
    SquareMatrix<ERROR_DIM> I = SquareMatrix<ERROR_DIM>::identity();
    SquareMatrix<ERROR_DIM> I_KH = I - KH;
    SquareMatrix<ERROR_DIM> I_KH_T = I_KH.transpose();
    
    P_ = I_KH * P_ * I_KH_T;
    
    for (int i = 0; i < ERROR_DIM; ++i) {
        for (int j = 0; j < ERROR_DIM; ++j) {
            for (int k = 0; k < M; ++k) {
                P_(i, j) += K[i][k] * R_diag[k] * K[j][k];
            }
        }
    }
    
    SquareMatrix<ERROR_DIM> Pt = P_.transpose();
    P_ = (P_ + Pt) * 0.5;
    
    for (int i = 0; i < ERROR_DIM; ++i) {
        P_(i, i) = std::max(P_(i, i), 1e-12);
    }
    
    injectError();
    resetError();
    
    return true;
}

bool ErrorStateKalmanFilter::updateGPSPosition(const Vec3& gps_pos) noexcept {
    std::array<double, 3> innovation = {
        gps_pos.x - nominal_.position.x,
        gps_pos.y - nominal_.position.y,
        gps_pos.z - nominal_.position.z
    };
    
    std::array<std::array<double, ERROR_DIM>, 3> H = {};
    H[0][0] = 1; H[1][1] = 1; H[2][2] = 1;
    
    double r_val = noise_.gps_pos_std * noise_.gps_pos_std;
    std::array<double, 3> R = {r_val, r_val, r_val};
    
    return applyUpdate<3>(innovation, H, R, 0, robustness_config_.chi_square_threshold_3dof);
}

bool ErrorStateKalmanFilter::updateGPSVelocity(const Vec3& gps_vel) noexcept {
    std::array<double, 3> innovation = {
        gps_vel.x - nominal_.velocity.x,
        gps_vel.y - nominal_.velocity.y,
        gps_vel.z - nominal_.velocity.z
    };
    
    std::array<std::array<double, ERROR_DIM>, 3> H = {};
    H[0][3] = 1; H[1][4] = 1; H[2][5] = 1;
    
    double r_val = noise_.gps_vel_std * noise_.gps_vel_std;
    std::array<double, 3> R = {r_val, r_val, r_val};
    
    return applyUpdate<3>(innovation, H, R, 0, robustness_config_.chi_square_threshold_3dof);
}

bool ErrorStateKalmanFilter::updateBarometer(double altitude) noexcept {
    std::array<double, 1> innovation = {altitude - nominal_.position.z};
    
    std::array<std::array<double, ERROR_DIM>, 1> H = {};
    H[0][2] = 1;
    
    std::array<double, 1> R = {noise_.baro_std * noise_.baro_std};
    
    return applyUpdate<1>(innovation, H, R, 1, robustness_config_.chi_square_threshold_1dof);
}

bool ErrorStateKalmanFilter::updateMagnetometer(const Vec3& mag) noexcept {
    if (!isAccelValid()) {
        ++total_updates_;
        ++rejected_updates_;
        return false;
    }
    
    Mat3 R_mat = quaternionToRotation(nominal_.orientation);
    Vec3 mag_body_pred;
    Mat3 R_t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_t.data[i][j] = R_mat.data[j][i];
        }
    }
    mag_body_pred.x = R_t.data[0][0]*mag_reference_.x + R_t.data[0][1]*mag_reference_.y + R_t.data[0][2]*mag_reference_.z;
    mag_body_pred.y = R_t.data[1][0]*mag_reference_.x + R_t.data[1][1]*mag_reference_.y + R_t.data[1][2]*mag_reference_.z;
    mag_body_pred.z = R_t.data[2][0]*mag_reference_.x + R_t.data[2][1]*mag_reference_.y + R_t.data[2][2]*mag_reference_.z;
    
    double inn_horiz_sq = (mag.x - mag_body_pred.x)*(mag.x - mag_body_pred.x) +
                          (mag.y - mag_body_pred.y)*(mag.y - mag_body_pred.y);
    
    if (inn_horiz_sq > 0.25) {
        ++total_updates_;
        ++rejected_updates_;
        return false;
    }
    
    double pred_yaw = std::atan2(mag_body_pred.y, mag_body_pred.x);
    double meas_yaw = std::atan2(mag.y, mag.x);
    
    double yaw_inn = meas_yaw - pred_yaw;
    while (yaw_inn > 3.14159265359) yaw_inn -= 6.28318530718;
    while (yaw_inn < -3.14159265359) yaw_inn += 6.28318530718;
    
    std::array<double, 1> innovation = {yaw_inn};
    
    std::array<std::array<double, ERROR_DIM>, 1> H = {};
    H[0][8] = 1.0;
    
    double r_val = noise_.mag_std * noise_.mag_std * 10.0;
    std::array<double, 1> R = {r_val};
    
    return applyUpdate<1>(innovation, H, R, 2, robustness_config_.chi_square_threshold_1dof);
}

bool ErrorStateKalmanFilter::isSensorHealthy(int sensor_id) const noexcept {
    if (sensor_id >= 0 && sensor_id < NUM_SENSORS) {
        return sensor_health_[sensor_id];
    }
    return false;
}

ConsistencyMetrics ErrorStateKalmanFilter::getConsistencyMetrics() const noexcept {
    ConsistencyMetrics m;
    m.current_nis = current_nis_;
    m.avg_nis = nis_count_ > 0 ? nis_sum_ / nis_count_ : 0.0;
    m.total_updates = total_updates_;
    m.rejected_updates = rejected_updates_;
    m.rejection_rate = total_updates_ > 0 ? static_cast<double>(rejected_updates_) / total_updates_ : 0.0;
    return m;
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
    , accel_bias_(), gyro_bias_(), gps_spoof_offset_() {
    reset();
}

SensorSimulator::SensorSimulator(const SensorNoise& noise) noexcept
    : noise_(noise), rng_(std::random_device{}()), dist_(0.0, 1.0)
    , accel_bias_(), gyro_bias_(), gps_spoof_offset_() {
    reset();
}

void SensorSimulator::reset() noexcept {
    accel_bias_ = Vec3();
    gyro_bias_ = Vec3();
    last_gps_time_ = -1e9;
    last_baro_time_ = -1e9;
    last_mag_time_ = -1e9;
    last_cable_time_ = -1e9;
    current_time_ = 0;
    gps_failed_ = false;
    baro_failed_ = false;
    mag_failed_ = false;
    cable_failed_ = false;
    gps_spoofed_ = false;
    gps_spoof_offset_ = Vec3();
}

CableSensorReading SensorSimulator::simulateCableAngle(double true_theta_x, double true_theta_y, double true_tension, double dt) noexcept {
    current_time_ += dt;
    CableSensorReading reading;
    reading.timestamp = current_time_;
    reading.valid = false;
    
    if (cable_failed_ || !noise_.cable_sensor_enabled) {
        return reading;
    }
    
    double cable_period = 1.0 / noise_.cable_angle_update_rate;
    if (current_time_ - last_cable_time_ >= cable_period) {
        reading.theta_x = addNoise(true_theta_x, noise_.cable_angle_std);
        reading.theta_y = addNoise(true_theta_y, noise_.cable_angle_std);
        reading.tension = addNoise(true_tension, noise_.cable_angle_std * 10.0);
        reading.valid = true;
        last_cable_time_ = current_time_;
    }
    
    return reading;
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
    
    reading.gps_valid = false;
    if (!gps_failed_) {
        double gps_period = 1.0 / noise_.gps_update_rate;
        if (current_time_ - last_gps_time_ >= gps_period) {
            Vec3 pos = true_state.position;
            Vec3 vel = true_state.velocity;
            
            if (gps_spoofed_) {
                pos = pos + gps_spoof_offset_;
            }
            
            reading.gps_position = addNoise(pos, noise_.gps_pos_std);
            reading.gps_velocity = addNoise(vel, noise_.gps_vel_std);
            reading.gps_valid = true;
            last_gps_time_ = current_time_;
        }
    }
    
    reading.baro_valid = false;
    if (!baro_failed_) {
        double baro_period = 1.0 / noise_.baro_update_rate;
        if (current_time_ - last_baro_time_ >= baro_period) {
            reading.barometer = addNoise(true_state.position.z, noise_.baro_std);
            reading.baro_valid = true;
            last_baro_time_ = current_time_;
        }
    }
    
    reading.mag_valid = false;
    if (!mag_failed_) {
        double mag_period = 1.0 / noise_.mag_update_rate;
        if (current_time_ - last_mag_time_ >= mag_period) {
            Vec3 earth_mag(0.22, 0, 0.42);
            reading.magnetometer = addNoise(true_state.orientation.inverseRotate(earth_mag), noise_.mag_std);
            reading.mag_valid = true;
            last_mag_time_ = current_time_;
        }
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
        eskf_.updateGPSPosition(last_reading_.gps_position);
        eskf_.updateGPSVelocity(last_reading_.gps_velocity);
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
    double dx = true_pos.x - est.x;
    double dy = true_pos.y - est.y;
    double dz = true_pos.z - est.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

double StateEstimator::getVelocityError(const Vec3& true_vel) const noexcept {
    Vec3 est = eskf_.getVelocity();
    double dx = true_vel.x - est.x;
    double dy = true_vel.y - est.y;
    double dz = true_vel.z - est.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

double StateEstimator::getOrientationError(const Quaternion& true_ori) const noexcept {
    Quaternion est = eskf_.getOrientation();
    Quaternion diff = true_ori.inverse() * est;
    return 2.0 * std::acos(std::min(1.0, std::abs(diff.w)));
}

template bool ErrorStateKalmanFilter::applyUpdate<1>(
    const std::array<double, 1>&,
    const std::array<std::array<double, ERROR_DIM>, 1>&,
    std::array<double, 1>&,
    int,
    double
) noexcept;

template bool ErrorStateKalmanFilter::applyUpdate<3>(
    const std::array<double, 3>&,
    const std::array<std::array<double, ERROR_DIM>, 3>&,
    std::array<double, 3>&,
    int,
    double
) noexcept;
