#include "PhysicsEngine.h"
#include <algorithm>
#include <cstring>

DrydenWindModel::DrydenWindModel() noexcept
    : altitude_(100), windSpeed_(5), Lu_(200), Lv_(200), Lw_(50)
    , sigmaU_(2.12), sigmaV_(2.12), sigmaW_(1.4), state_()
    , rng_(std::random_device{}()), dist_(0.0, 1.0) {
    updateScales();
}

void DrydenWindModel::setAltitude(double altitude) noexcept {
    altitude_ = std::max(1.0, altitude);
    updateScales();
}

void DrydenWindModel::setWindSpeed(double speed) noexcept {
    windSpeed_ = std::max(0.0, speed);
    updateScales();
}

void DrydenWindModel::updateScales() noexcept {
    double h = std::max(altitude_, 3.0);
    Lu_ = h / std::pow(0.177 + 0.000823 * h, 1.2);
    Lv_ = Lu_;
    Lw_ = h;
    double w20 = windSpeed_;
    sigmaU_ = w20 / std::pow(0.177 + 0.000823 * h, 0.4);
    sigmaV_ = sigmaU_;
    sigmaW_ = 0.1 * w20;
}

Vec3 DrydenWindModel::update(double dt) noexcept {
    double V = std::max(windSpeed_, 1.0);
    
    double au = V / Lu_;
    double av = V / Lv_;
    double aw = V / Lw_;
    
    double expU = std::exp(-au * dt);
    double expV = std::exp(-av * dt);
    double expW = std::exp(-aw * dt);
    
    double nu = dist_(rng_) * sigmaU_ * std::sqrt(2.0 * au * dt);
    double nv = dist_(rng_) * sigmaV_ * std::sqrt(2.0 * av * dt);
    double nw = dist_(rng_) * sigmaW_ * std::sqrt(2.0 * aw * dt);
    
    state_.x = expU * state_.x + nu;
    state_.y = expV * state_.y + nv;
    state_.z = expW * state_.z + nw;
    
    return state_;
}

double GroundEffect::getCoefficient(double altitude, double rotorRadius, double baseCoeff) noexcept {
    if (altitude <= 0) return 1.0 + baseCoeff;
    double ratio = altitude / rotorRadius;
    if (ratio > 4.0) return 1.0;
    return 1.0 + baseCoeff / (ratio * ratio + 1.0);
}

Vec3 GroundEffect::computeForce(const Vec3& thrustBody, double altitude, const AeroConfig& config, double rotorRadius) noexcept {
    double coeff = getCoefficient(altitude, rotorRadius, config.groundEffectCoeff);
    return thrustBody * coeff;
}

double BladeElementTheory::computeThrust(double rpm, double airDensity, const RotorConfig& rotor) noexcept {
    if (rpm <= 0) return 0;
    double omega = rpm * M_PI / 30.0;
    double R = rotor.radius;
    double c = rotor.chord;
    double theta = rotor.pitchAngle;
    double a = rotor.liftSlope;
    double lambda = rotor.inflowRatio;
    
    double solidity = 4.0 * c / (M_PI * R);
    double Ct = solidity * a * (theta / 3.0 - lambda / 2.0) / 2.0;
    
    return Ct * airDensity * M_PI * R * R * (omega * R) * (omega * R);
}

double BladeElementTheory::computeTorque(double rpm, double airDensity, const RotorConfig& rotor) noexcept {
    if (rpm <= 0) return 0;
    double omega = rpm * M_PI / 30.0;
    double R = rotor.radius;
    double c = rotor.chord;
    double theta = rotor.pitchAngle;
    double a = rotor.liftSlope;
    double Cd0 = rotor.dragCoeff;
    double lambda = rotor.inflowRatio;
    
    double solidity = 4.0 * c / (M_PI * R);
    double Cq = solidity * (Cd0 / 8.0 + a * lambda * (theta / 3.0 - lambda / 2.0) / 2.0);
    
    return Cq * airDensity * M_PI * R * R * R * (omega * R) * (omega * R);
}

double BladeElementTheory::computePower(double rpm, double airDensity, const RotorConfig& rotor) noexcept {
    double omega = rpm * M_PI / 30.0;
    return computeTorque(rpm, airDensity, rotor) * omega;
}

double BladeElementTheory::computeInflow(double rpm, double climbRate, double airDensity, const RotorConfig& rotor) noexcept {
    if (rpm <= 0) return 0;
    double omega = rpm * M_PI / 30.0;
    double R = rotor.radius;
    double Vclimb = climbRate;
    
    double thrust = computeThrust(rpm, airDensity, rotor);
    double vi0 = std::sqrt(thrust / (2.0 * airDensity * M_PI * R * R));
    
    double Vc = Vclimb / vi0;
    double vi = vi0 * (-Vc / 2.0 + std::sqrt(Vc * Vc / 4.0 + 1.0));
    
    return vi / (omega * R);
}

IMUSimulator::IMUSimulator() noexcept
    : accelBias_(), gyroBias_()
    , accelNoise_(0.01), gyroNoise_(0.001)
    , rng_(std::random_device{}()), dist_(0.0, 1.0) {}

void IMUSimulator::setAccelNoise(double stddev) noexcept { accelNoise_ = stddev; }
void IMUSimulator::setGyroNoise(double stddev) noexcept { gyroNoise_ = stddev; }
void IMUSimulator::setAccelBias(const Vec3& bias) noexcept { accelBias_ = bias; }
void IMUSimulator::setGyroBias(const Vec3& bias) noexcept { gyroBias_ = bias; }

Vec3 IMUSimulator::addNoise(const Vec3& v, double stddev) noexcept {
    return Vec3(
        v.x + dist_(rng_) * stddev,
        v.y + dist_(rng_) * stddev,
        v.z + dist_(rng_) * stddev
    );
}

IMUReading IMUSimulator::simulate(const State& state, const Vec3& acceleration, double dt) noexcept {
    IMUReading reading;
    
    Vec3 gravity(0, 0, 9.81);
    Vec3 specificForce = state.orientation.inverseRotate(acceleration + gravity);
    reading.accelerometer = addNoise(specificForce + accelBias_, accelNoise_);
    
    reading.gyroscope = addNoise(state.angularVelocity + gyroBias_, gyroNoise_);
    
    Vec3 earthMag(0.22, 0, 0.42);
    reading.magnetometer = state.orientation.inverseRotate(earthMag);
    
    reading.barometer = 101325.0 * std::exp(-state.position.z / 8500.0);
    reading.timestamp = state.time;
    
    return reading;
}

PhysicsEngine::PhysicsEngine(IntegrationMethod method) noexcept
    : method_(method), absTol_(1e-6), relTol_(1e-6), minDt_(1e-6), maxDt_(0.1) {}

void PhysicsEngine::setIntegrationMethod(IntegrationMethod method) noexcept {
    method_ = method;
}

IntegrationMethod PhysicsEngine::getIntegrationMethod() const noexcept {
    return method_;
}

void PhysicsEngine::setAdaptiveTolerance(double absTol, double relTol) noexcept {
    absTol_ = absTol;
    relTol_ = relTol;
}

void PhysicsEngine::setMinMaxStep(double minDt, double maxDt) noexcept {
    minDt_ = minDt;
    maxDt_ = maxDt;
}

RigidBodyState PhysicsEngine::integrate(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    switch (method_) {
        case IntegrationMethod::EULER:
            return integrateEuler(state, derivative, dt, t);
        case IntegrationMethod::SEMI_IMPLICIT_EULER:
            return integrateSemiImplicitEuler(state, derivative, dt, t);
        case IntegrationMethod::RK45_ADAPTIVE:
            return integrateRK45(state, derivative, dt, t).state;
        case IntegrationMethod::VELOCITY_VERLET:
            return integrateVelocityVerlet(state, derivative, dt, t);
        case IntegrationMethod::RK4:
        default:
            return integrateRK4(state, derivative, dt, t);
    }
}

AdaptiveStepResult PhysicsEngine::integrateAdaptive(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    return integrateRK45(state, derivative, dt, t);
}

RigidBodyState PhysicsEngine::integrateEuler(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    RigidBodyDerivative d = derivative(state, t);
    
    RigidBodyState next;
    next.position = state.position + d.velocity * dt;
    next.velocity = state.velocity + d.acceleration * dt;
    next.orientation = integrateQuaternion(state.orientation, state.angularVelocity, dt);
    next.angularVelocity = state.angularVelocity + d.angularAcceleration * dt;
    
    return next;
}

RigidBodyState PhysicsEngine::integrateSemiImplicitEuler(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    RigidBodyDerivative d = derivative(state, t);
    
    RigidBodyState next;
    next.velocity = state.velocity + d.acceleration * dt;
    next.position = state.position + next.velocity * dt;
    next.angularVelocity = state.angularVelocity + d.angularAcceleration * dt;
    next.orientation = integrateQuaternion(state.orientation, next.angularVelocity, dt);
    
    return next;
}

RigidBodyState PhysicsEngine::integrateRK4(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    auto k1 = derivative(state, t).toArray();
    
    auto state2 = stateAdd(state, arrayScale(k1, dt * 0.5));
    auto k2 = derivative(state2, t + dt * 0.5).toArray();
    
    auto state3 = stateAdd(state, arrayScale(k2, dt * 0.5));
    auto k3 = derivative(state3, t + dt * 0.5).toArray();
    
    auto state4 = stateAdd(state, arrayScale(k3, dt));
    auto k4 = derivative(state4, t + dt).toArray();
    
    std::array<double, RigidBodyState::DIMENSION> weighted;
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i) {
        weighted[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    
    RigidBodyState next = stateAdd(state, arrayScale(weighted, dt));
    next.orientation = next.orientation.normalized();
    
    return next;
}

AdaptiveStepResult PhysicsEngine::integrateRK45(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    constexpr double a21 = 1.0/5.0;
    constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
    constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
    constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
    
    constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
    constexpr double e1 = 71.0/57600.0, e3 = -71.0/16695.0, e4 = 71.0/1920.0, e5 = -17253.0/339200.0, e6 = 22.0/525.0, e7 = -1.0/40.0;
    
    auto k1 = derivative(state, t).toArray();
    
    auto s2 = stateAdd(state, arrayScale(k1, dt * a21));
    auto k2 = derivative(s2, t + dt * a21).toArray();
    
    std::array<double, RigidBodyState::DIMENSION> tmp;
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i)
        tmp[i] = a31 * k1[i] + a32 * k2[i];
    auto s3 = stateAdd(state, arrayScale(tmp, dt));
    auto k3 = derivative(s3, t + dt * 0.3).toArray();
    
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i)
        tmp[i] = a41 * k1[i] + a42 * k2[i] + a43 * k3[i];
    auto s4 = stateAdd(state, arrayScale(tmp, dt));
    auto k4 = derivative(s4, t + dt * 0.8).toArray();
    
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i)
        tmp[i] = a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i];
    auto s5 = stateAdd(state, arrayScale(tmp, dt));
    auto k5 = derivative(s5, t + dt * 8.0/9.0).toArray();
    
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i)
        tmp[i] = a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i];
    auto s6 = stateAdd(state, arrayScale(tmp, dt));
    auto k6 = derivative(s6, t + dt).toArray();
    
    std::array<double, RigidBodyState::DIMENSION> y5, yErr;
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i) {
        y5[i] = state.toArray()[i] + dt * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
    }
    
    auto s7 = RigidBodyState::fromArray(y5);
    auto k7 = derivative(s7, t + dt).toArray();
    
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i) {
        yErr[i] = dt * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
    }
    
    double err = computeError(y5, yErr);
    
    AdaptiveStepResult result;
    result.state = RigidBodyState::fromArray(y5);
    result.state.orientation = result.state.orientation.normalized();
    result.errorEstimate = err;
    result.accepted = err < absTol_ + relTol_ * std::max(1.0, result.state.position.norm());
    
    double factor = 0.9 * std::pow(absTol_ / (err + 1e-12), 0.2);
    factor = std::clamp(factor, 0.1, 5.0);
    result.actualDt = std::clamp(dt * factor, minDt_, maxDt_);
    
    return result;
}

RigidBodyState PhysicsEngine::integrateVelocityVerlet(
    const RigidBodyState& state,
    const DerivativeFunc& derivative,
    double dt,
    double t
) const {
    RigidBodyDerivative d0 = derivative(state, t);
    
    RigidBodyState next;
    next.position = state.position + state.velocity * dt + d0.acceleration * (0.5 * dt * dt);
    next.orientation = integrateQuaternion(state.orientation, state.angularVelocity, dt);
    next.angularVelocity = state.angularVelocity;
    next.velocity = state.velocity;
    
    RigidBodyDerivative d1 = derivative(next, t + dt);
    
    next.velocity = state.velocity + (d0.acceleration + d1.acceleration) * (0.5 * dt);
    next.angularVelocity = state.angularVelocity + (d0.angularAcceleration + d1.angularAcceleration) * (0.5 * dt);
    
    return next;
}

RigidBodyState PhysicsEngine::stateAdd(
    const RigidBodyState& s,
    const std::array<double, RigidBodyState::DIMENSION>& delta
) noexcept {
    RigidBodyState result;
    result.position = Vec3(s.position.x + delta[0], s.position.y + delta[1], s.position.z + delta[2]);
    result.velocity = Vec3(s.velocity.x + delta[3], s.velocity.y + delta[4], s.velocity.z + delta[5]);
    result.orientation = Quaternion(
        s.orientation.w + delta[6], s.orientation.x + delta[7],
        s.orientation.y + delta[8], s.orientation.z + delta[9]
    ).normalized();
    result.angularVelocity = Vec3(s.angularVelocity.x + delta[10], s.angularVelocity.y + delta[11], s.angularVelocity.z + delta[12]);
    return result;
}

std::array<double, RigidBodyState::DIMENSION> PhysicsEngine::arrayScale(
    const std::array<double, RigidBodyState::DIMENSION>& arr,
    double scale
) noexcept {
    std::array<double, RigidBodyState::DIMENSION> result;
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i) {
        result[i] = arr[i] * scale;
    }
    return result;
}

std::array<double, RigidBodyState::DIMENSION> PhysicsEngine::arrayAdd(
    const std::array<double, RigidBodyState::DIMENSION>& a,
    const std::array<double, RigidBodyState::DIMENSION>& b
) noexcept {
    std::array<double, RigidBodyState::DIMENSION> result;
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

double PhysicsEngine::computeError(
    const std::array<double, RigidBodyState::DIMENSION>& y,
    const std::array<double, RigidBodyState::DIMENSION>& yErr
) noexcept {
    double maxErr = 0;
    for (size_t i = 0; i < RigidBodyState::DIMENSION; ++i) {
        double scale = std::max(std::abs(y[i]), 1.0);
        maxErr = std::max(maxErr, std::abs(yErr[i]) / scale);
    }
    return maxErr;
}

double PhysicsEngine::computeThrust(
    const double* motorRPM,
    const RotorConfig& rotor,
    const AeroConfig& aero
) noexcept {
    double total = 0;
    for (int i = 0; i < 4; ++i) {
        total += BladeElementTheory::computeThrust(motorRPM[i], aero.airDensity, rotor);
    }
    return total;
}

Vec3 PhysicsEngine::computeTorques(
    const double* motorRPM,
    const RotorConfig& rotor,
    double armLength,
    double dragCoeff,
    MotorConfiguration config
) noexcept {
    double forces[4], moments[4];
    
    for (int i = 0; i < 4; ++i) {
        forces[i] = BladeElementTheory::computeThrust(motorRPM[i], 1.225, rotor);
        moments[i] = BladeElementTheory::computeTorque(motorRPM[i], 1.225, rotor);
    }
    
    double rollTorque, pitchTorque;
    
    if (config == MotorConfiguration::PLUS) {
        rollTorque = armLength * (forces[1] - forces[3]);
        pitchTorque = armLength * (forces[0] - forces[2]);
    } else {
        constexpr double INV_SQRT2 = 0.7071067811865476;
        double L = armLength * INV_SQRT2;
        rollTorque = L * (forces[1] + forces[2] - forces[0] - forces[3]);
        pitchTorque = L * (forces[0] + forces[1] - forces[2] - forces[3]);
    }
    
    double yawTorque = moments[0] - moments[1] + moments[2] - moments[3];
    
    return Vec3(rollTorque, pitchTorque, yawTorque);
}

Vec3 PhysicsEngine::computeAerodynamicDrag(
    const Vec3& velocity,
    const Vec3& angularVelocity,
    const AeroConfig& config
) noexcept {
    double speed = velocity.norm();
    if (speed < 1e-9) return Vec3();
    
    double dragMagnitude = 0.5 * config.airDensity * config.parasiticDragArea * speed * speed;
    Vec3 linearDrag = velocity.normalized() * (-dragMagnitude);
    
    double omegaNorm = angularVelocity.norm();
    Vec3 rotationalDrag = angularVelocity * (-0.001 * omegaNorm);
    
    return linearDrag;
}

Mat3 PhysicsEngine::computeInertiaFromMass(double mass, double armLength) noexcept {
    double Ixx = 0.5 * mass * armLength * armLength;
    double Iyy = Ixx;
    double Izz = mass * armLength * armLength;
    return Mat3::diagonal(Ixx, Iyy, Izz);
}

Quaternion PhysicsEngine::integrateQuaternion(const Quaternion& q, const Vec3& omega, double dt) noexcept {
    double angle = omega.norm() * dt;
    if (angle < 1e-12) return q;
    
    Vec3 axis = omega.normalized();
    double halfAngle = angle * 0.5;
    double s = std::sin(halfAngle);
    Quaternion deltaQ(std::cos(halfAngle), axis.x * s, axis.y * s, axis.z * s);
    return (deltaQ * q).normalized();
}

double PhysicsEngine::computeMotorRPM(
    double commandedRPM,
    double currentRPM,
    double voltage,
    double dt,
    const MotorConfig& motor
) noexcept {
    double backEmf = currentRPM / motor.kv;
    double current = (voltage - backEmf) / motor.resistance;
    current = std::clamp(current, -motor.maxCurrent, motor.maxCurrent);
    
    double torque = motor.torqueConstant * current * motor.efficiency;
    double friction = motor.frictionCoeff * currentRPM;
    double netTorque = torque - friction;
    
    double angularAccel = netTorque / motor.inertia;
    double newRPM = currentRPM + angularAccel * dt * 30.0 / M_PI;
    
    double alpha = dt / (0.02 + dt);
    return currentRPM + alpha * (std::clamp(newRPM, 0.0, commandedRPM) - currentRPM);
}

double PhysicsEngine::computeBatteryVoltage(
    const BatteryConfig& battery,
    double totalCurrent
) noexcept {
    double voltageDrop = totalCurrent * battery.internalResistance;
    return std::max(battery.minVoltage, battery.nominalVoltage - voltageDrop);
}
