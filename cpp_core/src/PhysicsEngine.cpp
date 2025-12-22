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

double NonlinearMotorModel::computeNonlinearResponse(double input, double gamma) noexcept {
    if (input <= 0) return 0;
    if (input >= 1) return 1;
    
    double x = input;
    double sign = 1.0;
    if (gamma < 1.0) {
        x = 1.0 - input;
        sign = -1.0;
    }
    
    double base = std::pow(std::abs(x), gamma);
    
    if (gamma < 1.0) {
        return 1.0 - base;
    }
    
    double blend = 0.5 + 0.5 * std::tanh(4.0 * (x - 0.5));
    double linearPart = x;
    double curvePart = base;
    
    return blend * curvePart + (1.0 - blend) * linearPart;
}

double NonlinearMotorModel::computeESCDelay(double voltage, double nominalVoltage, const ESCConfig& esc) noexcept {
    double voltageRatio = nominalVoltage / std::max(voltage, 0.1);
    double scaledDelay = esc.baseResponseTime * voltageRatio * esc.voltageScaleFactor;
    return std::clamp(scaledDelay, esc.minResponseTime, esc.maxResponseTime);
}

double NonlinearMotorModel::computeSoftCurrentLimit(double current, double maxCurrent, double softness) noexcept {
    if (current <= maxCurrent) return current;
    
    double excess = current - maxCurrent;
    double softExcess = maxCurrent * softness * std::tanh(excess / (maxCurrent * softness));
    
    return maxCurrent + softExcess;
}

double NonlinearMotorModel::computeThermalDerating(double temperature, double thermalCoeff) noexcept {
    constexpr double AMBIENT = 25.0;
    constexpr double MAX_TEMP = 120.0;
    
    if (temperature <= AMBIENT) return 1.0;
    if (temperature >= MAX_TEMP) return 0.5;
    
    double ratio = (temperature - AMBIENT) / (MAX_TEMP - AMBIENT);
    return 1.0 - thermalCoeff * ratio * ratio;
}

double NonlinearMotorModel::computeMotorRPM(
    double commandedRPM,
    double currentRPM,
    double voltage,
    double temperature,
    double dt,
    const MotorConfig& motor,
    const BatteryConfig& battery
) noexcept {
    double normalizedCommand = std::clamp(commandedRPM / 20000.0, 0.0, 1.0);
    double nonlinearCommand = computeNonlinearResponse(normalizedCommand, motor.esc.nonlinearGamma);
    double targetRPM = nonlinearCommand * 20000.0;
    
    double escDelay = computeESCDelay(voltage, battery.nominalVoltage, motor.esc);
    double thermalFactor = computeThermalDerating(temperature, motor.esc.thermalCoeff);
    
    double effectiveMaxCurrent = motor.maxCurrent * thermalFactor;
    
    double backEmf = currentRPM / motor.kv;
    double rawCurrent = (voltage - backEmf) / motor.resistance;
    double current = computeSoftCurrentLimit(rawCurrent, effectiveMaxCurrent, motor.esc.currentLimitSoftness);
    current = std::max(current, 0.0);
    
    double efficiency = motor.efficiency * (0.9 + 0.1 * (1.0 - std::abs(current) / effectiveMaxCurrent));
    double torque = motor.torqueConstant * current * efficiency;
    
    double friction = motor.frictionCoeff * currentRPM;
    double loadTorque = motor.frictionCoeff * 0.1 * (currentRPM / 1000.0) * (currentRPM / 1000.0);
    double netTorque = torque - friction - loadTorque;
    
    double angularAccel = netTorque / motor.inertia;
    double newRPM = currentRPM + angularAccel * dt * 30.0 / M_PI;
    
    double alpha = dt / (escDelay + dt);
    double filteredTarget = currentRPM + alpha * (targetRPM - currentRPM);
    
    double rpmError = filteredTarget - currentRPM;
    double maxRPMChange = 20000.0 * dt / escDelay;
    double clampedError = std::clamp(rpmError, -maxRPMChange, maxRPMChange);
    
    return std::clamp(currentRPM + clampedError, 0.0, 20000.0);
}

void NonlinearMotorModel::updateMotorState(
    MotorDynamicsState& state,
    const MotorCommand& command,
    double voltage,
    double dt,
    const MotorConfig& motor,
    const BatteryConfig& battery
) noexcept {
    for (int i = 0; i < 4; ++i) {
        double targetRPM = std::clamp(command.rpm[i], 0.0, 20000.0);
        
        state.rpm[i] = computeMotorRPM(
            targetRPM, state.rpm[i], voltage, state.temperature[i],
            dt, motor, battery
        );
        
        double backEmf = state.rpm[i] / motor.kv;
        double current = (voltage - backEmf) / motor.resistance;
        state.current[i] = std::max(0.0, current);
        
        double powerDissipated = state.current[i] * state.current[i] * motor.resistance;
        double heatGenerated = powerDissipated * dt / motor.thermalMass;
        double heatLost = (state.temperature[i] - 25.0) / motor.thermalResistance * dt / motor.thermalMass;
        state.temperature[i] += heatGenerated - heatLost;
        state.temperature[i] = std::clamp(state.temperature[i], 25.0, 150.0);
        
        state.escDelay[i] = computeESCDelay(voltage, battery.nominalVoltage, motor.esc);
    }
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
    simd::array_scale_simd(arr.data(), scale, result.data(), RigidBodyState::DIMENSION);
    return result;
}

std::array<double, RigidBodyState::DIMENSION> PhysicsEngine::arrayAdd(
    const std::array<double, RigidBodyState::DIMENSION>& a,
    const std::array<double, RigidBodyState::DIMENSION>& b
) noexcept {
    std::array<double, RigidBodyState::DIMENSION> result;
    simd::array_add_simd(a.data(), b.data(), result.data(), RigidBodyState::DIMENSION);
    return result;
}

double PhysicsEngine::computeError(
    const std::array<double, RigidBodyState::DIMENSION>& y,
    const std::array<double, RigidBodyState::DIMENSION>& yErr
) noexcept {
    return simd::array_max_error_simd(y.data(), yErr.data(), RigidBodyState::DIMENSION);
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
    BatteryConfig battery;
    return NonlinearMotorModel::computeMotorRPM(
        commandedRPM, currentRPM, voltage, 25.0, dt, motor, battery
    );
}

double PhysicsEngine::computeBatteryVoltage(
    const BatteryConfig& battery,
    double totalCurrent,
    double stateOfCharge
) noexcept {
    double socFactor = battery.socCurveAlpha + battery.socCurveBeta * stateOfCharge;
    double baseVoltage = battery.minVoltage + (battery.maxVoltage - battery.minVoltage) * socFactor * stateOfCharge;
    
    double voltageDrop = totalCurrent * battery.internalResistance;
    
    double dynamicResistance = battery.internalResistance * (1.0 + 0.5 * (1.0 - stateOfCharge));
    double dynamicDrop = totalCurrent * dynamicResistance;
    
    return std::max(battery.minVoltage, baseVoltage - dynamicDrop);
}

double BladeFlappingModel::computeAdvanceRatio(double forwardSpeed, double rpm, double rotorRadius) noexcept {
    if (rpm <= 0 || rotorRadius <= 0) return 0;
    double omega = rpm * M_PI / 30.0;
    double tipSpeed = omega * rotorRadius;
    return tipSpeed > 1e-6 ? forwardSpeed / tipSpeed : 0;
}

void BladeFlappingModel::computeFlappingCoeffs(
    double advanceRatio,
    double lockNumber,
    double collectivePitch,
    double& a1s,
    double& b1s
) noexcept {
    double mu = advanceRatio;
    double mu2 = mu * mu;
    double gamma = lockNumber;
    
    double theta0 = collectivePitch;
    double lambda = 0.05;
    
    double denom = 1.0 + 0.5 * mu2;
    double invDenom = 1.0 / denom;
    
    a1s = (2.0 * mu * (theta0 * (4.0/3.0) - lambda)) * invDenom / gamma;
    
    b1s = (4.0 * mu * theta0 / 3.0) * invDenom / (gamma * (1.0 + mu2));
    
    double satLimit = 0.15;
    a1s = std::clamp(a1s, -satLimit, satLimit);
    b1s = std::clamp(b1s, -satLimit, satLimit);
}

Vec3 BladeFlappingModel::computeFlappingMoment(
    const BladeFlappingState& state,
    const Vec3& velocityBody,
    const double* motorRPM,
    const RotorConfig& rotor,
    double armLength,
    double airDensity
) noexcept {
    double rollMoment = 0;
    double pitchMoment = 0;
    
    for (int i = 0; i < 4; ++i) {
        if (motorRPM[i] <= 0) continue;
        
        double thrust = BladeElementTheory::computeThrust(motorRPM[i], airDensity, rotor);
        double hubMomentCoeff = rotor.flapping.hingeOffset * rotor.radius;
        
        double Mx = thrust * hubMomentCoeff * state.b1s[i] * rotor.flapping.rollFlapCoupling;
        double My = thrust * hubMomentCoeff * state.a1s[i] * rotor.flapping.pitchFlapCoupling;
        
        rollMoment += Mx;
        pitchMoment += My;
    }
    
    return Vec3(rollMoment, pitchMoment, 0);
}

void BladeFlappingModel::updateFlappingState(
    BladeFlappingState& state,
    const Vec3& velocityBody,
    const Vec3& angularVelocity,
    const double* motorRPM,
    const RotorConfig& rotor,
    double dt
) noexcept {
    if (!rotor.flapping.enabled) return;
    
    double forwardSpeed = std::sqrt(velocityBody.x * velocityBody.x + velocityBody.y * velocityBody.y);
    
    for (int i = 0; i < 4; ++i) {
        if (motorRPM[i] <= 0) {
            state.a1s[i] = 0;
            state.b1s[i] = 0;
            continue;
        }
        
        double mu = computeAdvanceRatio(forwardSpeed, motorRPM[i], rotor.radius);
        
        if (mu < rotor.flapping.advanceRatioThreshold) {
            double decay = std::exp(-dt * rotor.flapping.flapFrequency);
            state.a1s[i] *= decay;
            state.b1s[i] *= decay;
            continue;
        }
        
        double a1s_new, b1s_new;
        computeFlappingCoeffs(mu, rotor.flapping.lockNumber, rotor.pitchAngle, a1s_new, b1s_new);
        
        double tau = 1.0 / rotor.flapping.flapFrequency;
        double alpha = dt / (tau + dt);
        state.a1s[i] += alpha * (a1s_new - state.a1s[i]);
        state.b1s[i] += alpha * (b1s_new - state.b1s[i]);
        
        state.flapAngle[i] = std::sqrt(state.a1s[i]*state.a1s[i] + state.b1s[i]*state.b1s[i]);
    }
    
    state.flappingMoment = computeFlappingMoment(
        state, velocityBody, motorRPM, rotor, 0.25, 1.225
    );
}

double VariableMassModel::computeFuelConsumption(
    const double* motorCurrent,
    const double* motorRPM,
    double dt,
    const FuelConfig& fuel
) noexcept {
    if (fuel.propulsionType == PropulsionType::ELECTRIC || !fuel.variableMassEnabled) {
        return 0;
    }
    
    double totalPower = 0;
    for (int i = 0; i < 4; ++i) {
        double powerW = motorCurrent[i] * 15.0;
        totalPower += powerW;
    }
    
    double fuelRate = fuel.specificFuelConsumption * totalPower;
    return fuelRate * dt;
}

void VariableMassModel::updateMassState(
    DynamicMassState& state,
    double baseMass,
    const FuelConfig& fuel,
    double fuelConsumed
) noexcept {
    state.currentFuel = std::max(0.0, fuel.currentFuelMass - fuelConsumed);
    state.currentMass = baseMass + state.currentFuel;
    
    double fuelRatio = fuel.initialFuelMass > 0 ? state.currentFuel / fuel.initialFuelMass : 0;
    state.centerOfGravityOffset = Vec3(
        fuel.tankCenterOfGravity[0] * (1.0 - fuelRatio),
        fuel.tankCenterOfGravity[1] * (1.0 - fuelRatio),
        fuel.tankCenterOfGravity[2] * (1.0 - fuelRatio)
    );
}

Mat3 VariableMassModel::computeInertiaWithFuel(
    const Mat3& baseInertia,
    double baseMass,
    double fuelMass,
    const double* tankCG
) noexcept {
    if (fuelMass <= 0) return baseInertia;
    
    double dx = tankCG[0];
    double dy = tankCG[1];
    double dz = tankCG[2];
    
    double Ixx_fuel = fuelMass * (dy*dy + dz*dz);
    double Iyy_fuel = fuelMass * (dx*dx + dz*dz);
    double Izz_fuel = fuelMass * (dx*dx + dy*dy);
    double Ixy_fuel = -fuelMass * dx * dy;
    double Ixz_fuel = -fuelMass * dx * dz;
    double Iyz_fuel = -fuelMass * dy * dz;
    
    return Mat3(
        baseInertia.data[0][0] + Ixx_fuel, baseInertia.data[0][1] + Ixy_fuel, baseInertia.data[0][2] + Ixz_fuel,
        baseInertia.data[1][0] + Ixy_fuel, baseInertia.data[1][1] + Iyy_fuel, baseInertia.data[1][2] + Iyz_fuel,
        baseInertia.data[2][0] + Ixz_fuel, baseInertia.data[2][1] + Iyz_fuel, baseInertia.data[2][2] + Izz_fuel
    );
}

Vec3 PhysicsEngine::computeAdvancedAeroDrag(
    const Vec3& velocityBody,
    const Vec3& angularVelocity,
    const double* motorRPM,
    const AeroConfig& aero,
    const RotorConfig& rotor,
    const BladeFlappingState& flapping
) noexcept {
    double speed = velocityBody.norm();
    if (speed < 1e-9) return Vec3();
    
    double q = 0.5 * aero.airDensity * speed * speed;
    double parasiticDrag = q * aero.parasiticDragArea;
    Vec3 dragDir = velocityBody.normalized() * (-1.0);
    Vec3 linearDrag = dragDir * parasiticDrag;
    
    double avgRPM = 0;
    for (int i = 0; i < 4; ++i) avgRPM += motorRPM[i];
    avgRPM *= 0.25;
    
    double mu = BladeFlappingModel::computeAdvanceRatio(speed, avgRPM, rotor.radius);
    double inducedDragFactor = 1.0 + aero.advanceRatioDragScale * mu * mu;
    
    double omega = avgRPM * M_PI / 30.0;
    double tipSpeed = omega * rotor.radius;
    double diskArea = M_PI * rotor.radius * rotor.radius * 4.0;
    double hForce = 0;
    if (tipSpeed > 1e-6) {
        double thrustCoeff = 0.01;
        hForce = aero.airDensity * diskArea * tipSpeed * tipSpeed * thrustCoeff * mu * inducedDragFactor;
    }
    Vec3 hubDrag = Vec3(-hForce * (velocityBody.x > 0 ? 1 : -1), 
                        -hForce * (velocityBody.y > 0 ? 1 : -1) * 0.5, 0);
    
    double omegaNorm = angularVelocity.norm();
    Vec3 rotationalDrag = angularVelocity * (-aero.rotationalDragCoeff * omegaNorm);
    
    Vec3 flappingDrag = flapping.flappingMoment * (-0.01);
    
    return linearDrag + hubDrag + rotationalDrag + flappingDrag;
}

