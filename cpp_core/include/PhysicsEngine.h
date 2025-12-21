#pragma once
#include "Types.h"
#include <functional>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>

enum class IntegrationMethod {
    EULER,
    SEMI_IMPLICIT_EULER,
    RK4,
    RK45_ADAPTIVE,
    VELOCITY_VERLET
};

enum class MotorConfiguration {
    PLUS,
    X
};

enum class RotorDirection {
    CW,
    CCW
};

struct RotorConfig {
    double radius;
    double chord;
    double pitchAngle;
    double liftSlope;
    double dragCoeff;
    double inflowRatio;
    RotorDirection direction;
    
    constexpr RotorConfig() noexcept
        : radius(0.127), chord(0.02), pitchAngle(0.26)
        , liftSlope(0.6), dragCoeff(0.01), inflowRatio(0.05)
        , direction(RotorDirection::CCW) {}
};

struct MotorConfig {
    double kv;
    double resistance;
    double inductance;
    double torqueConstant;
    double frictionCoeff;
    double inertia;
    double maxCurrent;
    double efficiency;
    
    constexpr MotorConfig() noexcept
        : kv(2300), resistance(0.1), inductance(0.00001)
        , torqueConstant(0.0043), frictionCoeff(0.00001)
        , inertia(0.00001), maxCurrent(30), efficiency(0.85) {}
};

struct BatteryConfig {
    double nominalVoltage;
    double maxVoltage;
    double minVoltage;
    double capacity;
    double internalResistance;
    double dischargeCurrent;
    
    constexpr BatteryConfig() noexcept
        : nominalVoltage(14.8), maxVoltage(16.8), minVoltage(13.2)
        , capacity(1500), internalResistance(0.02), dischargeCurrent(0) {}
};

struct AeroConfig {
    double airDensity;
    double groundEffectCoeff;
    double groundEffectHeight;
    double parasiticDragArea;
    double inducedDragFactor;
    
    constexpr AeroConfig() noexcept
        : airDensity(1.225), groundEffectCoeff(0.5)
        , groundEffectHeight(0.5), parasiticDragArea(0.01)
        , inducedDragFactor(1.1) {}
};

struct RigidBodyState {
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 angularVelocity;
    
    static constexpr size_t DIMENSION = 13;
    
    std::array<double, DIMENSION> toArray() const noexcept {
        return {
            position.x, position.y, position.z,
            velocity.x, velocity.y, velocity.z,
            orientation.w, orientation.x, orientation.y, orientation.z,
            angularVelocity.x, angularVelocity.y, angularVelocity.z
        };
    }
    
    static RigidBodyState fromArray(const std::array<double, DIMENSION>& arr) noexcept {
        RigidBodyState s;
        s.position = Vec3(arr[0], arr[1], arr[2]);
        s.velocity = Vec3(arr[3], arr[4], arr[5]);
        s.orientation = Quaternion(arr[6], arr[7], arr[8], arr[9]).normalized();
        s.angularVelocity = Vec3(arr[10], arr[11], arr[12]);
        return s;
    }
    
    RigidBodyState operator+(const RigidBodyState& other) const noexcept {
        RigidBodyState r;
        r.position = position + other.position;
        r.velocity = velocity + other.velocity;
        r.orientation = (orientation + other.orientation).normalized();
        r.angularVelocity = angularVelocity + other.angularVelocity;
        return r;
    }
    
    RigidBodyState operator*(double s) const noexcept {
        RigidBodyState r;
        r.position = position * s;
        r.velocity = velocity * s;
        r.orientation = orientation * s;
        r.angularVelocity = angularVelocity * s;
        return r;
    }
};

struct RigidBodyDerivative {
    Vec3 velocity;
    Vec3 acceleration;
    Quaternion orientationDot;
    Vec3 angularAcceleration;
    
    static constexpr size_t DIMENSION = 13;
    
    std::array<double, DIMENSION> toArray() const noexcept {
        return {
            velocity.x, velocity.y, velocity.z,
            acceleration.x, acceleration.y, acceleration.z,
            orientationDot.w, orientationDot.x, orientationDot.y, orientationDot.z,
            angularAcceleration.x, angularAcceleration.y, angularAcceleration.z
        };
    }
    
    RigidBodyDerivative operator+(const RigidBodyDerivative& other) const noexcept {
        RigidBodyDerivative r;
        r.velocity = velocity + other.velocity;
        r.acceleration = acceleration + other.acceleration;
        r.orientationDot = orientationDot + other.orientationDot;
        r.angularAcceleration = angularAcceleration + other.angularAcceleration;
        return r;
    }
    
    RigidBodyDerivative operator*(double s) const noexcept {
        RigidBodyDerivative r;
        r.velocity = velocity * s;
        r.acceleration = acceleration * s;
        r.orientationDot = orientationDot * s;
        r.angularAcceleration = angularAcceleration * s;
        return r;
    }
};

struct AdaptiveStepResult {
    RigidBodyState state;
    double actualDt;
    double errorEstimate;
    bool accepted;
};

class DrydenWindModel {
public:
    DrydenWindModel() noexcept;
    void setAltitude(double altitude) noexcept;
    void setWindSpeed(double speed) noexcept;
    Vec3 update(double dt) noexcept;
    
private:
    double altitude_;
    double windSpeed_;
    double Lu_, Lv_, Lw_;
    double sigmaU_, sigmaV_, sigmaW_;
    Vec3 state_;
    std::mt19937 rng_;
    std::normal_distribution<double> dist_;
    
    void updateScales() noexcept;
};

class GroundEffect {
public:
    static double getCoefficient(double altitude, double rotorRadius, double baseCoeff) noexcept;
    static Vec3 computeForce(const Vec3& thrustBody, double altitude, const AeroConfig& config, double rotorRadius) noexcept;
};

class BladeElementTheory {
public:
    static double computeThrust(double rpm, double airDensity, const RotorConfig& rotor) noexcept;
    static double computeTorque(double rpm, double airDensity, const RotorConfig& rotor) noexcept;
    static double computePower(double rpm, double airDensity, const RotorConfig& rotor) noexcept;
    static double computeInflow(double rpm, double climbRate, double airDensity, const RotorConfig& rotor) noexcept;
};

class IMUSimulator {
public:
    IMUSimulator() noexcept;
    
    void setAccelNoise(double stddev) noexcept;
    void setGyroNoise(double stddev) noexcept;
    void setAccelBias(const Vec3& bias) noexcept;
    void setGyroBias(const Vec3& bias) noexcept;
    
    IMUReading simulate(const State& state, const Vec3& acceleration, double dt) noexcept;
    
private:
    Vec3 accelBias_, gyroBias_;
    double accelNoise_, gyroNoise_;
    std::mt19937 rng_;
    std::normal_distribution<double> dist_;
    
    Vec3 addNoise(const Vec3& v, double stddev) noexcept;
};

class PhysicsEngine {
public:
    using DerivativeFunc = std::function<RigidBodyDerivative(const RigidBodyState&, double)>;
    
    explicit PhysicsEngine(IntegrationMethod method = IntegrationMethod::RK4) noexcept;
    
    void setIntegrationMethod(IntegrationMethod method) noexcept;
    IntegrationMethod getIntegrationMethod() const noexcept;
    
    void setAdaptiveTolerance(double absTol, double relTol) noexcept;
    void setMinMaxStep(double minDt, double maxDt) noexcept;
    
    RigidBodyState integrate(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t = 0.0
    ) const;
    
    AdaptiveStepResult integrateAdaptive(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t = 0.0
    ) const;
    
    static double computeThrust(
        const double* motorRPM,
        const RotorConfig& rotor,
        const AeroConfig& aero
    ) noexcept;
    
    static Vec3 computeTorques(
        const double* motorRPM,
        const RotorConfig& rotor,
        double armLength,
        double dragCoeff,
        MotorConfiguration config
    ) noexcept;
    
    static Vec3 computeAerodynamicDrag(
        const Vec3& velocity,
        const Vec3& angularVelocity,
        const AeroConfig& config
    ) noexcept;
    
    static Mat3 computeInertiaFromMass(double mass, double armLength) noexcept;
    static Quaternion integrateQuaternion(const Quaternion& q, const Vec3& omega, double dt) noexcept;
    
    static double computeMotorRPM(
        double commandedRPM,
        double currentRPM,
        double voltage,
        double dt,
        const MotorConfig& motor
    ) noexcept;
    
    static double computeBatteryVoltage(
        const BatteryConfig& battery,
        double totalCurrent
    ) noexcept;

private:
    IntegrationMethod method_;
    double absTol_, relTol_;
    double minDt_, maxDt_;
    
    RigidBodyState integrateEuler(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t
    ) const;
    
    RigidBodyState integrateSemiImplicitEuler(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t
    ) const;
    
    RigidBodyState integrateRK4(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t
    ) const;
    
    AdaptiveStepResult integrateRK45(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t
    ) const;
    
    RigidBodyState integrateVelocityVerlet(
        const RigidBodyState& state,
        const DerivativeFunc& derivative,
        double dt,
        double t
    ) const;
    
    static RigidBodyState stateAdd(
        const RigidBodyState& s,
        const std::array<double, RigidBodyState::DIMENSION>& delta
    ) noexcept;
    
    static std::array<double, RigidBodyState::DIMENSION> arrayScale(
        const std::array<double, RigidBodyState::DIMENSION>& arr,
        double scale
    ) noexcept;
    
    static std::array<double, RigidBodyState::DIMENSION> arrayAdd(
        const std::array<double, RigidBodyState::DIMENSION>& a,
        const std::array<double, RigidBodyState::DIMENSION>& b
    ) noexcept;
    
    static double computeError(
        const std::array<double, RigidBodyState::DIMENSION>& y4,
        const std::array<double, RigidBodyState::DIMENSION>& y5
    ) noexcept;
};
