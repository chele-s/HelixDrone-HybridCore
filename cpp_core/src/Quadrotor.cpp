#include "Quadrotor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

Quadrotor::Quadrotor() noexcept
    : config_(), motorState_(), motorDynamics_(), flappingState_(), massState_(), imuReading_()
    , physics_(IntegrationMethod::RK4), windModel_(), imuSim_()
    , simulationTime_(0), totalFuelConsumed_(0), lastSubStepCount_(0), lastForce_(), lastTorque_(), windVelocity_()
    , integrating_(false) {
    initialize();
}

Quadrotor::Quadrotor(const QuadrotorConfig& config) noexcept
    : config_(config), motorState_(), motorDynamics_(), flappingState_(), massState_(), imuReading_()
    , physics_(config.integrationMethod), windModel_(), imuSim_()
    , simulationTime_(0), totalFuelConsumed_(0), lastSubStepCount_(0), lastForce_(), lastTorque_(), windVelocity_()
    , integrating_(false) {
    initialize();
}

void Quadrotor::initialize() noexcept {
    currentInertia_ = config_.inertiaMatrix;
    cachedInertiaInverse_ = currentInertia_.inverse();
    massState_.currentMass = config_.mass + config_.fuel.currentFuelMass;
    massState_.currentFuel = config_.fuel.currentFuelMass;
    reset();
}

void Quadrotor::reset() noexcept {
    state_ = State();
    state_.position = Vec3(0, 0, 0);
    state_.velocity = Vec3(0, 0, 0);
    state_.orientation = Quaternion();
    state_.angularVelocity = Vec3(0, 0, 0);
    state_.batteryVoltage = config_.battery.maxVoltage;
    state_.time = 0;
    std::memset(state_.motorRPM, 0, sizeof(state_.motorRPM));
    
    motorState_ = MotorState();
    motorDynamics_ = MotorDynamicsState();
    flappingState_ = BladeFlappingState();
    massState_.currentMass = config_.mass + config_.fuel.currentFuelMass;
    massState_.currentFuel = config_.fuel.currentFuelMass;
    totalFuelConsumed_ = 0;
    lastSubStepCount_ = 0;
    simulationTime_ = 0;
    lastForce_ = Vec3();
    lastTorque_ = Vec3();
    windVelocity_ = Vec3();
    
    updateInertia();
}

State Quadrotor::getState() const noexcept {
    return state_;
}

MotorState Quadrotor::getMotorState() const noexcept {
    return motorState_;
}

IMUReading Quadrotor::getIMUReading() const noexcept {
    return imuReading_;
}

const QuadrotorConfig& Quadrotor::getConfig() const noexcept {
    return config_;
}

Vec3 Quadrotor::getForces() const noexcept {
    return lastForce_;
}

Vec3 Quadrotor::getTorques() const noexcept {
    return lastTorque_;
}

double Quadrotor::getSimulationTime() const noexcept {
    return simulationTime_;
}

double Quadrotor::getCurrentMass() const noexcept {
    return massState_.currentMass;
}

double Quadrotor::getCurrentFuel() const noexcept {
    return massState_.currentFuel;
}

int Quadrotor::getSubStepCount() const noexcept {
    return lastSubStepCount_;
}

bool Quadrotor::isIntegrating() const noexcept {
    return integrating_.load(std::memory_order_acquire);
}

void Quadrotor::checkNotIntegrating() const {
    if (integrating_.load(std::memory_order_acquire)) {
        throw std::runtime_error("Cannot modify state during physics integration");
    }
}

void Quadrotor::updateInertia() noexcept {
    if (config_.enableVariableMass && config_.fuel.variableMassEnabled) {
        currentInertia_ = VariableMassModel::computeInertiaWithFuel(
            config_.inertiaMatrix, config_.mass, massState_.currentFuel, config_.fuel.tankCenterOfGravity
        );
    } else {
        currentInertia_ = config_.inertiaMatrix;
    }
    cachedInertiaInverse_ = currentInertia_.inverse();
}

void Quadrotor::setState(const State& state) {
    checkNotIntegrating();
    state_ = state;
    for (int i = 0; i < 4; ++i) {
        motorState_.rpm[i] = state.motorRPM[i];
        motorDynamics_.rpm[i] = state.motorRPM[i];
    }
}

void Quadrotor::setPosition(const Vec3& pos) {
    checkNotIntegrating();
    state_.position = pos;
}

void Quadrotor::setVelocity(const Vec3& vel) {
    checkNotIntegrating();
    state_.velocity = vel;
}

void Quadrotor::setOrientation(const Quaternion& q) {
    checkNotIntegrating();
    state_.orientation = q.normalized();
}

void Quadrotor::setAngularVelocity(const Vec3& omega) {
    checkNotIntegrating();
    state_.angularVelocity = omega;
}

void Quadrotor::setMotorConfiguration(MotorConfiguration config) noexcept {
    config_.motorConfig = config;
}

void Quadrotor::setIntegrationMethod(IntegrationMethod method) noexcept {
    config_.integrationMethod = method;
    physics_.setIntegrationMethod(method);
}

void Quadrotor::setSubStepConfig(const SubStepConfig& config) noexcept {
    config_.subStep = config;
}

void Quadrotor::setWind(const Vec3& meanWind) noexcept {
    windModel_.setWindSpeed(meanWind.norm());
}

void Quadrotor::enableFeature(const char* feature, bool enable) noexcept {
    if (std::strcmp(feature, "ground_effect") == 0) config_.enableGroundEffect = enable;
    else if (std::strcmp(feature, "wind") == 0) config_.enableWindDisturbance = enable;
    else if (std::strcmp(feature, "motor_dynamics") == 0) config_.enableMotorDynamics = enable;
    else if (std::strcmp(feature, "battery") == 0) config_.enableBatteryDynamics = enable;
    else if (std::strcmp(feature, "imu") == 0) config_.enableIMU = enable;
    else if (std::strcmp(feature, "nonlinear_motor") == 0) config_.enableNonlinearMotor = enable;
    else if (std::strcmp(feature, "blade_flapping") == 0) config_.enableBladeFlapping = enable;
    else if (std::strcmp(feature, "variable_mass") == 0) config_.enableVariableMass = enable;
    else if (std::strcmp(feature, "advanced_aero") == 0) config_.enableAdvancedAero = enable;
    else if (std::strcmp(feature, "sub_stepping") == 0) config_.subStep.enableSubStepping = enable;
}

void Quadrotor::updateMotorDynamics(const MotorCommand& command, double dt) noexcept {
    double voltage = state_.batteryVoltage;
    
    if (config_.enableNonlinearMotor && config_.enableMotorDynamics) {
        NonlinearMotorModel::updateMotorState(
            motorDynamics_, command, voltage, dt, config_.motor, config_.battery
        );
        for (int i = 0; i < 4; ++i) {
            motorState_.rpm[i] = motorDynamics_.rpm[i];
            motorState_.current[i] = motorDynamics_.current[i];
            motorState_.temperature[i] = motorDynamics_.temperature[i];
            state_.motorRPM[i] = motorDynamics_.rpm[i];
        }
    } else if (config_.enableMotorDynamics) {
        for (int i = 0; i < 4; ++i) {
            double targetRPM = std::clamp(command.rpm[i], 0.0, config_.motor.maxRpm);
            
            motorState_.rpm[i] = PhysicsEngine::computeMotorRPM(
                targetRPM, motorState_.rpm[i], voltage, dt, config_.motor
            );
            
            double backEmf = motorState_.rpm[i] / config_.motor.kv;
            motorState_.current[i] = std::max(0.0, (voltage - backEmf) / config_.motor.resistance);
            state_.motorRPM[i] = motorState_.rpm[i];
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            double targetRPM = std::clamp(command.rpm[i], 0.0, config_.motor.maxRpm);
            motorState_.rpm[i] = targetRPM;
            motorState_.current[i] = 0;
            state_.motorRPM[i] = motorState_.rpm[i];
        }
    }
}

void Quadrotor::updateBladeFlapping(double dt) noexcept {
    if (!config_.enableBladeFlapping) return;
    
    Vec3 velocityBody = state_.orientation.inverseRotate(state_.velocity);
    BladeFlappingModel::updateFlappingState(
        flappingState_, velocityBody, state_.angularVelocity, motorState_.rpm, config_.rotor, dt
    );
}

void Quadrotor::updateMassDynamics(double dt) noexcept {
    if (!config_.enableVariableMass || !config_.fuel.variableMassEnabled) return;
    
    double fuelConsumed = VariableMassModel::computeFuelConsumption(
        motorState_.current, motorState_.rpm, dt, config_.fuel
    );
    
    totalFuelConsumed_ += fuelConsumed;
    config_.fuel.currentFuelMass = std::max(0.0, config_.fuel.initialFuelMass - totalFuelConsumed_);
    
    VariableMassModel::updateMassState(massState_, config_.mass, config_.fuel, totalFuelConsumed_);
    updateInertia();
}

void Quadrotor::updateBattery() noexcept {
    if (!config_.enableBatteryDynamics) return;
    
    double totalCurrent = 0;
    for (int i = 0; i < 4; ++i) {
        totalCurrent += motorState_.current[i];
    }
    
    state_.batteryVoltage = PhysicsEngine::computeBatteryVoltage(config_.battery, totalCurrent, 1.0);
}

void Quadrotor::updateWind(double dt) noexcept {
    if (!config_.enableWindDisturbance) {
        windVelocity_ = Vec3();
        return;
    }
    
    windModel_.setAltitude(state_.position.z);
    windVelocity_ = windModel_.update(dt);
}

void Quadrotor::updateIMU() noexcept {
    if (!config_.enableIMU) return;
    
    Vec3 acceleration = lastForce_ / massState_.currentMass;
    imuReading_ = imuSim_.simulate(state_, acceleration, simulationTime_);
}

Vec3 Quadrotor::computeTotalThrust() const noexcept {
    double totalThrust = PhysicsEngine::computeThrust(
        motorState_.rpm, config_.rotor, config_.aero
    );
    
    Vec3 thrustBody(0, 0, totalThrust);
    
    if (config_.enableGroundEffect) {
        double altitude = state_.position.z - config_.groundZ;
        thrustBody = GroundEffect::computeForce(
            thrustBody, altitude, config_.aero, config_.rotor.radius
        );
    }
    
    return thrustBody;
}

Vec3 Quadrotor::computeTotalTorque() const noexcept {
    Vec3 baseTorque = PhysicsEngine::computeTorques(
        motorState_.rpm, config_.rotor, config_.armLength,
        config_.rotor.dragCoeff, config_.motorConfig
    );
    
    if (config_.enableBladeFlapping) {
        baseTorque = baseTorque + flappingState_.flappingMoment;
    }
    
    return baseTorque;
}

Vec3 Quadrotor::computeAerodynamicForces(const Vec3& velocityBody) const noexcept {
    Vec3 relativeVelocity = velocityBody - state_.orientation.inverseRotate(windVelocity_);
    
    if (config_.enableAdvancedAero) {
        return PhysicsEngine::computeAdvancedAeroDrag(
            relativeVelocity, state_.angularVelocity, motorState_.rpm,
            config_.aero, config_.rotor, flappingState_
        );
    }
    
    return PhysicsEngine::computeAerodynamicDrag(
        relativeVelocity, state_.angularVelocity, config_.aero
    );
}

Vec3 Quadrotor::computeGyroscopicTorque() const noexcept {
    Vec3 omega = state_.angularVelocity;
    Vec3 Iomega = currentInertia_ * omega;
    return omega.cross(Iomega);
}

RigidBodyDerivative Quadrotor::computeDerivative(const RigidBodyState& rbState, double t) const {
    RigidBodyDerivative d;
    
    Vec3 thrustBody = computeTotalThrust();
    Vec3 thrustWorld = rbState.orientation.rotate(thrustBody);
    
    Vec3 gravity(0, 0, -GRAVITY * massState_.currentMass);
    
    Vec3 velocityBody = rbState.orientation.inverseRotate(rbState.velocity);
    Vec3 dragBody = computeAerodynamicForces(velocityBody);
    Vec3 dragWorld = rbState.orientation.rotate(dragBody);
    
    Vec3 windForce = windVelocity_ * (config_.aero.airDensity * config_.aero.parasiticDragArea);
    
    Vec3 totalForce = thrustWorld + gravity + dragWorld + windForce;
    
    d.velocity = rbState.velocity;
    d.acceleration = totalForce / massState_.currentMass;
    
    Vec3 torques = computeTotalTorque();
    Vec3 omega = rbState.angularVelocity;
    Vec3 Iomega = currentInertia_ * omega;
    Vec3 gyroscopic = omega.cross(Iomega);
    
    d.angularAcceleration = cachedInertiaInverse_ * (torques - gyroscopic);
    
    d.orientationDot = rbState.orientation.derivative(omega);
    
    return d;
}

void Quadrotor::stepInternal(const MotorCommand& command, double dt) {
    updateMotorDynamics(command, dt);
    updateBattery();
    updateWind(dt);
    updateBladeFlapping(dt);
    updateMassDynamics(dt);
    
    RigidBodyState rbState;
    rbState.position = state_.position;
    rbState.velocity = state_.velocity;
    rbState.orientation = state_.orientation;
    rbState.angularVelocity = state_.angularVelocity;
    
    auto derivativeFunc = [this](const RigidBodyState& s, double t) {
        return this->computeDerivative(s, t);
    };
    
    RigidBodyState nextState = physics_.integrate(rbState, derivativeFunc, dt, simulationTime_);
    
    state_.position = nextState.position;
    state_.velocity = nextState.velocity;
    state_.orientation = nextState.orientation;
    state_.angularVelocity = nextState.angularVelocity;
    
    lastForce_ = state_.orientation.rotate(computeTotalThrust()) + Vec3(0, 0, -GRAVITY * massState_.currentMass);
    lastTorque_ = computeTotalTorque();
    
    simulationTime_ += dt;
    state_.time = simulationTime_;
    
    enforceGroundConstraint();
}

void Quadrotor::step(const MotorCommand& command, double dt) {
    IntegrationGuard guard(integrating_);
    
    stepInternal(command, dt);
    lastSubStepCount_ = 1;
    
    updateIMU();
}

void Quadrotor::stepWithSubStepping(const MotorCommand& command, double agentDt) {
    IntegrationGuard guard(integrating_);
    
    if (!config_.subStep.enableSubStepping || config_.subStep.physicsSubSteps <= 1) {
        stepInternal(command, agentDt);
        lastSubStepCount_ = 1;
        updateIMU();
        return;
    }
    
    int numSubSteps = config_.subStep.physicsSubSteps;
    double physicsDt = agentDt / static_cast<double>(numSubSteps);
    
    physicsDt = std::clamp(physicsDt, config_.subStep.minSubStepDt, config_.subStep.maxSubStepDt);
    numSubSteps = static_cast<int>(std::ceil(agentDt / physicsDt));
    physicsDt = agentDt / static_cast<double>(numSubSteps);
    
    for (int i = 0; i < numSubSteps; ++i) {
        stepInternal(command, physicsDt);
    }
    
    lastSubStepCount_ = numSubSteps;
    updateIMU();
}

void Quadrotor::stepAdaptive(const MotorCommand& command, double& dt) {
    IntegrationGuard guard(integrating_);
    
    updateMotorDynamics(command, dt);
    updateBattery();
    updateWind(dt);
    updateBladeFlapping(dt);
    updateMassDynamics(dt);
    
    RigidBodyState rbState;
    rbState.position = state_.position;
    rbState.velocity = state_.velocity;
    rbState.orientation = state_.orientation;
    rbState.angularVelocity = state_.angularVelocity;
    
    auto derivativeFunc = [this](const RigidBodyState& s, double t) {
        return this->computeDerivative(s, t);
    };
    
    AdaptiveStepResult result = physics_.integrateAdaptive(rbState, derivativeFunc, dt, simulationTime_);
    
    if (result.accepted) {
        state_.position = result.state.position;
        state_.velocity = result.state.velocity;
        state_.orientation = result.state.orientation;
        state_.angularVelocity = result.state.angularVelocity;
        simulationTime_ += dt;
    }
    
    dt = result.actualDt;
    
    lastForce_ = state_.orientation.rotate(computeTotalThrust()) + Vec3(0, 0, -GRAVITY * massState_.currentMass);
    lastTorque_ = computeTotalTorque();
    state_.time = simulationTime_;
    
    enforceGroundConstraint();
    updateIMU();
}

void Quadrotor::enforceGroundConstraint() noexcept {
    double groundZ = config_.groundZ;
    
    if (state_.position.z < groundZ) {
        state_.position.z = groundZ;
        
        if (state_.velocity.z < 0) {
            state_.velocity.z *= -config_.groundRestitution;
            
            double frictionMag = config_.groundFriction * std::abs(state_.velocity.z);
            double horizontalSpeed = std::sqrt(
                state_.velocity.x * state_.velocity.x + 
                state_.velocity.y * state_.velocity.y
            );
            
            if (horizontalSpeed > 1e-6) {
                double reduction = std::min(1.0, frictionMag / horizontalSpeed);
                state_.velocity.x *= (1.0 - reduction);
                state_.velocity.y *= (1.0 - reduction);
            }
        }
        
        state_.angularVelocity = state_.angularVelocity * 0.9;
    }
}