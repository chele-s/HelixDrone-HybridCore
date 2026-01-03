#include "SOTAActuator.h"
#include <cstring>

SOTAActuatorModel::SOTAActuatorModel() noexcept
    : config_()
    , state_()
    , rng_(std::random_device{}())
    , noiseDist_(0.0, 1.0) {
    reset();
}

SOTAActuatorModel::SOTAActuatorModel(const SOTAActuatorConfig& config) noexcept
    : config_(config)
    , state_()
    , rng_(std::random_device{}())
    , noiseDist_(0.0, 1.0) {
    reset();
}

void SOTAActuatorModel::reset() noexcept {
    std::memset(&state_, 0, sizeof(SOTAActuatorState));
    for (int i = 0; i < 4; ++i) {
        state_.currentRpm[i] = config_.hoverRpm;
        state_.filteredRpm[i] = config_.hoverRpm;
        state_.motorTemperature[i] = 25.0;
        for (int j = 0; j < 32; ++j) {
            state_.commandBuffer[i][j] = config_.hoverRpm;
        }
    }
    state_.delaySteps = static_cast<int>(config_.delayMs / 10.0 + 0.5);
    state_.delaySteps = std::clamp(state_.delaySteps, 1, 30);
}

void SOTAActuatorModel::setConfig(const SOTAActuatorConfig& config) noexcept {
    config_ = config;
    state_.delaySteps = static_cast<int>(config_.delayMs / 10.0 + 0.5);
    state_.delaySteps = std::clamp(state_.delaySteps, 1, 30);
}

double SOTAActuatorModel::computeSecondOrderResponse(
    double target,
    double current,
    double derivative,
    double tau,
    double damping,
    double dt,
    double& newDerivative
) noexcept {
    double omega = 1.0 / tau;
    double omega2 = omega * omega;
    
    double error = target - current;
    double acceleration = omega2 * error - 2.0 * damping * omega * derivative;
    
    newDerivative = derivative + acceleration * dt;
    double newPosition = current + derivative * dt + 0.5 * acceleration * dt * dt;
    
    return newPosition;
}

double SOTAActuatorModel::computeAsymmetricTau(
    double targetRpm,
    double currentRpm,
    double tauUp,
    double tauDown,
    double activeBrakingGain
) noexcept {
    double delta = targetRpm - currentRpm;
    
    if (delta > 0) {
        double relativeLoad = currentRpm / 20000.0;
        return tauUp * (1.0 + 0.3 * relativeLoad);
    } else {
        double brakingEfficiency = 1.0 + activeBrakingGain * (std::abs(delta) / 10000.0);
        brakingEfficiency = std::min(brakingEfficiency, 2.5);
        return tauDown / brakingEfficiency;
    }
}

double SOTAActuatorModel::computeVoltageSag(
    double voltage,
    double nominalVoltage,
    const double* motorCurrents,
    double sagFactor
) noexcept {
    double totalCurrent = 0.0;
    for (int i = 0; i < 4; ++i) {
        totalCurrent += motorCurrents[i];
    }
    
    double sagRatio = totalCurrent / 120.0;
    sagRatio = std::min(sagRatio, 1.0);
    
    double sagMultiplier = 1.0 - sagFactor * sagRatio * sagRatio;
    
    return voltage * sagMultiplier / nominalVoltage;
}

void SOTAActuatorModel::updateDelayBuffer(const double* commandedRpm) noexcept {
    for (int i = 0; i < 4; ++i) {
        state_.commandBuffer[i][state_.bufferIndex] = commandedRpm[i];
    }
    state_.bufferIndex = (state_.bufferIndex + 1) % 32;
}

void SOTAActuatorModel::getDelayedCommand(double* delayedRpm) const noexcept {
    int readIndex = (state_.bufferIndex - state_.delaySteps + 32) % 32;
    for (int i = 0; i < 4; ++i) {
        delayedRpm[i] = state_.commandBuffer[i][readIndex];
    }
}

void SOTAActuatorModel::applyMotorDynamics(const double* targetRpm, double dt, double voltage) noexcept {
    double voltageRatio = voltage / config_.nominalVoltage;
    voltageRatio = std::clamp(voltageRatio, 0.7, 1.1);
    
    double currentSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        currentSum += state_.totalCurrent[i];
    }
    double sagFactor = computeVoltageSag(voltage, config_.nominalVoltage, state_.totalCurrent, config_.voltageSagFactor);
    
    for (int i = 0; i < 4; ++i) {
        double effectiveTarget = targetRpm[i] * voltageRatio * sagFactor;
        effectiveTarget = std::clamp(effectiveTarget, config_.minRpm, config_.maxRpm);
        
        double tau = computeAsymmetricTau(
            effectiveTarget,
            state_.currentRpm[i],
            config_.tauSpinUp,
            config_.tauSpinDown,
            config_.activeBrakingGain
        );
        
        double thermalDerating = 1.0;
        if (state_.motorTemperature[i] > 60.0) {
            thermalDerating = 1.0 - 0.01 * (state_.motorTemperature[i] - 60.0);
            thermalDerating = std::max(thermalDerating, 0.7);
        }
        tau /= thermalDerating;
        
        double newDerivative;
        double damping = 0.85;
        
        double newRpm = computeSecondOrderResponse(
            effectiveTarget,
            state_.currentRpm[i],
            state_.rpmDerivative[i],
            tau,
            damping,
            dt,
            newDerivative
        );
        
        state_.rpmDerivative[i] = newDerivative;
        state_.currentRpm[i] = std::clamp(newRpm, config_.minRpm, config_.maxRpm);
        
        double rpmRatio = state_.currentRpm[i] / config_.maxRpm;
        state_.totalCurrent[i] = 30.0 * rpmRatio * rpmRatio / voltageRatio;
    }
}

void SOTAActuatorModel::applySlewRateLimit(double dt) noexcept {
    double maxChange = config_.maxSlewRate * dt;
    
    for (int i = 0; i < 4; ++i) {
        double delta = state_.currentRpm[i] - state_.filteredRpm[i];
        delta = std::clamp(delta, -maxChange, maxChange);
        state_.filteredRpm[i] += delta;
    }
}

void SOTAActuatorModel::addProcessNoise() noexcept {
    for (int i = 0; i < 4; ++i) {
        double noise = noiseDist_(rng_) * config_.processNoiseStd;
        state_.filteredRpm[i] += noise;
        state_.filteredRpm[i] = std::clamp(state_.filteredRpm[i], config_.minRpm, config_.maxRpm);
    }
}

void SOTAActuatorModel::updateThermalModel(double dt) noexcept {
    for (int i = 0; i < 4; ++i) {
        double powerDissipation = state_.totalCurrent[i] * state_.totalCurrent[i] * 0.1;
        double equilibriumTemp = 25.0 + powerDissipation * 0.5;
        
        double alpha = dt / config_.thermalTimeConstant;
        state_.motorTemperature[i] += alpha * (equilibriumTemp - state_.motorTemperature[i]);
    }
}

void SOTAActuatorModel::step(const double* commandedRpm, double dt, double voltage, double* outputRpm) noexcept {
    updateDelayBuffer(commandedRpm);
    
    double delayedCommand[4];
    getDelayedCommand(delayedCommand);
    
    applyMotorDynamics(delayedCommand, dt, voltage);
    
    applySlewRateLimit(dt);
    
    addProcessNoise();
    
    updateThermalModel(dt);
    
    for (int i = 0; i < 4; ++i) {
        outputRpm[i] = state_.filteredRpm[i];
    }
}

void SOTAActuatorModel::stepNormalized(const double* normalizedAction, double dt, double voltage, double* outputRpm) noexcept {
    double commandedRpm[4];
    double rpmRange = config_.rpmRange;
    for (int i = 0; i < 4; ++i) {
        double action = std::clamp(normalizedAction[i], -1.0, 1.0);
        commandedRpm[i] = config_.hoverRpm + action * rpmRange;
        commandedRpm[i] = std::clamp(commandedRpm[i], config_.minRpm, config_.maxRpm);
    }
    step(commandedRpm, dt, voltage, outputRpm);
}

double SOTAActuatorModel::getVoltageSagFactor() const noexcept {
    return computeVoltageSag(config_.nominalVoltage, config_.nominalVoltage, state_.totalCurrent, config_.voltageSagFactor);
}

double SOTAActuatorModel::getEffectiveTau(int motor, double targetRpm) const noexcept {
    if (motor < 0 || motor >= 4) return config_.tauSpinUp;
    return computeAsymmetricTau(targetRpm, state_.currentRpm[motor], 
                                 config_.tauSpinUp, config_.tauSpinDown, 
                                 config_.activeBrakingGain);
}

SOTASecondOrderFilter::SOTASecondOrderFilter() noexcept
    : a0_(1), a1_(0), a2_(0)
    , b0_(1), b1_(0), b2_(0)
    , x1_(0), x2_(0), y1_(0), y2_(0) {}

SOTASecondOrderFilter::SOTASecondOrderFilter(double cutoffHz, double damping, double dt) noexcept
    : x1_(0), x2_(0), y1_(0), y2_(0) {
    configure(cutoffHz, damping, dt);
}

void SOTASecondOrderFilter::configure(double cutoffHz, double damping, double dt) noexcept {
    double omega = 2.0 * M_PI * cutoffHz;
    double omega_d = omega * dt;
    double omega_d2 = omega_d * omega_d;
    double zeta_omega_d = 2.0 * damping * omega_d;
    
    double denom = 4.0 + zeta_omega_d * 2.0 + omega_d2;
    
    b0_ = omega_d2 / denom;
    b1_ = 2.0 * omega_d2 / denom;
    b2_ = omega_d2 / denom;
    
    a0_ = 1.0;
    a1_ = (2.0 * omega_d2 - 8.0) / denom;
    a2_ = (4.0 - zeta_omega_d * 2.0 + omega_d2) / denom;
}

double SOTASecondOrderFilter::filter(double input) noexcept {
    double output = b0_ * input + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    
    x2_ = x1_;
    x1_ = input;
    y2_ = y1_;
    y1_ = output;
    
    return output;
}

void SOTASecondOrderFilter::reset(double initialValue) noexcept {
    x1_ = x2_ = initialValue;
    y1_ = y2_ = initialValue;
}

SOTAButterworthFilter::SOTAButterworthFilter() noexcept
    : order_(2) {
    std::memset(a_, 0, sizeof(a_));
    std::memset(b_, 0, sizeof(b_));
    std::memset(x_, 0, sizeof(x_));
    std::memset(y_, 0, sizeof(y_));
    a_[0] = 1.0;
    b_[0] = 1.0;
}

SOTAButterworthFilter::SOTAButterworthFilter(int order, double cutoffHz, double sampleRate) noexcept {
    configure(order, cutoffHz, sampleRate);
}

void SOTAButterworthFilter::configure(int order, double cutoffHz, double sampleRate) noexcept {
    order_ = std::clamp(order, 1, MAX_ORDER);
    std::memset(x_, 0, sizeof(x_));
    std::memset(y_, 0, sizeof(y_));
    
    double w = cutoffHz / (sampleRate / 2.0);
    w = std::clamp(w, 0.001, 0.99);
    
    if (order_ == 2) {
        double omega = std::tan(M_PI * w / 2.0);
        double omega2 = omega * omega;
        double sqrt2_omega = std::sqrt(2.0) * omega;
        double denom = 1.0 + sqrt2_omega + omega2;
        
        b_[0] = omega2 / denom;
        b_[1] = 2.0 * omega2 / denom;
        b_[2] = omega2 / denom;
        a_[0] = 1.0;
        a_[1] = 2.0 * (omega2 - 1.0) / denom;
        a_[2] = (1.0 - sqrt2_omega + omega2) / denom;
    } else {
        double alpha = std::sin(M_PI * w) / (2.0 * 0.7071);
        double cosw = std::cos(M_PI * w);
        double denom = 1.0 + alpha;
        
        b_[0] = (1.0 - cosw) / 2.0 / denom;
        b_[1] = (1.0 - cosw) / denom;
        b_[2] = (1.0 - cosw) / 2.0 / denom;
        a_[0] = 1.0;
        a_[1] = -2.0 * cosw / denom;
        a_[2] = (1.0 - alpha) / denom;
    }
}

double SOTAButterworthFilter::filter(double input) noexcept {
    double output = b_[0] * input;
    for (int i = 1; i <= order_; ++i) {
        output += b_[i] * x_[i-1] - a_[i] * y_[i-1];
    }
    
    for (int i = order_ - 1; i > 0; --i) {
        x_[i] = x_[i-1];
        y_[i] = y_[i-1];
    }
    x_[0] = input;
    y_[0] = output;
    
    return output;
}

void SOTAButterworthFilter::reset() noexcept {
    std::memset(x_, 0, sizeof(x_));
    std::memset(y_, 0, sizeof(y_));
}
