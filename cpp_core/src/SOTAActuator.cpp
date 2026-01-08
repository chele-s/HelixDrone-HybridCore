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

void SOTAActuatorModel::computeDelaySteps() noexcept {
    if (config_.delayMs <= 0.0 || config_.simulationDt <= 0.0) {
        state_.delaySteps = 0;
        return;
    }
    
    double delaySeconds = config_.delayMs * 0.001;
    double stepsFloat = delaySeconds / config_.simulationDt;
    state_.delaySteps = static_cast<int>(stepsFloat + 0.5);
    state_.delaySteps = std::clamp(state_.delaySteps, 0, 31);
}

void SOTAActuatorModel::reset() noexcept {
    for (int i = 0; i < 4; ++i) {
        state_.currentRpm[i] = config_.hoverRpm;
        state_.filteredRpm[i] = config_.hoverRpm;
        state_.motorTemperature[i] = 25.0;
        state_.rpmDerivative[i] = 0.0;
        state_.totalCurrent[i] = 0.0;
        state_.thermalEfficiency[i] = 1.0;
        state_.previousTarget[i] = config_.hoverRpm;
        for (int j = 0; j < 32; ++j) {
            state_.commandBuffer[i][j] = config_.hoverRpm;
        }
    }
    state_.bufferIndex = 0;
    computeDelaySteps();
}

void SOTAActuatorModel::setConfig(const SOTAActuatorConfig& config) noexcept {
    bool needsDerivativeReset = std::abs(config.tauSpinUp - config_.tauSpinUp) > 0.001 ||
                                 std::abs(config.tauSpinDown - config_.tauSpinDown) > 0.001 ||
                                 std::abs(config.dampingRatio - config_.dampingRatio) > 0.01;
    
    config_ = config;
    
    if (needsDerivativeReset) {
        for (int i = 0; i < 4; ++i) {
            state_.rpmDerivative[i] = 0.0;
        }
    }
    
    computeDelaySteps();
}

double SOTAActuatorModel::computeFirstOrderResponse(
    double target,
    double current,
    double tau,
    double dt
) noexcept {
    if (tau <= 1e-9) return target;
    double alpha = 1.0 - std::exp(-dt / tau);
    return current + alpha * (target - current);
}

void SOTAActuatorModel::computeSecondOrderResponse(
    double target,
    double current,
    double derivative,
    double tau,
    double damping,
    double dt,
    double inertia,
    double& newPosition,
    double& newDerivative
) const noexcept {
    double effectiveTau = std::max(tau, 1e-6);
    double inertiaFactor = 1.0 + inertia * 1e4;
    double omega_n = 1.0 / (effectiveTau * std::sqrt(inertiaFactor));
    double zeta = std::clamp(damping, 0.01, 5.0);
    
    double x0 = current - target;
    double v0 = derivative;
    
    if (zeta >= 1.0) {
        if (std::abs(zeta - 1.0) < 0.01) {
            double lambda = -omega_n;
            double expLt = std::exp(lambda * dt);
            double c1 = x0;
            double c2 = v0 - lambda * x0;
            
            newPosition = target + expLt * (c1 + c2 * dt);
            newDerivative = expLt * (lambda * c1 + c2 + lambda * c2 * dt);
        } else {
            double sqrtTerm = std::sqrt(zeta * zeta - 1.0);
            double lambda1 = omega_n * (-zeta + sqrtTerm);
            double lambda2 = omega_n * (-zeta - sqrtTerm);
            
            double denom = lambda1 - lambda2;
            if (std::abs(denom) < 1e-12) {
                denom = 1e-12;
            }
            
            double c1 = (v0 - lambda2 * x0) / denom;
            double c2 = (lambda1 * x0 - v0) / denom;
            
            double exp1 = std::exp(lambda1 * dt);
            double exp2 = std::exp(lambda2 * dt);
            
            newPosition = target + c1 * exp1 + c2 * exp2;
            newDerivative = c1 * lambda1 * exp1 + c2 * lambda2 * exp2;
        }
    } else {
        double omega_d = omega_n * std::sqrt(1.0 - zeta * zeta);
        double sigma = zeta * omega_n;
        
        double expDecay = std::exp(-sigma * dt);
        double cosWd = std::cos(omega_d * dt);
        double sinWd = std::sin(omega_d * dt);
        
        double A = x0;
        double B = (v0 + sigma * x0) / omega_d;
        
        newPosition = target + expDecay * (A * cosWd + B * sinWd);
        
        double dA_dt = -sigma * A + omega_d * B;
        double dB_dt = -sigma * B - omega_d * A;
        newDerivative = expDecay * (dA_dt * cosWd + dB_dt * sinWd);
    }
    
    if (!std::isfinite(newPosition)) {
        double alpha = 1.0 - std::exp(-dt / effectiveTau);
        newPosition = current + alpha * (target - current);
    }
    if (!std::isfinite(newDerivative)) {
        newDerivative = (newPosition - current) / std::max(dt, 1e-6);
    }
}

double SOTAActuatorModel::computeAsymmetricTau(
    double targetRpm,
    double currentRpm,
    double tauUp,
    double tauDown,
    double activeBrakingGain
) noexcept {
    double delta = targetRpm - currentRpm;
    
    if (delta >= 0.0) {
        return std::max(tauUp, 1e-6);
    }
    
    double brakingFactor = 1.0;
    if (activeBrakingGain > 0.0) {
        constexpr double maxDelta = 10000.0;
        brakingFactor = 1.0 + activeBrakingGain * (std::abs(delta) / maxDelta);
        brakingFactor = std::min(brakingFactor, 3.0);
    }
    return std::max(tauDown / brakingFactor, 1e-6);
}

double SOTAActuatorModel::computeVoltageSag(
    double voltage,
    double nominalVoltage,
    const double* motorCurrents,
    double sagFactor
) noexcept {
    if (sagFactor <= 0.0 || nominalVoltage <= 0.0) return 1.0;
    
    double totalCurrent = 0.0;
    for (int i = 0; i < 4; ++i) {
        totalCurrent += motorCurrents[i];
    }
    
    constexpr double maxCurrent = 120.0;
    double sagRatio = std::clamp(totalCurrent / maxCurrent, 0.0, 1.0);
    
    double sagMultiplier = 1.0 - sagFactor * sagRatio * sagRatio;
    
    return sagMultiplier * (voltage / nominalVoltage);
}

void SOTAActuatorModel::updateDelayBuffer(const double* commandedRpm) noexcept {
    for (int i = 0; i < 4; ++i) {
        state_.commandBuffer[i][state_.bufferIndex] = commandedRpm[i];
    }
    state_.bufferIndex = (state_.bufferIndex + 1) & 31;
}

void SOTAActuatorModel::getDelayedCommand(double* delayedRpm) const noexcept {
    int writePos = (state_.bufferIndex - 1 + 32) & 31;
    int readIndex = (writePos - state_.delaySteps + 32) & 31;
    
    for (int i = 0; i < 4; ++i) {
        delayedRpm[i] = state_.commandBuffer[i][readIndex];
    }
}

void SOTAActuatorModel::applyMotorDynamics(const double* targetRpm, double dt, double voltage) noexcept {
    double voltageRatio = 1.0;
    if (config_.voltageSagFactor > 0.0) {
        voltageRatio = computeVoltageSag(voltage, config_.nominalVoltage, state_.totalCurrent, config_.voltageSagFactor);
        voltageRatio = std::clamp(voltageRatio, 0.8, 1.2);
    }
    
    for (int i = 0; i < 4; ++i) {
        double effectiveTarget = targetRpm[i];
        
        if (config_.voltageSagFactor > 0.0) {
            effectiveTarget *= voltageRatio;
        }
        
        effectiveTarget *= state_.thermalEfficiency[i];
        effectiveTarget = std::clamp(effectiveTarget, config_.minRpm, config_.maxRpm);
        
        state_.previousTarget[i] = effectiveTarget;
        
        double tau = computeAsymmetricTau(
            effectiveTarget,
            state_.currentRpm[i],
            config_.tauSpinUp,
            config_.tauSpinDown,
            config_.activeBrakingGain
        );
        
        double newRpm;
        double newDerivative;
        
        if (config_.tauSpinUp < 0.001 && config_.tauSpinDown < 0.001) {
            newRpm = effectiveTarget;
            newDerivative = 0.0;
        } else {
            computeSecondOrderResponse(
                effectiveTarget,
                state_.currentRpm[i],
                state_.rpmDerivative[i],
                tau,
                config_.dampingRatio,
                dt,
                config_.rotorInertia,
                newRpm,
                newDerivative
            );
        }
        
        double maxChange = config_.maxSlewRate * dt;
        double deltaRpm = newRpm - state_.currentRpm[i];
        deltaRpm = std::clamp(deltaRpm, -maxChange, maxChange);
        newRpm = state_.currentRpm[i] + deltaRpm;
        
        state_.currentRpm[i] = std::clamp(newRpm, config_.minRpm, config_.maxRpm);
        state_.rpmDerivative[i] = std::clamp(newDerivative, -config_.maxSlewRate, config_.maxSlewRate);
        
        double rpmRatio = state_.currentRpm[i] / config_.maxRpm;
        state_.totalCurrent[i] = 30.0 * rpmRatio * rpmRatio;
    }
}

void SOTAActuatorModel::applyOutputSmoothing(double dt) noexcept {
    for (int i = 0; i < 4; ++i) {
        double alpha = std::min(dt * 100.0, 1.0);
        state_.filteredRpm[i] += alpha * (state_.currentRpm[i] - state_.filteredRpm[i]);
        state_.filteredRpm[i] = std::clamp(state_.filteredRpm[i], config_.minRpm, config_.maxRpm);
    }
}

void SOTAActuatorModel::addProcessNoise() noexcept {
    if (config_.processNoiseStd <= 0.0) return;
    
    for (int i = 0; i < 4; ++i) {
        double noise = noiseDist_(rng_) * config_.processNoiseStd;
        state_.filteredRpm[i] += noise;
        state_.filteredRpm[i] = std::clamp(state_.filteredRpm[i], config_.minRpm, config_.maxRpm);
    }
}

void SOTAActuatorModel::updateThermalModel(double dt) noexcept {
    if (config_.thermalTimeConstant <= 0.0) return;
    
    constexpr double ambientTemp = 25.0;
    double tempRange = config_.maxTemperature - ambientTemp;
    if (tempRange <= 0.0) {
        tempRange = 55.0;
    }
    
    for (int i = 0; i < 4; ++i) {
        double powerDissipation = state_.totalCurrent[i] * state_.totalCurrent[i] * 0.1;
        double equilibriumTemp = ambientTemp + powerDissipation * 0.5;
        equilibriumTemp = std::min(equilibriumTemp, config_.maxTemperature);
        
        double alpha = std::min(dt / config_.thermalTimeConstant, 1.0);
        state_.motorTemperature[i] += alpha * (equilibriumTemp - state_.motorTemperature[i]);
        state_.motorTemperature[i] = std::clamp(state_.motorTemperature[i], ambientTemp, config_.maxTemperature);
        
        double tempRatio = (state_.motorTemperature[i] - ambientTemp) / tempRange;
        state_.thermalEfficiency[i] = 1.0 - config_.thermalDerating * tempRatio;
        state_.thermalEfficiency[i] = std::clamp(state_.thermalEfficiency[i], 0.8, 1.0);
    }
}

void SOTAActuatorModel::step(const double* commandedRpm, double dt, double voltage, double* outputRpm) noexcept {
    updateDelayBuffer(commandedRpm);
    
    double delayedCommand[4];
    if (state_.delaySteps > 0) {
        getDelayedCommand(delayedCommand);
    } else {
        for (int i = 0; i < 4; ++i) {
            delayedCommand[i] = commandedRpm[i];
        }
    }
    
    applyMotorDynamics(delayedCommand, dt, voltage);
    applyOutputSmoothing(dt);
    addProcessNoise();
    updateThermalModel(dt);
    
    for (int i = 0; i < 4; ++i) {
        outputRpm[i] = state_.filteredRpm[i];
    }
}

void SOTAActuatorModel::stepNormalized(const double* normalizedAction, double dt, double voltage, double* outputRpm) noexcept {
    double commandedRpm[4];
    for (int i = 0; i < 4; ++i) {
        double action = std::clamp(normalizedAction[i], -1.0, 1.0);
        commandedRpm[i] = config_.hoverRpm + action * config_.rpmRange;
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
    : a1_(0.0), a2_(0.0)
    , b0_(1.0), b1_(0.0), b2_(0.0)
    , x1_(0.0), x2_(0.0), y1_(0.0), y2_(0.0) {}

SOTASecondOrderFilter::SOTASecondOrderFilter(double cutoffHz, double damping, double dt) noexcept
    : a1_(0.0), a2_(0.0)
    , b0_(1.0), b1_(0.0), b2_(0.0)
    , x1_(0.0), x2_(0.0), y1_(0.0), y2_(0.0) {
    configure(cutoffHz, damping, dt);
}

void SOTASecondOrderFilter::configure(double cutoffHz, double damping, double dt) noexcept {
    if (cutoffHz <= 0.0 || dt <= 0.0) {
        a1_ = 0.0; a2_ = 0.0;
        b0_ = 1.0; b1_ = 0.0; b2_ = 0.0;
        return;
    }
    
    double omega = 2.0 * M_PI * cutoffHz;
    double omega_d = omega * dt;
    double omega_d2 = omega_d * omega_d;
    double zeta_omega_d_2 = 4.0 * damping * omega_d;
    
    double denom = 4.0 + zeta_omega_d_2 + omega_d2;
    
    b0_ = omega_d2 / denom;
    b1_ = 2.0 * omega_d2 / denom;
    b2_ = omega_d2 / denom;
    
    a1_ = (2.0 * omega_d2 - 8.0) / denom;
    a2_ = (4.0 - zeta_omega_d_2 + omega_d2) / denom;
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

SOTAButterworthFilter::SOTAButterworthFilter(int order, double cutoffHz, double sampleRate) noexcept
    : order_(2) {
    std::memset(a_, 0, sizeof(a_));
    std::memset(b_, 0, sizeof(b_));
    std::memset(x_, 0, sizeof(x_));
    std::memset(y_, 0, sizeof(y_));
    configure(order, cutoffHz, sampleRate);
}

void SOTAButterworthFilter::configure(int order, double cutoffHz, double sampleRate) noexcept {
    order_ = std::clamp(order, 1, MAX_ORDER);
    std::memset(x_, 0, sizeof(x_));
    std::memset(y_, 0, sizeof(y_));
    
    if (cutoffHz <= 0.0 || sampleRate <= 0.0) {
        a_[0] = 1.0; a_[1] = 0.0; a_[2] = 0.0;
        b_[0] = 1.0; b_[1] = 0.0; b_[2] = 0.0;
        return;
    }
    
    double nyquist = sampleRate * 0.5;
    double w = std::clamp(cutoffHz / nyquist, 0.001, 0.99);
    
    if (order_ == 1) {
        double gamma = std::tan(M_PI * w * 0.5);
        double denom = 1.0 + gamma;
        
        b_[0] = gamma / denom;
        b_[1] = gamma / denom;
        b_[2] = 0.0;
        a_[0] = 1.0;
        a_[1] = (gamma - 1.0) / denom;
        a_[2] = 0.0;
    } else {
        double omega = std::tan(M_PI * w * 0.5);
        double omega2 = omega * omega;
        double sqrt2_omega = std::sqrt(2.0) * omega;
        double denom = 1.0 + sqrt2_omega + omega2;
        
        b_[0] = omega2 / denom;
        b_[1] = 2.0 * omega2 / denom;
        b_[2] = omega2 / denom;
        a_[0] = 1.0;
        a_[1] = 2.0 * (omega2 - 1.0) / denom;
        a_[2] = (1.0 - sqrt2_omega + omega2) / denom;
    }
}

double SOTAButterworthFilter::filter(double input) noexcept {
    double output = b_[0] * input + b_[1] * x_[0] - a_[1] * y_[0];
    
    if (order_ == 2) {
        output += b_[2] * x_[1] - a_[2] * y_[1];
        x_[1] = x_[0];
        y_[1] = y_[0];
    }
    
    x_[0] = input;
    y_[0] = output;
    
    return output;
}

void SOTAButterworthFilter::reset() noexcept {
    std::memset(x_, 0, sizeof(x_));
    std::memset(y_, 0, sizeof(y_));
}
