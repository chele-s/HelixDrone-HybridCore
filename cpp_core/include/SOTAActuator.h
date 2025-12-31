#pragma once
#include "Types.h"
#include <array>
#include <cmath>
#include <algorithm>
#include <random>

struct SOTAActuatorConfig {
    double delayMs;
    double tauSpinUp;
    double tauSpinDown;
    double rotorInertia;
    double voltageSagFactor;
    double maxRpm;
    double minRpm;
    double hoverRpm;
    double maxSlewRate;
    double processNoiseStd;
    double activeBrakingGain;
    double thermalTimeConstant;
    double nominalVoltage;
    
    constexpr SOTAActuatorConfig() noexcept
        : delayMs(20.0)
        , tauSpinUp(0.12)
        , tauSpinDown(0.06)
        , rotorInertia(2.5e-5)
        , voltageSagFactor(0.08)
        , maxRpm(35000.0)
        , minRpm(2000.0)
        , hoverRpm(2750.0)
        , maxSlewRate(45000.0)
        , processNoiseStd(50.0)
        , activeBrakingGain(1.5)
        , thermalTimeConstant(30.0)
        , nominalVoltage(16.8) {}
};

struct SOTAActuatorState {
    alignas(32) double commandBuffer[4][32];
    alignas(32) double currentRpm[4];
    alignas(32) double filteredRpm[4];
    alignas(32) double rpmDerivative[4];
    alignas(32) double motorTemperature[4];
    alignas(32) double totalCurrent[4];
    int bufferIndex;
    int delaySteps;
    double accumulatedSag;
    
    constexpr SOTAActuatorState() noexcept
        : commandBuffer{}
        , currentRpm{0,0,0,0}
        , filteredRpm{0,0,0,0}
        , rpmDerivative{0,0,0,0}
        , motorTemperature{25,25,25,25}
        , totalCurrent{0,0,0,0}
        , bufferIndex(0)
        , delaySteps(2)
        , accumulatedSag(0) {}
};

class alignas(64) SOTAActuatorModel {
public:
    SOTAActuatorModel() noexcept;
    explicit SOTAActuatorModel(const SOTAActuatorConfig& config) noexcept;
    
    void reset() noexcept;
    
    void step(const double* commandedRpm, double dt, double voltage, double* outputRpm) noexcept;
    
    void stepNormalized(const double* normalizedAction, double dt, double voltage, double* outputRpm) noexcept;
    
    double getVoltageSagFactor() const noexcept;
    double getEffectiveTau(int motor, double targetRpm) const noexcept;
    
    const SOTAActuatorState& getState() const noexcept { return state_; }
    const SOTAActuatorConfig& getConfig() const noexcept { return config_; }
    
    void setConfig(const SOTAActuatorConfig& config) noexcept;
    
    static double computeSecondOrderResponse(
        double target,
        double current,
        double derivative,
        double tau,
        double damping,
        double dt,
        double& newDerivative
    ) noexcept;
    
    static double computeAsymmetricTau(
        double targetRpm,
        double currentRpm,
        double tauUp,
        double tauDown,
        double activeBrakingGain
    ) noexcept;
    
    static double computeVoltageSag(
        double voltage,
        double nominalVoltage,
        const double* motorCurrents,
        double sagFactor
    ) noexcept;

private:
    SOTAActuatorConfig config_;
    SOTAActuatorState state_;
    std::mt19937 rng_;
    std::normal_distribution<double> noiseDist_;
    
    void updateDelayBuffer(const double* commandedRpm) noexcept;
    void getDelayedCommand(double* delayedRpm) const noexcept;
    void applyMotorDynamics(const double* targetRpm, double dt, double voltage) noexcept;
    void applySlewRateLimit(double dt) noexcept;
    void addProcessNoise() noexcept;
    void updateThermalModel(double dt) noexcept;
};

class SOTASecondOrderFilter {
public:
    SOTASecondOrderFilter() noexcept;
    explicit SOTASecondOrderFilter(double cutoffHz, double damping, double dt) noexcept;
    
    void configure(double cutoffHz, double damping, double dt) noexcept;
    double filter(double input) noexcept;
    void reset(double initialValue = 0.0) noexcept;
    
private:
    double a0_, a1_, a2_;
    double b0_, b1_, b2_;
    double x1_, x2_;
    double y1_, y2_;
};

class SOTAButterworthFilter {
public:
    SOTAButterworthFilter() noexcept;
    explicit SOTAButterworthFilter(int order, double cutoffHz, double sampleRate) noexcept;
    
    void configure(int order, double cutoffHz, double sampleRate) noexcept;
    double filter(double input) noexcept;
    void reset() noexcept;
    
private:
    static constexpr int MAX_ORDER = 4;
    double a_[MAX_ORDER + 1];
    double b_[MAX_ORDER + 1];
    double x_[MAX_ORDER + 1];
    double y_[MAX_ORDER + 1];
    int order_;
};
