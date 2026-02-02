#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "Types.h"
#include <array>
#include <cmath>
#include <algorithm>
#include <random>

struct SOTAActuatorConfig {
    double tauSpinUp;
    double tauSpinDown;
    double rotorInertia;
    double voltageSagFactor;
    double maxRpm;
    double minRpm;
    double hoverRpm;
    double rpmRange;
    double maxSlewRate;
    double processNoiseStd;
    double activeBrakingGain;
    double thermalTimeConstant;
    double nominalVoltage;
    double delayMs;
    double dampingRatio;
    double thermalDerating;
    double maxTemperature;
    double simulationDt;
    
    constexpr SOTAActuatorConfig() noexcept
        : tauSpinUp(0.015)
        , tauSpinDown(0.012)
        , rotorInertia(2.5e-5)
        , voltageSagFactor(0.0)
        , maxRpm(35000.0)
        , minRpm(2000.0)
        , hoverRpm(2600.0)
        , rpmRange(3600.0)
        , maxSlewRate(1000000.0)
        , processNoiseStd(0.0)
        , activeBrakingGain(1.2)
        , thermalTimeConstant(30.0)
        , nominalVoltage(16.8)
        , delayMs(0.0)
        , dampingRatio(1.0)
        , thermalDerating(0.0)
        , maxTemperature(80.0)
        , simulationDt(0.004) {}
};

struct SOTAActuatorState {
    alignas(32) double currentRpm[4];
    alignas(32) double filteredRpm[4];
    alignas(32) double rpmDerivative[4];
    alignas(32) double motorTemperature[4];
    alignas(32) double totalCurrent[4];
    alignas(32) double thermalEfficiency[4];
    alignas(32) double previousTarget[4];
    alignas(32) double commandBuffer[4][32];
    int bufferIndex;
    int delaySteps;
    
    SOTAActuatorState() noexcept {
        for (int i = 0; i < 4; ++i) {
            currentRpm[i] = 0.0;
            filteredRpm[i] = 0.0;
            rpmDerivative[i] = 0.0;
            motorTemperature[i] = 25.0;
            totalCurrent[i] = 0.0;
            thermalEfficiency[i] = 1.0;
            previousTarget[i] = 0.0;
            for (int j = 0; j < 32; ++j) {
                commandBuffer[i][j] = 0.0;
            }
        }
        bufferIndex = 0;
        delaySteps = 0;
    }
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
    
    static double computeFirstOrderResponse(
        double target,
        double current,
        double tau,
        double dt
    ) noexcept;
    
    void computeSecondOrderResponse(
        double target,
        double current,
        double derivative,
        double tau,
        double damping,
        double dt,
        double inertia,
        double& newPosition,
        double& newDerivative
    ) const noexcept;

private:
    SOTAActuatorConfig config_;
    SOTAActuatorState state_;
    std::mt19937 rng_;
    std::normal_distribution<double> noiseDist_;
    
    void computeDelaySteps() noexcept;
    void applyMotorDynamics(const double* targetRpm, double dt, double voltage) noexcept;
    void applyOutputSmoothing(double dt) noexcept;
    void addProcessNoise() noexcept;
    void updateThermalModel(double dt) noexcept;
    void updateDelayBuffer(const double* commandedRpm) noexcept;
    void getDelayedCommand(double* delayedRpm) const noexcept;
};

class SOTASecondOrderFilter {
public:
    SOTASecondOrderFilter() noexcept;
    explicit SOTASecondOrderFilter(double cutoffHz, double damping, double dt) noexcept;
    
    void configure(double cutoffHz, double damping, double dt) noexcept;
    double filter(double input) noexcept;
    void reset(double initialValue = 0.0) noexcept;
    
private:
    double a1_, a2_;
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
    static constexpr int MAX_ORDER = 2;
    double a_[MAX_ORDER + 1];
    double b_[MAX_ORDER + 1];
    double x_[MAX_ORDER + 1];
    double y_[MAX_ORDER + 1];
    int order_;
};
