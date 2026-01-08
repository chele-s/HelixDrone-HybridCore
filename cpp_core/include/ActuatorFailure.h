#pragma once
#include "Types.h"
#include <array>
#include <random>
#include <cmath>
#include <functional>

enum class FailureType {
    NONE = 0,
    STUCK,
    REDUCED_POWER,
    OSCILLATING,
    DELAYED_RESPONSE,
    RANDOM_CUTOUT,
    THERMAL_DEGRADATION,
    PARTIAL_LOSS,
    COMPLETE_LOSS
};

enum class FailureMode {
    PERMANENT,
    INTERMITTENT,
    PROGRESSIVE
};

struct FailureConfig {
    FailureType type = FailureType::NONE;
    FailureMode mode = FailureMode::PERMANENT;
    int motor_id = -1;
    double severity = 0.0;
    double stuck_rpm = 0.0;
    double oscillation_freq = 5.0;
    double oscillation_amplitude = 0.2;
    double delay_factor = 2.0;
    double cutout_probability = 0.1;
    double thermal_rate = 0.01;
    double intermittent_on_time = 1.0;
    double intermittent_off_time = 0.5;
    double progressive_rate = 0.001;
    double start_time = 0.0;
    double duration = -1.0;
};

struct MotorFailureState {
    bool active[4] = {false, false, false, false};
    double severity[4] = {0.0, 0.0, 0.0, 0.0};
    double stuck_rpm[4] = {0.0, 0.0, 0.0, 0.0};
    double thermal_factor[4] = {1.0, 1.0, 1.0, 1.0};
    double phase[4] = {0.0, 0.0, 0.0, 0.0};
    double intermittent_timer[4] = {0.0, 0.0, 0.0, 0.0};
    bool intermittent_on[4] = {true, true, true, true};
    double progressive_severity[4] = {0.0, 0.0, 0.0, 0.0};
    FailureConfig configs[4];
};

class FailureEffect {
public:
    virtual ~FailureEffect() = default;
    virtual double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept = 0;
    virtual void reset(int motor_id, MotorFailureState& state) noexcept = 0;
};

class StuckFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        return state.stuck_rpm[motor_id];
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {
        state.stuck_rpm[motor_id] = 0.0;
    }
};

class ReducedPowerFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        double factor = 1.0 - state.severity[motor_id];
        return commanded_rpm * factor;
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {
        state.severity[motor_id] = 0.0;
    }
};

class OscillatingFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        const auto& cfg = state.configs[motor_id];
        state.phase[motor_id] += 2.0 * M_PI * cfg.oscillation_freq * dt;
        if (state.phase[motor_id] > 2.0 * M_PI) state.phase[motor_id] -= 2.0 * M_PI;
        
        double osc = std::sin(state.phase[motor_id]) * cfg.oscillation_amplitude;
        return commanded_rpm * (1.0 + osc);
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {
        state.phase[motor_id] = 0.0;
    }
};

class DelayedResponseFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        const auto& cfg = state.configs[motor_id];
        double tau = 0.05 * cfg.delay_factor;
        double alpha = dt / (tau + dt);
        state.stuck_rpm[motor_id] += alpha * (commanded_rpm - state.stuck_rpm[motor_id]);
        return state.stuck_rpm[motor_id];
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {
        state.stuck_rpm[motor_id] = 0.0;
    }
};

class RandomCutoutFailure : public FailureEffect {
public:
    explicit RandomCutoutFailure(uint32_t seed = 42) : rng_(seed), dist_(0.0, 1.0) {}
    
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        const auto& cfg = state.configs[motor_id];
        if (dist_(rng_) < cfg.cutout_probability * dt) {
            return 0.0;
        }
        return commanded_rpm;
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {}
    
private:
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};

class ThermalDegradationFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        const auto& cfg = state.configs[motor_id];
        
        double power_factor = commanded_rpm / 20000.0;
        state.thermal_factor[motor_id] -= cfg.thermal_rate * power_factor * dt;
        state.thermal_factor[motor_id] = std::max(0.3, state.thermal_factor[motor_id]);
        
        return commanded_rpm * state.thermal_factor[motor_id];
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {
        state.thermal_factor[motor_id] = 1.0;
    }
};

class PartialLossFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        double factor = 1.0 - state.severity[motor_id];
        double noise = (std::sin(state.phase[motor_id]) * 0.1 + 1.0);
        state.phase[motor_id] += 10.0 * dt;
        return commanded_rpm * factor * noise;
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {
        state.severity[motor_id] = 0.0;
        state.phase[motor_id] = 0.0;
    }
};

class CompleteLossFailure : public FailureEffect {
public:
    double apply(double commanded_rpm, int motor_id, double dt, MotorFailureState& state) noexcept override {
        return 0.0;
    }
    
    void reset(int motor_id, MotorFailureState& state) noexcept override {}
};

class ActuatorFailureInjector {
public:
    ActuatorFailureInjector() noexcept;
    
    void injectFailure(const FailureConfig& config) noexcept;
    void injectFailure(int motor_id, FailureType type, double severity) noexcept;
    void clearFailure(int motor_id) noexcept;
    void clearAllFailures() noexcept;
    
    void scheduleFailure(const FailureConfig& config, double time) noexcept;
    
    void applyFailures(double* commanded_rpm, double dt, double current_time) noexcept;
    void applyFailures(double* commanded_rpm, double dt) noexcept;
    
    void reset() noexcept;
    
    bool hasActiveFailure() const noexcept;
    bool hasActiveFailure(int motor_id) const noexcept;
    FailureType getFailureType(int motor_id) const noexcept;
    double getFailureSeverity(int motor_id) const noexcept;
    
    const MotorFailureState& getState() const noexcept { return state_; }
    
    void setRandomFailureProbability(double prob) noexcept { random_failure_prob_ = prob; }
    void enableRandomFailures(bool enable) noexcept { random_failures_enabled_ = enable; }
    void generateRandomFailure(double dt) noexcept;
    
private:
    MotorFailureState state_;
    double current_time_ = 0.0;
    
    StuckFailure stuck_effect_;
    ReducedPowerFailure reduced_effect_;
    OscillatingFailure oscillating_effect_;
    DelayedResponseFailure delayed_effect_;
    RandomCutoutFailure cutout_effect_;
    ThermalDegradationFailure thermal_effect_;
    PartialLossFailure partial_effect_;
    CompleteLossFailure complete_effect_;
    
    bool random_failures_enabled_ = false;
    double random_failure_prob_ = 0.0001;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
    
    struct ScheduledFailure {
        FailureConfig config;
        double trigger_time;
        bool triggered;
    };
    std::array<ScheduledFailure, 16> scheduled_failures_;
    int scheduled_count_ = 0;
    
    FailureEffect* getEffect(FailureType type) noexcept;
    void updateIntermittent(int motor_id, double dt) noexcept;
    void updateProgressive(int motor_id, double dt) noexcept;
    void checkScheduledFailures(double current_time) noexcept;
};

class FailureScenarioGenerator {
public:
    enum class Scenario {
        SINGLE_MOTOR_LOSS,
        DUAL_MOTOR_LOSS,
        OPPOSITE_MOTOR_LOSS,
        ADJACENT_MOTOR_LOSS,
        PROGRESSIVE_DEGRADATION,
        RANDOM_INTERMITTENT,
        THERMAL_RUNWAY,
        CASCADING_FAILURE
    };
    
    static void generateScenario(ActuatorFailureInjector& injector, Scenario scenario, double severity = 1.0) noexcept;
    
    static void singleMotorLoss(ActuatorFailureInjector& injector, int motor_id) noexcept;
    static void dualMotorLoss(ActuatorFailureInjector& injector, int motor1, int motor2) noexcept;
    static void oppositeMotorLoss(ActuatorFailureInjector& injector) noexcept;
    static void adjacentMotorLoss(ActuatorFailureInjector& injector) noexcept;
    static void progressiveDegradation(ActuatorFailureInjector& injector, double rate) noexcept;
    static void randomIntermittent(ActuatorFailureInjector& injector, double probability) noexcept;
    static void thermalRunaway(ActuatorFailureInjector& injector, int motor_id, double rate) noexcept;
    static void cascadingFailure(ActuatorFailureInjector& injector, double interval) noexcept;
};
