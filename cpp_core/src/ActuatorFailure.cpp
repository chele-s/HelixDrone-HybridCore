#include "ActuatorFailure.h"
#include <algorithm>

ActuatorFailureInjector::ActuatorFailureInjector() noexcept
    : state_(), current_time_(0.0)
    , stuck_effect_(), reduced_effect_(), oscillating_effect_()
    , delayed_effect_(), cutout_effect_(), thermal_effect_()
    , partial_effect_(), complete_effect_()
    , random_failures_enabled_(false), random_failure_prob_(0.0001)
    , rng_(std::random_device{}()), dist_(0.0, 1.0)
    , scheduled_failures_(), scheduled_count_(0) {
    reset();
}

void ActuatorFailureInjector::reset() noexcept {
    current_time_ = 0.0;
    scheduled_count_ = 0;
    
    for (int i = 0; i < 4; ++i) {
        state_.active[i] = false;
        state_.severity[i] = 0.0;
        state_.stuck_rpm[i] = 0.0;
        state_.thermal_factor[i] = 1.0;
        state_.phase[i] = 0.0;
        state_.intermittent_timer[i] = 0.0;
        state_.intermittent_on[i] = true;
        state_.progressive_severity[i] = 0.0;
        state_.configs[i] = FailureConfig();
    }
    
    for (int i = 0; i < 16; ++i) {
        scheduled_failures_[i].triggered = true;
    }
}

void ActuatorFailureInjector::injectFailure(const FailureConfig& config) noexcept {
    if (config.motor_id < 0 || config.motor_id >= 4) return;
    
    int id = config.motor_id;
    state_.configs[id] = config;
    state_.active[id] = true;
    state_.severity[id] = config.severity;
    state_.stuck_rpm[id] = config.stuck_rpm;
    state_.progressive_severity[id] = config.severity;
    
    if (config.type == FailureType::THERMAL_DEGRADATION) {
        state_.thermal_factor[id] = 1.0;
    }
}

void ActuatorFailureInjector::injectFailure(int motor_id, FailureType type, double severity) noexcept {
    FailureConfig config;
    config.motor_id = motor_id;
    config.type = type;
    config.severity = severity;
    config.stuck_rpm = severity * 5000.0;
    injectFailure(config);
}

void ActuatorFailureInjector::clearFailure(int motor_id) noexcept {
    if (motor_id < 0 || motor_id >= 4) return;
    
    state_.active[motor_id] = false;
    state_.configs[motor_id] = FailureConfig();
    
    auto* effect = getEffect(state_.configs[motor_id].type);
    if (effect) effect->reset(motor_id, state_);
}

void ActuatorFailureInjector::clearAllFailures() noexcept {
    for (int i = 0; i < 4; ++i) {
        clearFailure(i);
    }
}

void ActuatorFailureInjector::scheduleFailure(const FailureConfig& config, double time) noexcept {
    if (scheduled_count_ >= 16) return;
    
    scheduled_failures_[scheduled_count_].config = config;
    scheduled_failures_[scheduled_count_].trigger_time = time;
    scheduled_failures_[scheduled_count_].triggered = false;
    scheduled_count_++;
}

void ActuatorFailureInjector::checkScheduledFailures(double current_time) noexcept {
    for (int i = 0; i < scheduled_count_; ++i) {
        if (!scheduled_failures_[i].triggered && 
            current_time >= scheduled_failures_[i].trigger_time) {
            injectFailure(scheduled_failures_[i].config);
            scheduled_failures_[i].triggered = true;
        }
    }
}

FailureEffect* ActuatorFailureInjector::getEffect(FailureType type) noexcept {
    switch (type) {
        case FailureType::STUCK: return &stuck_effect_;
        case FailureType::REDUCED_POWER: return &reduced_effect_;
        case FailureType::OSCILLATING: return &oscillating_effect_;
        case FailureType::DELAYED_RESPONSE: return &delayed_effect_;
        case FailureType::RANDOM_CUTOUT: return &cutout_effect_;
        case FailureType::THERMAL_DEGRADATION: return &thermal_effect_;
        case FailureType::PARTIAL_LOSS: return &partial_effect_;
        case FailureType::COMPLETE_LOSS: return &complete_effect_;
        default: return nullptr;
    }
}

void ActuatorFailureInjector::updateIntermittent(int motor_id, double dt) noexcept {
    auto& cfg = state_.configs[motor_id];
    if (cfg.mode != FailureMode::INTERMITTENT) return;
    
    state_.intermittent_timer[motor_id] += dt;
    
    if (state_.intermittent_on[motor_id]) {
        if (state_.intermittent_timer[motor_id] >= cfg.intermittent_on_time) {
            state_.intermittent_timer[motor_id] = 0.0;
            state_.intermittent_on[motor_id] = false;
        }
    } else {
        if (state_.intermittent_timer[motor_id] >= cfg.intermittent_off_time) {
            state_.intermittent_timer[motor_id] = 0.0;
            state_.intermittent_on[motor_id] = true;
        }
    }
}

void ActuatorFailureInjector::updateProgressive(int motor_id, double dt) noexcept {
    auto& cfg = state_.configs[motor_id];
    if (cfg.mode != FailureMode::PROGRESSIVE) return;
    
    state_.progressive_severity[motor_id] += cfg.progressive_rate * dt;
    state_.progressive_severity[motor_id] = std::min(1.0, state_.progressive_severity[motor_id]);
    state_.severity[motor_id] = state_.progressive_severity[motor_id];
}

void ActuatorFailureInjector::generateRandomFailure(double dt) noexcept {
    if (!random_failures_enabled_) return;
    
    for (int i = 0; i < 4; ++i) {
        if (!state_.active[i] && dist_(rng_) < random_failure_prob_ * dt) {
            int failure_type = static_cast<int>(dist_(rng_) * 7) + 1;
            double severity = 0.3 + dist_(rng_) * 0.7;
            injectFailure(i, static_cast<FailureType>(failure_type), severity);
        }
    }
}

void ActuatorFailureInjector::applyFailures(double* commanded_rpm, double dt) noexcept {
    applyFailures(commanded_rpm, dt, current_time_);
    current_time_ += dt;
}

void ActuatorFailureInjector::applyFailures(double* commanded_rpm, double dt, double current_time) noexcept {
    checkScheduledFailures(current_time);
    generateRandomFailure(dt);
    
    for (int i = 0; i < 4; ++i) {
        if (!state_.active[i]) continue;
        
        auto& cfg = state_.configs[i];
        
        if (cfg.duration > 0 && current_time > cfg.start_time + cfg.duration) {
            clearFailure(i);
            continue;
        }
        
        updateIntermittent(i, dt);
        updateProgressive(i, dt);
        
        if (cfg.mode == FailureMode::INTERMITTENT && !state_.intermittent_on[i]) {
            continue;
        }
        
        auto* effect = getEffect(cfg.type);
        if (effect) {
            commanded_rpm[i] = effect->apply(commanded_rpm[i], i, dt, state_);
        }
    }
}

bool ActuatorFailureInjector::hasActiveFailure() const noexcept {
    for (int i = 0; i < 4; ++i) {
        if (state_.active[i]) return true;
    }
    return false;
}

bool ActuatorFailureInjector::hasActiveFailure(int motor_id) const noexcept {
    if (motor_id < 0 || motor_id >= 4) return false;
    return state_.active[motor_id];
}

FailureType ActuatorFailureInjector::getFailureType(int motor_id) const noexcept {
    if (motor_id < 0 || motor_id >= 4) return FailureType::NONE;
    return state_.configs[motor_id].type;
}

double ActuatorFailureInjector::getFailureSeverity(int motor_id) const noexcept {
    if (motor_id < 0 || motor_id >= 4) return 0.0;
    return state_.severity[motor_id];
}

void FailureScenarioGenerator::generateScenario(
    ActuatorFailureInjector& injector, Scenario scenario, double severity
) noexcept {
    injector.clearAllFailures();
    
    switch (scenario) {
        case Scenario::SINGLE_MOTOR_LOSS:
            singleMotorLoss(injector, 0);
            break;
        case Scenario::DUAL_MOTOR_LOSS:
            dualMotorLoss(injector, 0, 2);
            break;
        case Scenario::OPPOSITE_MOTOR_LOSS:
            oppositeMotorLoss(injector);
            break;
        case Scenario::ADJACENT_MOTOR_LOSS:
            adjacentMotorLoss(injector);
            break;
        case Scenario::PROGRESSIVE_DEGRADATION:
            progressiveDegradation(injector, severity * 0.01);
            break;
        case Scenario::RANDOM_INTERMITTENT:
            randomIntermittent(injector, severity * 0.1);
            break;
        case Scenario::THERMAL_RUNWAY:
            thermalRunaway(injector, 0, severity * 0.02);
            break;
        case Scenario::CASCADING_FAILURE:
            cascadingFailure(injector, 2.0 / severity);
            break;
    }
}

void FailureScenarioGenerator::singleMotorLoss(ActuatorFailureInjector& injector, int motor_id) noexcept {
    injector.injectFailure(motor_id, FailureType::COMPLETE_LOSS, 1.0);
}

void FailureScenarioGenerator::dualMotorLoss(ActuatorFailureInjector& injector, int motor1, int motor2) noexcept {
    injector.injectFailure(motor1, FailureType::COMPLETE_LOSS, 1.0);
    injector.injectFailure(motor2, FailureType::COMPLETE_LOSS, 1.0);
}

void FailureScenarioGenerator::oppositeMotorLoss(ActuatorFailureInjector& injector) noexcept {
    injector.injectFailure(0, FailureType::COMPLETE_LOSS, 1.0);
    injector.injectFailure(2, FailureType::COMPLETE_LOSS, 1.0);
}

void FailureScenarioGenerator::adjacentMotorLoss(ActuatorFailureInjector& injector) noexcept {
    injector.injectFailure(0, FailureType::COMPLETE_LOSS, 1.0);
    injector.injectFailure(1, FailureType::COMPLETE_LOSS, 1.0);
}

void FailureScenarioGenerator::progressiveDegradation(ActuatorFailureInjector& injector, double rate) noexcept {
    for (int i = 0; i < 4; ++i) {
        FailureConfig cfg;
        cfg.motor_id = i;
        cfg.type = FailureType::REDUCED_POWER;
        cfg.mode = FailureMode::PROGRESSIVE;
        cfg.severity = 0.0;
        cfg.progressive_rate = rate * (1.0 + i * 0.2);
        injector.injectFailure(cfg);
    }
}

void FailureScenarioGenerator::randomIntermittent(ActuatorFailureInjector& injector, double probability) noexcept {
    for (int i = 0; i < 4; ++i) {
        FailureConfig cfg;
        cfg.motor_id = i;
        cfg.type = FailureType::RANDOM_CUTOUT;
        cfg.mode = FailureMode::PERMANENT;
        cfg.cutout_probability = probability;
        injector.injectFailure(cfg);
    }
}

void FailureScenarioGenerator::thermalRunaway(ActuatorFailureInjector& injector, int motor_id, double rate) noexcept {
    FailureConfig cfg;
    cfg.motor_id = motor_id;
    cfg.type = FailureType::THERMAL_DEGRADATION;
    cfg.mode = FailureMode::PERMANENT;
    cfg.thermal_rate = rate;
    injector.injectFailure(cfg);
}

void FailureScenarioGenerator::cascadingFailure(ActuatorFailureInjector& injector, double interval) noexcept {
    for (int i = 0; i < 4; ++i) {
        FailureConfig cfg;
        cfg.motor_id = i;
        cfg.type = FailureType::COMPLETE_LOSS;
        cfg.mode = FailureMode::PERMANENT;
        cfg.start_time = interval * i;
        injector.scheduleFailure(cfg, interval * i);
    }
}
