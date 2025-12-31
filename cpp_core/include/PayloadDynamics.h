#pragma once
#include "Types.h"
#include <array>
#include <cmath>
#include <algorithm>

struct PhysicsConstants {
    static constexpr double GRAVITY = 9.81;
    static constexpr double AIR_DENSITY = 1.225;
    static constexpr double MIN_DT = 1e-6;
    static constexpr double MAX_DT = 0.1;
    static constexpr double POSITION_EPSILON = 1e-9;
    static constexpr double VELOCITY_EPSILON = 1e-12;
    static constexpr double SAFE_NORM_EPSILON = 1e-8;
};

enum class CableState {
    SLACK,
    TENSIONED,
    STRETCHED
};

enum class IntegratorType {
    VERLET,
    RK4,
    XPBD
};

struct CableConfig {
    double rest_length = 1.0;
    double compliance = 0.0001;
    double damping = 0.1;
    double linear_density = 0.05;
    int num_segments = 8;
    double drag_coefficient = 1.2;
    double diameter = 0.005;
    bool enable_drag = true;
    bool enable_catenary = true;
    double max_strain = 0.15;
    int prestabilization_iterations = 50;
};

struct PayloadConfig {
    double mass = 0.5;
    Vec3 inertia = Vec3(0.01, 0.01, 0.01);
    double drag_area = 0.01;
    double drag_coefficient = 1.0;
    Vec3 center_of_mass = Vec3(0, 0, 0);
    double restitution = 0.3;
    double friction = 0.5;
};

struct CableParticle {
    Vec3 position;
    Vec3 prev_position;
    Vec3 velocity;
    Vec3 acceleration;
    double inv_mass;
    bool fixed;
    
    CableParticle() noexcept 
        : position(), prev_position(), velocity(), acceleration()
        , inv_mass(1.0), fixed(false) {}
};

struct DistanceConstraint {
    int p1;
    int p2;
    double rest_length;
    double compliance;
    double lambda;
    
    DistanceConstraint() noexcept : p1(0), p2(0), rest_length(1.0), compliance(0.0001), lambda(0) {}
    DistanceConstraint(int a, int b, double len, double comp) noexcept 
        : p1(a), p2(b), rest_length(len), compliance(comp), lambda(0) {}
};

struct GroundConstraint {
    int particle_index;
    double ground_height;
    double compliance;
    double friction;
    double restitution;
    
    GroundConstraint() noexcept 
        : particle_index(-1), ground_height(0), compliance(0.00001)
        , friction(0.5), restitution(0.3) {}
};

struct PayloadState {
    Vec3 position;
    Vec3 velocity;
    Quaternion orientation;
    Vec3 angular_velocity;
    
    PayloadState() noexcept 
        : position(), velocity(), orientation(), angular_velocity() {}
    
    PayloadState(const Vec3& pos, const Vec3& vel) noexcept
        : position(pos), velocity(vel), orientation(), angular_velocity() {}
};

struct CableForces {
    Vec3 tension_force;
    Vec3 damping_force;
    Vec3 drag_force;
    double tension_magnitude;
    double strain;
    CableState state;
    
    CableForces() noexcept 
        : tension_force(), damping_force(), drag_force()
        , tension_magnitude(0), strain(0), state(CableState::SLACK) {}
    
    Vec3 total() const noexcept {
        return tension_force + damping_force + drag_force;
    }
};

struct PayloadForces {
    Vec3 gravity;
    Vec3 cable_tension;
    Vec3 aerodynamic_drag;
    Vec3 ground_reaction;
    Vec3 total_force;
    Vec3 total_torque;
    
    PayloadForces() noexcept
        : gravity(), cable_tension(), aerodynamic_drag()
        , ground_reaction(), total_force(), total_torque() {}
};

struct DronePayloadCoupling {
    Vec3 force_on_drone;
    Vec3 torque_on_drone;
    std::array<CableForces, 4> cable_forces;
    double total_tension;
    bool payload_attached;
    double total_energy;
    
    DronePayloadCoupling() noexcept
        : force_on_drone(), torque_on_drone(), cable_forces()
        , total_tension(0), payload_attached(false), total_energy(0) {}
};

struct AttachmentPoint {
    Vec3 local_position;
    int cable_id;
    bool active;
    
    AttachmentPoint() noexcept : local_position(), cable_id(-1), active(true) {}
    AttachmentPoint(const Vec3& pos, int id) noexcept : local_position(pos), cable_id(id), active(true) {}
};

struct InputShaperSample {
    double timestamp;
    Vec3 command;
};

inline Vec3 safeNormalize(const Vec3& v) noexcept {
    double len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq < PhysicsConstants::SAFE_NORM_EPSILON * PhysicsConstants::SAFE_NORM_EPSILON) {
        return Vec3();
    }
    double inv_len = 1.0 / std::sqrt(len_sq);
    return Vec3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

inline double safeLength(const Vec3& v) noexcept {
    double len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq < PhysicsConstants::SAFE_NORM_EPSILON * PhysicsConstants::SAFE_NORM_EPSILON) {
        return 0.0;
    }
    return std::sqrt(len_sq);
}

class UnifiedXPBDSystem {
public:
    static constexpr int MAX_PARTICLES = 32;
    static constexpr int XPBD_ITERATIONS = 10;
    static constexpr int GROUND_ITERATIONS = 3;
    
    UnifiedXPBDSystem() noexcept;
    UnifiedXPBDSystem(const CableConfig& cable_cfg, const PayloadConfig& payload_cfg) noexcept;
    
    void initialize(const Vec3& anchor, const Vec3& payload_pos) noexcept;
    void prestabilize() noexcept;
    
    void step(
        const Vec3& anchor_pos,
        const Vec3& anchor_vel,
        double dt,
        const Vec3& wind = Vec3(),
        double ground_height = 0.0
    ) noexcept;
    
    Vec3 getAnchorForce() const noexcept;
    
    CableState getState() const noexcept { return state_; }
    double getTension() const noexcept { return tension_; }
    double getStrain() const noexcept { return strain_; }
    double getTotalLength() const noexcept;
    
    Vec3 getPayloadPosition() const noexcept;
    Vec3 getPayloadVelocity() const noexcept;
    
    const CableParticle& getParticle(int index) const noexcept { return particles_[index]; }
    int getParticleCount() const noexcept { return num_particles_; }
    int getPayloadParticleIndex() const noexcept { return num_particles_ - 1; }
    
    void setCableConfig(const CableConfig& config) noexcept { cable_config_ = config; }
    void setPayloadConfig(const PayloadConfig& config) noexcept;
    const CableConfig& getCableConfig() const noexcept { return cable_config_; }
    const PayloadConfig& getPayloadConfig() const noexcept { return payload_config_; }
    
    void reset() noexcept;
    
    double getKineticEnergy() const noexcept;
    double getPotentialEnergy(double ground_height = 0) const noexcept;
    double getTotalEnergy(double ground_height = 0) const noexcept;
    
private:
    CableConfig cable_config_;
    PayloadConfig payload_config_;
    
    std::array<CableParticle, MAX_PARTICLES> particles_;
    std::array<DistanceConstraint, MAX_PARTICLES - 1> distance_constraints_;
    GroundConstraint ground_constraint_;
    
    int num_particles_ = 0;
    int num_distance_constraints_ = 0;
    
    CableState state_ = CableState::SLACK;
    double tension_ = 0;
    double strain_ = 0;
    
    void applyExternalForces(double dt, const Vec3& wind) noexcept;
    void predictPositions(double dt) noexcept;
    void solveDistanceConstraints(double dt) noexcept;
    void solveGroundConstraints(double dt, double ground_height) noexcept;
    void updateVelocities(double dt) noexcept;
    void applyDamping(double dt) noexcept;
    
    Vec3 computeSegmentDrag(int seg_idx, const Vec3& wind) const noexcept;
};

class PayloadDynamics {
public:
    static constexpr int MAX_ATTACHMENTS = 4;
    static constexpr int SUB_STEPS = 4;
    
    PayloadDynamics() noexcept;
    PayloadDynamics(const PayloadConfig& payload_config, const CableConfig& cable_config) noexcept;
    
    void step(
        const Vec3& drone_position,
        const Quaternion& drone_orientation,
        const Vec3& drone_velocity,
        const Vec3& drone_angular_velocity,
        double dt,
        double air_density = PhysicsConstants::AIR_DENSITY
    ) noexcept;
    
    DronePayloadCoupling getCouplingForces() const noexcept;
    
    void attach(const Vec3& initial_offset = Vec3(0, 0, -1.0)) noexcept;
    void detach() noexcept;
    bool isAttached() const noexcept { return attached_; }
    
    void addAttachmentPoint(const Vec3& local_position) noexcept;
    void clearAttachmentPoints() noexcept;
    int getAttachmentCount() const noexcept { return attachment_count_; }
    
    PayloadState getPayloadState() const noexcept;
    const PayloadForces& getPayloadForces() const noexcept { return forces_; }
    
    void setPayloadConfig(const PayloadConfig& config) noexcept;
    void setCableConfig(const CableConfig& config) noexcept;
    const PayloadConfig& getPayloadConfig() const noexcept { return payload_config_; }
    const CableConfig& getCableConfig() const noexcept { return cable_config_; }
    
    void setGroundHeight(double height) noexcept { ground_height_ = height; }
    void setWindVelocity(const Vec3& wind) noexcept { wind_velocity_ = wind; }
    void setIntegrator(IntegratorType type) noexcept { integrator_ = type; }
    
    void reset() noexcept;
    void reset(const PayloadState& initial_state) noexcept;
    
    double getKineticEnergy() const noexcept;
    double getPotentialEnergy() const noexcept;
    double getCableEnergy() const noexcept;
    double getTotalEnergy() const noexcept;
    
    Vec3 getSwingAngle() const noexcept;
    double getCableAngleFromVertical() const noexcept;
    double getNaturalFrequency() const noexcept;
    
    const UnifiedXPBDSystem& getXPBDSystem(int index = 0) const noexcept { return xpbd_systems_[index]; }
    
private:
    PayloadForces forces_;
    PayloadConfig payload_config_;
    CableConfig cable_config_;
    
    std::array<AttachmentPoint, MAX_ATTACHMENTS> attachments_;
    std::array<UnifiedXPBDSystem, MAX_ATTACHMENTS> xpbd_systems_;
    int attachment_count_ = 0;
    
    bool attached_ = false;
    double ground_height_ = 0;
    Vec3 wind_velocity_;
    IntegratorType integrator_ = IntegratorType::XPBD;
    
    Vec3 prev_drone_pos_;
    Vec3 prev_drone_vel_;
    Quaternion prev_drone_ori_;
    Vec3 prev_drone_angvel_;
    
    Vec3 last_drone_pos_;
    Quaternion last_drone_ori_;
    
    Vec3 getAttachmentWorldPosition(
        int index,
        const Vec3& drone_position,
        const Quaternion& drone_orientation
    ) const noexcept;
    
    Vec3 getAttachmentWorldVelocity(
        int index,
        const Vec3& drone_velocity,
        const Vec3& drone_angular_velocity,
        const Quaternion& drone_orientation
    ) const noexcept;
    
    Vec3 interpolatePosition(const Vec3& p0, const Vec3& p1, double alpha) const noexcept;
    Vec3 interpolateVelocity(const Vec3& v0, const Vec3& v1, double alpha) const noexcept;
};

class SwingingPayloadController {
public:
    struct LQRGains {
        std::array<double, 4> K_pos = {2.0, 0.5, 1.0, 0.3};
        std::array<double, 4> K_swing = {3.0, 1.5, 2.5, 1.0};
        double max_compensation = 5.0;
        double damping_ratio = 0.7;
    };
    
    static constexpr int SHAPER_BUFFER_SIZE = 500;
    
    SwingingPayloadController() noexcept;
    explicit SwingingPayloadController(const LQRGains& gains) noexcept;
    
    Vec3 computeCompensation(
        const PayloadState& payload_state,
        const Vec3& drone_position,
        const Vec3& drone_velocity,
        const Vec3& target_position,
        double cable_length
    ) noexcept;
    
    Vec3 computeInputShaping(
        const Vec3& commanded_accel,
        double cable_length,
        double current_time
    ) noexcept;
    
    Vec3 computeEnergyBasedControl(
        const PayloadState& payload_state,
        const Vec3& drone_position,
        double cable_length,
        double target_energy = 0
    ) noexcept;
    
    void setGains(const LQRGains& gains) noexcept { gains_ = gains; }
    const LQRGains& getGains() const noexcept { return gains_; }
    
    void reset() noexcept;
    
    static PayloadState estimatePayloadFromCableAngles(
        double theta_x,
        double theta_y,
        double cable_length,
        const Vec3& drone_position,
        const Vec3& drone_velocity
    ) noexcept;
    
    Vec3 computeCompensationFromAngles(
        double theta_x,
        double theta_y,
        double theta_x_rate,
        double theta_y_rate,
        const Vec3& drone_position,
        const Vec3& drone_velocity,
        double cable_length
    ) noexcept;
    
private:
    LQRGains gains_;
    
    std::array<InputShaperSample, SHAPER_BUFFER_SIZE> shaper_buffer_;
    double last_anchor_z_ = 0;
    int shaper_head_ = 0;
    int shaper_count_ = 0;
    
    double computeNaturalFrequency(double cable_length) const noexcept;
    Vec3 getDelayedCommand(double delay_time, double current_time) const noexcept;
};

using Cable = UnifiedXPBDSystem;
using XPBDCable = UnifiedXPBDSystem;
