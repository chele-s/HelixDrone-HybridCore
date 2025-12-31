#include "PayloadDynamics.h"

UnifiedXPBDSystem::UnifiedXPBDSystem() noexcept 
    : cable_config_(), payload_config_() {
    reset();
}

UnifiedXPBDSystem::UnifiedXPBDSystem(const CableConfig& cable_cfg, const PayloadConfig& payload_cfg) noexcept
    : cable_config_(cable_cfg), payload_config_(payload_cfg) {
    reset();
}

void UnifiedXPBDSystem::reset() noexcept {
    num_particles_ = 0;
    num_distance_constraints_ = 0;
    state_ = CableState::SLACK;
    tension_ = 0;
    strain_ = 0;
    for (int i = 0; i < MAX_PARTICLES; ++i) {
        particles_[i] = CableParticle();
    }
    ground_constraint_ = GroundConstraint();
}

void UnifiedXPBDSystem::setPayloadConfig(const PayloadConfig& config) noexcept {
    payload_config_ = config;
    if (num_particles_ > 0) {
        particles_[num_particles_ - 1].inv_mass = 1.0 / payload_config_.mass;
    }
}

void UnifiedXPBDSystem::initialize(const Vec3& anchor, const Vec3& payload_pos) noexcept {
    num_particles_ = std::min(cable_config_.num_segments + 1, MAX_PARTICLES - 1) + 1;
    num_distance_constraints_ = num_particles_ - 1;
    
    double segment_length = cable_config_.rest_length / (num_particles_ - 1);
    double segment_mass = cable_config_.linear_density * segment_length;
    
    for (int i = 0; i < num_particles_; ++i) {
        double t = static_cast<double>(i) / (num_particles_ - 1);
        
        particles_[i].position.x = anchor.x + t * (payload_pos.x - anchor.x);
        particles_[i].position.y = anchor.y + t * (payload_pos.y - anchor.y);
        particles_[i].position.z = anchor.z + t * (payload_pos.z - anchor.z);
        
        if (cable_config_.enable_catenary && num_particles_ > 2) {
            double sag = 0.05 * cable_config_.rest_length * std::sin(t * 3.14159265359);
            particles_[i].position.z -= sag;
        }
        
        particles_[i].prev_position = particles_[i].position;
        particles_[i].velocity = Vec3();
        particles_[i].acceleration = Vec3();
        
        if (i == 0) {
            particles_[i].inv_mass = 0.0;
            particles_[i].fixed = true;
        } else if (i == num_particles_ - 1) {
            particles_[i].inv_mass = 1.0 / payload_config_.mass;
            particles_[i].fixed = false;
        } else {
            particles_[i].inv_mass = 1.0 / segment_mass;
            particles_[i].fixed = false;
        }
    }
    
    for (int i = 0; i < num_distance_constraints_; ++i) {
        distance_constraints_[i] = DistanceConstraint(i, i + 1, segment_length, cable_config_.compliance);
    }
    
    ground_constraint_.particle_index = num_particles_ - 1;
    ground_constraint_.ground_height = 0;
    ground_constraint_.friction = payload_config_.friction;
    ground_constraint_.restitution = payload_config_.restitution;
    
    prestabilize();
}

void UnifiedXPBDSystem::prestabilize() noexcept {
    if (num_particles_ < 2) return;
    
    double prestab_dt = 0.01;
    
    for (int iter = 0; iter < cable_config_.prestabilization_iterations; ++iter) {
        for (int i = 0; i < num_distance_constraints_; ++i) {
            distance_constraints_[i].lambda = 0;
        }
        
        for (int sub = 0; sub < XPBD_ITERATIONS; ++sub) {
            solveDistanceConstraints(prestab_dt);
        }
        
        for (int i = 1; i < num_particles_; ++i) {
            particles_[i].prev_position = particles_[i].position;
            particles_[i].velocity = Vec3();
        }
    }
}

double UnifiedXPBDSystem::getTotalLength() const noexcept {
    double length = 0;
    for (int i = 0; i < num_particles_ - 1; ++i) {
        Vec3 d;
        d.x = particles_[i + 1].position.x - particles_[i].position.x;
        d.y = particles_[i + 1].position.y - particles_[i].position.y;
        d.z = particles_[i + 1].position.z - particles_[i].position.z;
        length += safeLength(d);
    }
    return length;
}

Vec3 UnifiedXPBDSystem::getPayloadPosition() const noexcept {
    if (num_particles_ > 0) {
        return particles_[num_particles_ - 1].position;
    }
    return Vec3();
}

Vec3 UnifiedXPBDSystem::getPayloadVelocity() const noexcept {
    if (num_particles_ > 0) {
        return particles_[num_particles_ - 1].velocity;
    }
    return Vec3();
}

double UnifiedXPBDSystem::getKineticEnergy() const noexcept {
    double ke = 0;
    for (int i = 1; i < num_particles_; ++i) {
        if (particles_[i].inv_mass > PhysicsConstants::VELOCITY_EPSILON) {
            double mass = 1.0 / particles_[i].inv_mass;
            double v_sq = particles_[i].velocity.x * particles_[i].velocity.x +
                         particles_[i].velocity.y * particles_[i].velocity.y +
                         particles_[i].velocity.z * particles_[i].velocity.z;
            ke += 0.5 * mass * v_sq;
        }
    }
    return ke;
}

double UnifiedXPBDSystem::getPotentialEnergy(double ground_height) const noexcept {
    double pe = 0;
    double anchor_z = (num_particles_ > 0) ? particles_[0].position.z : ground_height;
    for (int i = 1; i < num_particles_; ++i) {
        if (particles_[i].inv_mass > PhysicsConstants::VELOCITY_EPSILON) {
            double mass = 1.0 / particles_[i].inv_mass;
            pe += mass * PhysicsConstants::GRAVITY * (particles_[i].position.z - anchor_z);
        }
    }
    return pe;
}

double UnifiedXPBDSystem::getTotalEnergy(double ground_height) const noexcept {
    return getKineticEnergy() + getPotentialEnergy(ground_height);
}

Vec3 UnifiedXPBDSystem::computeSegmentDrag(int seg_idx, const Vec3& wind) const noexcept {
    if (!cable_config_.enable_drag || seg_idx < 0 || seg_idx >= num_particles_ - 1) {
        return Vec3();
    }
    
    const CableParticle& p1 = particles_[seg_idx];
    const CableParticle& p2 = particles_[seg_idx + 1];
    
    Vec3 seg;
    seg.x = p2.position.x - p1.position.x;
    seg.y = p2.position.y - p1.position.y;
    seg.z = p2.position.z - p1.position.z;
    
    double seg_len = safeLength(seg);
    if (seg_len < PhysicsConstants::POSITION_EPSILON) return Vec3();
    
    Vec3 seg_dir = safeNormalize(seg);
    
    Vec3 avg_vel;
    avg_vel.x = (p1.velocity.x + p2.velocity.x) * 0.5 - wind.x;
    avg_vel.y = (p1.velocity.y + p2.velocity.y) * 0.5 - wind.y;
    avg_vel.z = (p1.velocity.z + p2.velocity.z) * 0.5 - wind.z;
    
    double dot = avg_vel.x * seg_dir.x + avg_vel.y * seg_dir.y + avg_vel.z * seg_dir.z;
    Vec3 perp_vel;
    perp_vel.x = avg_vel.x - seg_dir.x * dot;
    perp_vel.y = avg_vel.y - seg_dir.y * dot;
    perp_vel.z = avg_vel.z - seg_dir.z * dot;
    
    double perp_speed = safeLength(perp_vel);
    if (perp_speed < PhysicsConstants::VELOCITY_EPSILON) return Vec3();
    
    double area = cable_config_.diameter * seg_len;
    double drag_mag = 0.5 * PhysicsConstants::AIR_DENSITY * cable_config_.drag_coefficient * area * perp_speed * perp_speed;
    
    Vec3 drag_dir = safeNormalize(perp_vel);
    return Vec3(-drag_dir.x * drag_mag, -drag_dir.y * drag_mag, -drag_dir.z * drag_mag);
}

void UnifiedXPBDSystem::applyExternalForces(double dt, const Vec3& wind) noexcept {
    for (int i = 1; i < num_particles_; ++i) {
        if (particles_[i].fixed) continue;
        
        particles_[i].acceleration = Vec3(0, 0, -PhysicsConstants::GRAVITY);
        
        if (i < num_particles_ - 1) {
            Vec3 drag = computeSegmentDrag(i - 1, wind);
            if (particles_[i].inv_mass > PhysicsConstants::VELOCITY_EPSILON) {
                double inv_m = particles_[i].inv_mass;
                particles_[i].acceleration.x += drag.x * inv_m;
                particles_[i].acceleration.y += drag.y * inv_m;
                particles_[i].acceleration.z += drag.z * inv_m;
            }
        }
        
        if (i == num_particles_ - 1 && cable_config_.enable_drag) {
            Vec3 vel = particles_[i].velocity;
            Vec3 rel_vel;
            rel_vel.x = vel.x - wind.x;
            rel_vel.y = vel.y - wind.y;
            rel_vel.z = vel.z - wind.z;
            
            double speed = safeLength(rel_vel);
            if (speed > PhysicsConstants::VELOCITY_EPSILON) {
                double drag_mag = 0.5 * PhysicsConstants::AIR_DENSITY * 
                                 payload_config_.drag_coefficient * 
                                 payload_config_.drag_area * speed * speed;
                Vec3 drag_dir = safeNormalize(rel_vel);
                double inv_m = particles_[i].inv_mass;
                particles_[i].acceleration.x -= drag_dir.x * drag_mag * inv_m;
                particles_[i].acceleration.y -= drag_dir.y * drag_mag * inv_m;
                particles_[i].acceleration.z -= drag_dir.z * drag_mag * inv_m;
            }
        }
    }
}

void UnifiedXPBDSystem::predictPositions(double dt) noexcept {
    for (int i = 1; i < num_particles_; ++i) {
        if (particles_[i].fixed) continue;
        
        particles_[i].velocity.x += particles_[i].acceleration.x * dt;
        particles_[i].velocity.y += particles_[i].acceleration.y * dt;
        particles_[i].velocity.z += particles_[i].acceleration.z * dt;
        
        particles_[i].prev_position = particles_[i].position;
        
        particles_[i].position.x += particles_[i].velocity.x * dt;
        particles_[i].position.y += particles_[i].velocity.y * dt;
        particles_[i].position.z += particles_[i].velocity.z * dt;
    }
}

void UnifiedXPBDSystem::solveDistanceConstraints(double dt) noexcept {
    double dt_sq = dt * dt;
    double beta = cable_config_.damping * dt;
    
    for (int i = 0; i < num_distance_constraints_; ++i) {
        DistanceConstraint& c = distance_constraints_[i];
        CableParticle& p1 = particles_[c.p1];
        CableParticle& p2 = particles_[c.p2];
        
        Vec3 diff;
        diff.x = p2.position.x - p1.position.x;
        diff.y = p2.position.y - p1.position.y;
        diff.z = p2.position.z - p1.position.z;
        
        double dist = safeLength(diff);
        if (dist < PhysicsConstants::POSITION_EPSILON) continue;
        
        double C = dist - c.rest_length;
        
        Vec3 grad = safeNormalize(diff);
        
        Vec3 rel_vel;
        rel_vel.x = p2.velocity.x - p1.velocity.x;
        rel_vel.y = p2.velocity.y - p1.velocity.y;
        rel_vel.z = p2.velocity.z - p1.velocity.z;
        double vel_proj = rel_vel.x * grad.x + rel_vel.y * grad.y + rel_vel.z * grad.z;
        
        double w = p1.inv_mass + p2.inv_mass;
        if (w < PhysicsConstants::VELOCITY_EPSILON) continue;
        
        double alpha = c.compliance / dt_sq;
        double delta_lambda = (-C - alpha * c.lambda - beta * vel_proj) / (w + alpha);
        c.lambda += delta_lambda;
        
        if (!p1.fixed) {
            p1.position.x -= grad.x * delta_lambda * p1.inv_mass;
            p1.position.y -= grad.y * delta_lambda * p1.inv_mass;
            p1.position.z -= grad.z * delta_lambda * p1.inv_mass;
        }
        if (!p2.fixed) {
            p2.position.x += grad.x * delta_lambda * p2.inv_mass;
            p2.position.y += grad.y * delta_lambda * p2.inv_mass;
            p2.position.z += grad.z * delta_lambda * p2.inv_mass;
        }
    }
    
    double total_length = getTotalLength();
    strain_ = (total_length - cable_config_.rest_length) / cable_config_.rest_length;
    
    if (strain_ < -0.01) {
        state_ = CableState::SLACK;
        tension_ = 0;
    } else if (strain_ > cable_config_.max_strain) {
        state_ = CableState::STRETCHED;
        tension_ = std::abs(distance_constraints_[0].lambda) / dt_sq;
    } else {
        state_ = CableState::TENSIONED;
        tension_ = std::abs(distance_constraints_[0].lambda) / dt_sq;
    }
}

void UnifiedXPBDSystem::solveGroundConstraints(double dt, double ground_height) noexcept {
    double dt_sq = dt * dt;
    
    for (int i = 1; i < num_particles_; ++i) {
        if (particles_[i].fixed) continue;
        
        double penetration = ground_height - particles_[i].position.z;
        if (penetration <= 0) continue;
        
        double C = -penetration;
        Vec3 grad(0, 0, 1);
        
        double w = particles_[i].inv_mass;
        if (w < PhysicsConstants::VELOCITY_EPSILON) continue;
        
        double alpha = ground_constraint_.compliance / dt_sq;
        double delta_lambda = -C / (w + alpha);
        
        particles_[i].position.z += delta_lambda * w;
        
        double vz = (particles_[i].position.z - particles_[i].prev_position.z) / dt;
        if (vz < 0) {
            particles_[i].position.z = particles_[i].prev_position.z - vz * ground_constraint_.restitution * dt;
        }
        
        double friction_impulse = ground_constraint_.friction * std::abs(delta_lambda);
        double vx = (particles_[i].position.x - particles_[i].prev_position.x) / dt;
        double vy = (particles_[i].position.y - particles_[i].prev_position.y) / dt;
        double horiz_speed = std::sqrt(vx * vx + vy * vy);
        
        if (horiz_speed > PhysicsConstants::VELOCITY_EPSILON) {
            double max_friction = horiz_speed * dt;
            double applied_friction = std::min(friction_impulse * w, max_friction);
            double scale = 1.0 - applied_friction / (horiz_speed * dt);
            scale = std::max(0.0, scale);
            
            Vec3 new_pos;
            new_pos.x = particles_[i].prev_position.x + vx * scale * dt;
            new_pos.y = particles_[i].prev_position.y + vy * scale * dt;
            particles_[i].position.x = new_pos.x;
            particles_[i].position.y = new_pos.y;
        }
    }
}

void UnifiedXPBDSystem::updateVelocities(double dt) noexcept {
    double inv_dt = 1.0 / dt;
    for (int i = 1; i < num_particles_; ++i) {
        if (particles_[i].fixed) continue;
        particles_[i].velocity.x = (particles_[i].position.x - particles_[i].prev_position.x) * inv_dt;
        particles_[i].velocity.y = (particles_[i].position.y - particles_[i].prev_position.y) * inv_dt;
        particles_[i].velocity.z = (particles_[i].position.z - particles_[i].prev_position.z) * inv_dt;
    }
}


void UnifiedXPBDSystem::step(const Vec3& anchor_pos, const Vec3& anchor_vel, double dt, const Vec3& wind, double ground_height) noexcept {
    if (num_particles_ < 2) return;
    
    particles_[0].position = anchor_pos;
    particles_[0].velocity = anchor_vel;
    
    for (int i = 0; i < num_distance_constraints_; ++i) {
        distance_constraints_[i].lambda = 0;
    }
    
    applyExternalForces(dt, wind);
    predictPositions(dt);
    
    for (int iter = 0; iter < XPBD_ITERATIONS; ++iter) {
        solveDistanceConstraints(dt);
    }
    
    for (int iter = 0; iter < GROUND_ITERATIONS; ++iter) {
        solveGroundConstraints(dt, ground_height);
    }
    
    updateVelocities(dt);
}

Vec3 UnifiedXPBDSystem::getAnchorForce() const noexcept {
    if (num_distance_constraints_ == 0 || tension_ < PhysicsConstants::VELOCITY_EPSILON) {
        return Vec3();
    }
    
    Vec3 dir;
    dir.x = particles_[1].position.x - particles_[0].position.x;
    dir.y = particles_[1].position.y - particles_[0].position.y;
    dir.z = particles_[1].position.z - particles_[0].position.z;
    
    Vec3 normalized = safeNormalize(dir);
    return Vec3(normalized.x * tension_, normalized.y * tension_, normalized.z * tension_);
}

PayloadDynamics::PayloadDynamics() noexcept
    : forces_(), payload_config_(), cable_config_()
    , attachments_(), xpbd_systems_(), attachment_count_(0)
    , attached_(false), ground_height_(0), wind_velocity_()
    , integrator_(IntegratorType::XPBD)
    , prev_drone_pos_(), prev_drone_vel_(), prev_drone_ori_(), prev_drone_angvel_()
    , last_drone_pos_(), last_drone_ori_() {
    addAttachmentPoint(Vec3(0, 0, 0));
}

PayloadDynamics::PayloadDynamics(const PayloadConfig& payload_config, const CableConfig& cable_config) noexcept
    : forces_(), payload_config_(payload_config), cable_config_(cable_config)
    , attachments_(), xpbd_systems_(), attachment_count_(0)
    , attached_(false), ground_height_(0), wind_velocity_()
    , integrator_(IntegratorType::XPBD)
    , prev_drone_pos_(), prev_drone_vel_(), prev_drone_ori_(), prev_drone_angvel_()
    , last_drone_pos_(), last_drone_ori_() {
    for (int i = 0; i < MAX_ATTACHMENTS; ++i) {
        xpbd_systems_[i] = UnifiedXPBDSystem(cable_config, payload_config);
    }
    addAttachmentPoint(Vec3(0, 0, 0));
}

void PayloadDynamics::setPayloadConfig(const PayloadConfig& config) noexcept {
    payload_config_ = config;
    for (int i = 0; i < MAX_ATTACHMENTS; ++i) {
        xpbd_systems_[i].setPayloadConfig(config);
    }
}

void PayloadDynamics::setCableConfig(const CableConfig& config) noexcept {
    cable_config_ = config;
    for (int i = 0; i < MAX_ATTACHMENTS; ++i) {
        xpbd_systems_[i].setCableConfig(config);
    }
}

void PayloadDynamics::addAttachmentPoint(const Vec3& local_position) noexcept {
    if (attachment_count_ < MAX_ATTACHMENTS) {
        attachments_[attachment_count_] = AttachmentPoint(local_position, attachment_count_);
        attachment_count_++;
    }
}

void PayloadDynamics::clearAttachmentPoints() noexcept {
    attachment_count_ = 0;
    for (int i = 0; i < MAX_ATTACHMENTS; ++i) {
        attachments_[i].active = false;
    }
}

void PayloadDynamics::attach(const Vec3& initial_offset) noexcept {
    attached_ = true;
    
    Vec3 payload_pos;
    payload_pos.x = last_drone_pos_.x + initial_offset.x;
    payload_pos.y = last_drone_pos_.y + initial_offset.y;
    payload_pos.z = last_drone_pos_.z + initial_offset.z;
    
    for (int i = 0; i < attachment_count_; ++i) {
        Vec3 anchor = getAttachmentWorldPosition(i, last_drone_pos_, last_drone_ori_);
        xpbd_systems_[i] = UnifiedXPBDSystem(cable_config_, payload_config_);
        xpbd_systems_[i].initialize(anchor, payload_pos);
    }
    
    prev_drone_pos_ = last_drone_pos_;
    prev_drone_vel_ = Vec3();
    prev_drone_ori_ = last_drone_ori_;
    prev_drone_angvel_ = Vec3();
}

void PayloadDynamics::detach() noexcept {
    attached_ = false;
}

void PayloadDynamics::reset() noexcept {
    forces_ = PayloadForces();
    attached_ = false;
    
    for (int i = 0; i < MAX_ATTACHMENTS; ++i) {
        xpbd_systems_[i].reset();
    }
}

void PayloadDynamics::reset(const PayloadState& initial_state) noexcept {
    forces_ = PayloadForces();
    attached_ = true;
    
    for (int i = 0; i < attachment_count_; ++i) {
        Vec3 anchor = getAttachmentWorldPosition(i, last_drone_pos_, last_drone_ori_);
        xpbd_systems_[i] = UnifiedXPBDSystem(cable_config_, payload_config_);
        xpbd_systems_[i].initialize(anchor, initial_state.position);
    }
}

Vec3 PayloadDynamics::getAttachmentWorldPosition(
    int index,
    const Vec3& drone_position,
    const Quaternion& drone_orientation
) const noexcept {
    if (index < 0 || index >= attachment_count_) return drone_position;
    
    Vec3 local_pos = attachments_[index].local_position;
    Vec3 rotated = drone_orientation.rotate(local_pos);
    return Vec3(drone_position.x + rotated.x, drone_position.y + rotated.y, drone_position.z + rotated.z);
}

Vec3 PayloadDynamics::getAttachmentWorldVelocity(
    int index,
    const Vec3& drone_velocity,
    const Vec3& drone_angular_velocity,
    const Quaternion& drone_orientation
) const noexcept {
    if (index < 0 || index >= attachment_count_) return drone_velocity;
    
    Vec3 local_pos = attachments_[index].local_position;
    Vec3 rotated = drone_orientation.rotate(local_pos);
    Vec3 angular_contribution = drone_angular_velocity.cross(rotated);
    
    return Vec3(drone_velocity.x + angular_contribution.x, 
                drone_velocity.y + angular_contribution.y, 
                drone_velocity.z + angular_contribution.z);
}

Vec3 PayloadDynamics::interpolatePosition(const Vec3& p0, const Vec3& p1, double alpha) const noexcept {
    double inv_alpha = 1.0 - alpha;
    return Vec3(p0.x * inv_alpha + p1.x * alpha,
                p0.y * inv_alpha + p1.y * alpha,
                p0.z * inv_alpha + p1.z * alpha);
}

Vec3 PayloadDynamics::interpolateVelocity(const Vec3& v0, const Vec3& v1, double alpha) const noexcept {
    double inv_alpha = 1.0 - alpha;
    return Vec3(v0.x * inv_alpha + v1.x * alpha,
                v0.y * inv_alpha + v1.y * alpha,
                v0.z * inv_alpha + v1.z * alpha);
}

void PayloadDynamics::step(
    const Vec3& drone_position,
    const Quaternion& drone_orientation,
    const Vec3& drone_velocity,
    const Vec3& drone_angular_velocity,
    double dt,
    double air_density
) noexcept {
    last_drone_pos_ = drone_position;
    last_drone_ori_ = drone_orientation;
    
    if (!attached_) {
        prev_drone_pos_ = drone_position;
        prev_drone_vel_ = drone_velocity;
        prev_drone_ori_ = drone_orientation;
        prev_drone_angvel_ = drone_angular_velocity;
        return;
    }
    
    dt = std::clamp(dt, PhysicsConstants::MIN_DT, PhysicsConstants::MAX_DT);
    
    double sub_dt = dt / SUB_STEPS;
    
    Vec3 total_anchor_force;
    
    for (int step = 0; step < SUB_STEPS; ++step) {
        double alpha = static_cast<double>(step + 1) / SUB_STEPS;
        
        Vec3 interp_pos = interpolatePosition(prev_drone_pos_, drone_position, alpha);
        Vec3 interp_vel = interpolateVelocity(prev_drone_vel_, drone_velocity, alpha);
        
        Quaternion interp_ori = Quaternion::slerp(prev_drone_ori_, drone_orientation, alpha);
        
        Vec3 interp_angvel = interpolateVelocity(prev_drone_angvel_, drone_angular_velocity, alpha);
        
        for (int i = 0; i < attachment_count_; ++i) {
            if (!attachments_[i].active) continue;
            
            Vec3 anchor_pos = getAttachmentWorldPosition(i, interp_pos, interp_ori);
            Vec3 anchor_vel = getAttachmentWorldVelocity(i, interp_vel, interp_angvel, interp_ori);
            
            xpbd_systems_[i].step(anchor_pos, anchor_vel, sub_dt, wind_velocity_, ground_height_);
        }
    }
    
    prev_drone_pos_ = drone_position;
    prev_drone_vel_ = drone_velocity;
    prev_drone_ori_ = drone_orientation;
    prev_drone_angvel_ = drone_angular_velocity;
    
    forces_.gravity = Vec3(0, 0, -PhysicsConstants::GRAVITY * payload_config_.mass);
    
    Vec3 cable_force;
    for (int i = 0; i < attachment_count_; ++i) {
        if (!attachments_[i].active) continue;
        Vec3 f = xpbd_systems_[i].getAnchorForce();
        cable_force.x -= f.x;
        cable_force.y -= f.y;
        cable_force.z -= f.z;
    }
    forces_.cable_tension = cable_force;
    
    forces_.total_force.x = forces_.gravity.x + forces_.cable_tension.x;
    forces_.total_force.y = forces_.gravity.y + forces_.cable_tension.y;
    forces_.total_force.z = forces_.gravity.z + forces_.cable_tension.z;
}

PayloadState PayloadDynamics::getPayloadState() const noexcept {
    PayloadState state;
    if (attachment_count_ > 0 && attached_) {
        state.position = xpbd_systems_[0].getPayloadPosition();
        state.velocity = xpbd_systems_[0].getPayloadVelocity();
    }
    return state;
}

DronePayloadCoupling PayloadDynamics::getCouplingForces() const noexcept {
    DronePayloadCoupling coupling;
    coupling.payload_attached = attached_;
    
    if (!attached_) return coupling;
    
    double total_tension = 0;
    
    for (int i = 0; i < attachment_count_; ++i) {
        if (!attachments_[i].active) continue;
        
        Vec3 force = xpbd_systems_[i].getAnchorForce();
        coupling.cable_forces[i].tension_force = force;
        coupling.cable_forces[i].tension_magnitude = xpbd_systems_[i].getTension();
        coupling.cable_forces[i].state = xpbd_systems_[i].getState();
        coupling.cable_forces[i].strain = xpbd_systems_[i].getStrain();
        
        coupling.force_on_drone.x += force.x;
        coupling.force_on_drone.y += force.y;
        coupling.force_on_drone.z += force.z;
        total_tension += xpbd_systems_[i].getTension();
        
        Vec3 attachment_world = getAttachmentWorldPosition(i, last_drone_pos_, last_drone_ori_);
        Vec3 r;
        r.x = attachment_world.x - last_drone_pos_.x;
        r.y = attachment_world.y - last_drone_pos_.y;
        r.z = attachment_world.z - last_drone_pos_.z;
        Vec3 torque = r.cross(force);
        coupling.torque_on_drone.x += torque.x;
        coupling.torque_on_drone.y += torque.y;
        coupling.torque_on_drone.z += torque.z;
    }
    
    coupling.total_tension = total_tension;
    coupling.total_energy = getTotalEnergy();
    
    return coupling;
}

double PayloadDynamics::getKineticEnergy() const noexcept {
    if (attachment_count_ > 0 && attached_) {
        return xpbd_systems_[0].getKineticEnergy();
    }
    return 0;
}

double PayloadDynamics::getPotentialEnergy() const noexcept {
    if (attachment_count_ > 0 && attached_) {
        return xpbd_systems_[0].getPotentialEnergy(ground_height_);
    }
    return 0;
}

double PayloadDynamics::getCableEnergy() const noexcept {
    double energy = 0;
    for (int i = 0; i < attachment_count_; ++i) {
        if (!attachments_[i].active) continue;
        energy += xpbd_systems_[i].getTotalEnergy(ground_height_);
    }
    return energy;
}

double PayloadDynamics::getTotalEnergy() const noexcept {
    return getCableEnergy();
}

Vec3 PayloadDynamics::getSwingAngle() const noexcept {
    if (!attached_ || attachment_count_ == 0) return Vec3();
    
    Vec3 payload_pos = xpbd_systems_[0].getPayloadPosition();
    Vec3 cable_vec;
    cable_vec.x = payload_pos.x - last_drone_pos_.x;
    cable_vec.y = payload_pos.y - last_drone_pos_.y;
    cable_vec.z = payload_pos.z - last_drone_pos_.z;
    
    double length = safeLength(cable_vec);
    if (length < PhysicsConstants::POSITION_EPSILON) return Vec3();
    
    Vec3 vertical(0, 0, -1);
    Vec3 normalized_cable = safeNormalize(cable_vec);
    
    double dot = normalized_cable.x * vertical.x + normalized_cable.y * vertical.y + normalized_cable.z * vertical.z;
    double angle_from_vertical = std::acos(std::clamp(dot, -1.0, 1.0));
    
    double swing_x = std::atan2(cable_vec.y, -cable_vec.z);
    double swing_y = std::atan2(cable_vec.x, -cable_vec.z);
    
    return Vec3(swing_x, swing_y, angle_from_vertical);
}

double PayloadDynamics::getCableAngleFromVertical() const noexcept {
    Vec3 swing = getSwingAngle();
    return swing.z;
}

double PayloadDynamics::getNaturalFrequency() const noexcept {
    double length = cable_config_.rest_length;
    if (length < PhysicsConstants::POSITION_EPSILON) return 0;
    return std::sqrt(PhysicsConstants::GRAVITY / length);
}

SwingingPayloadController::SwingingPayloadController() noexcept : gains_() {
    reset();
}

SwingingPayloadController::SwingingPayloadController(const LQRGains& gains) noexcept : gains_(gains) {
    reset();
}

void SwingingPayloadController::reset() noexcept {
    shaper_head_ = 0;
    shaper_count_ = 0;
    for (int i = 0; i < SHAPER_BUFFER_SIZE; ++i) {
        shaper_buffer_[i] = InputShaperSample();
    }
}

double SwingingPayloadController::computeNaturalFrequency(double cable_length) const noexcept {
    if (cable_length < PhysicsConstants::POSITION_EPSILON) return 3.13;
    return std::sqrt(PhysicsConstants::GRAVITY / cable_length);
}

Vec3 SwingingPayloadController::getDelayedCommand(double delay_time, double current_time) const noexcept {
    double target_time = current_time - delay_time;
    
    if (shaper_count_ < 2) {
        if (shaper_count_ == 1) {
            return shaper_buffer_[(shaper_head_ - 1 + SHAPER_BUFFER_SIZE) % SHAPER_BUFFER_SIZE].command;
        }
        return Vec3();
    }
    
    int before_idx = -1;
    int after_idx = -1;
    double before_time = -1e9;
    double after_time = 1e9;
    
    for (int i = 0; i < shaper_count_; ++i) {
        int idx = (shaper_head_ - 1 - i + SHAPER_BUFFER_SIZE) % SHAPER_BUFFER_SIZE;
        double t = shaper_buffer_[idx].timestamp;
        
        if (t <= target_time && t > before_time) {
            before_time = t;
            before_idx = idx;
        }
        if (t >= target_time && t < after_time) {
            after_time = t;
            after_idx = idx;
        }
    }
    
    if (before_idx < 0 && after_idx < 0) {
        return Vec3();
    }
    
    if (before_idx < 0) {
        return shaper_buffer_[after_idx].command;
    }
    
    if (after_idx < 0 || before_idx == after_idx) {
        return shaper_buffer_[before_idx].command;
    }
    
    double dt = after_time - before_time;
    if (dt < PhysicsConstants::VELOCITY_EPSILON) {
        return shaper_buffer_[before_idx].command;
    }
    
    double alpha = (target_time - before_time) / dt;
    alpha = std::clamp(alpha, 0.0, 1.0);
    
    const Vec3& v0 = shaper_buffer_[before_idx].command;
    const Vec3& v1 = shaper_buffer_[after_idx].command;
    
    return Vec3(
        v0.x * (1.0 - alpha) + v1.x * alpha,
        v0.y * (1.0 - alpha) + v1.y * alpha,
        v0.z * (1.0 - alpha) + v1.z * alpha
    );
}

Vec3 SwingingPayloadController::computeCompensation(
    const PayloadState& payload_state,
    const Vec3& drone_position,
    const Vec3& drone_velocity,
    const Vec3& target_position,
    double cable_length
) noexcept {
    Vec3 payload_offset;
    payload_offset.x = payload_state.position.x - drone_position.x;
    payload_offset.y = payload_state.position.y - drone_position.y;
    payload_offset.z = payload_state.position.z - drone_position.z;
    
    Vec3 ideal_offset(0, 0, -cable_length);
    
    Vec3 swing_error;
    swing_error.x = payload_offset.x - ideal_offset.x;
    swing_error.y = payload_offset.y - ideal_offset.y;
    swing_error.z = 0;
    
    Vec3 swing_rate;
    swing_rate.x = payload_state.velocity.x - drone_velocity.x;
    swing_rate.y = payload_state.velocity.y - drone_velocity.y;
    swing_rate.z = 0;
    
    double omega_n = computeNaturalFrequency(cable_length);
    
    Vec3 compensation;
    compensation.x = -gains_.K_swing[0] * swing_error.x - gains_.K_swing[1] * swing_rate.x;
    compensation.y = -gains_.K_swing[2] * swing_error.y - gains_.K_swing[3] * swing_rate.y;
    compensation.z = 0;
    
    compensation.x *= omega_n * omega_n;
    compensation.y *= omega_n * omega_n;
    
    double comp_mag = safeLength(compensation);
    if (comp_mag > gains_.max_compensation) {
        double scale = gains_.max_compensation / comp_mag;
        compensation.x *= scale;
        compensation.y *= scale;
    }
    
    return compensation;
}

Vec3 SwingingPayloadController::computeInputShaping(
    const Vec3& commanded_accel,
    double cable_length,
    double current_time
) noexcept {
    shaper_buffer_[shaper_head_] = {current_time, commanded_accel};
    shaper_head_ = (shaper_head_ + 1) % SHAPER_BUFFER_SIZE;
    if (shaper_count_ < SHAPER_BUFFER_SIZE) shaper_count_++;
    
    double omega_n = computeNaturalFrequency(cable_length);
    double T = (omega_n > PhysicsConstants::VELOCITY_EPSILON) ? (3.14159265359 / omega_n) : 1.0;
    
    double zeta = gains_.damping_ratio;
    double K = std::exp(-zeta * 3.14159265359 / std::sqrt(std::max(1e-6, 1.0 - zeta * zeta)));
    double A1 = 1.0 / (1.0 + K);
    double A2 = K / (1.0 + K);
    
    Vec3 delayed = getDelayedCommand(T, current_time);
    
    Vec3 shaped;
    shaped.x = commanded_accel.x * A1 + delayed.x * A2;
    shaped.y = commanded_accel.y * A1 + delayed.y * A2;
    shaped.z = commanded_accel.z * A1 + delayed.z * A2;
    
    return shaped;
}

Vec3 SwingingPayloadController::computeEnergyBasedControl(
    const PayloadState& payload_state,
    const Vec3& drone_position,
    double cable_length,
    double target_energy
) noexcept {
    double vel_sq = payload_state.velocity.x * payload_state.velocity.x +
                    payload_state.velocity.y * payload_state.velocity.y +
                    payload_state.velocity.z * payload_state.velocity.z;
    double ke = 0.5 * vel_sq;
    
    Vec3 cable_vec;
    cable_vec.x = payload_state.position.x - drone_position.x;
    cable_vec.y = payload_state.position.y - drone_position.y;
    cable_vec.z = payload_state.position.z - drone_position.z;
    
    double height_from_lowest = cable_length + cable_vec.z;
    double pe = PhysicsConstants::GRAVITY * height_from_lowest;
    
    double current_energy = ke + pe;
    double energy_error = current_energy - target_energy;
    
    if (std::abs(energy_error) < 0.01) {
        return Vec3();
    }
    
    Vec3 swing_dir(cable_vec.x, cable_vec.y, 0);
    swing_dir = safeNormalize(swing_dir);
    
    double control_gain = 2.0;
    double control_mag = control_gain * energy_error;
    control_mag = std::clamp(control_mag, -gains_.max_compensation, gains_.max_compensation);
    
    double phase = payload_state.velocity.x * swing_dir.x + payload_state.velocity.y * swing_dir.y;
    if (energy_error > 0) {
        control_mag = -control_mag * (phase > 0 ? 1.0 : -1.0);
    }
    
    return Vec3(swing_dir.x * control_mag, swing_dir.y * control_mag, 0);
}

PayloadState SwingingPayloadController::estimatePayloadFromCableAngles(
    double theta_x,
    double theta_y,
    double cable_length,
    const Vec3& drone_position,
    const Vec3& drone_velocity
) noexcept {
    PayloadState estimated;
    
    double sin_x = std::sin(theta_x);
    double sin_y = std::sin(theta_y);
    double cos_x = std::cos(theta_x);
    double cos_y = std::cos(theta_y);
    
    estimated.position.x = drone_position.x + cable_length * sin_y;
    estimated.position.y = drone_position.y + cable_length * sin_x;
    estimated.position.z = drone_position.z - cable_length * cos_x * cos_y;
    
    estimated.velocity.x = drone_velocity.x + cable_length * cos_y;
    estimated.velocity.y = drone_velocity.y + cable_length * cos_x;
    estimated.velocity.z = drone_velocity.z;
    
    return estimated;
}

Vec3 SwingingPayloadController::computeCompensationFromAngles(
    double theta_x,
    double theta_y,
    double theta_x_rate,
    double theta_y_rate,
    const Vec3& drone_position,
    const Vec3& drone_velocity,
    double cable_length
) noexcept {
    double omega_n = computeNaturalFrequency(cable_length);
    
    Vec3 compensation;
    compensation.x = -gains_.K_swing[0] * theta_y - gains_.K_swing[1] * theta_y_rate;
    compensation.y = -gains_.K_swing[2] * theta_x - gains_.K_swing[3] * theta_x_rate;
    compensation.z = 0;
    
    compensation.x *= cable_length * omega_n * omega_n;
    compensation.y *= cable_length * omega_n * omega_n;
    
    double comp_mag = safeLength(compensation);
    if (comp_mag > gains_.max_compensation) {
        double scale = gains_.max_compensation / comp_mag;
        compensation.x *= scale;
        compensation.y *= scale;
    }
    
    return compensation;
}
