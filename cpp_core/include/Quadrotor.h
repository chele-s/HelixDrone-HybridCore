#pragma once
#include "Types.h"
#include "PhysicsEngine.h"
#include <array>
#include <random>
#include <atomic>
#include <stdexcept>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, align) _aligned_malloc(size, align)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define aligned_malloc(size, align) std::aligned_alloc(align, size)
#define aligned_free(ptr) std::free(ptr)
#endif

struct SubStepConfig {
    int physicsSubSteps;
    bool enableSubStepping;
    double minSubStepDt;
    double maxSubStepDt;
    
    constexpr SubStepConfig() noexcept
        : physicsSubSteps(4), enableSubStepping(true)
        , minSubStepDt(0.0001), maxSubStepDt(0.005) {}
};

struct QuadrotorConfig {
    double mass = 1.0;
    double armLength = 0.25;
    Mat3 inertiaMatrix = Mat3::diagonal(0.0082, 0.0082, 0.0149);
    MotorConfiguration motorConfig = MotorConfiguration::X;
    IntegrationMethod integrationMethod = IntegrationMethod::RK4;
    
    RotorConfig rotor;
    MotorConfig motor;
    BatteryConfig battery;
    AeroConfig aero;
    FuelConfig fuel;
    SubStepConfig subStep;
    
    bool enableGroundEffect = true;
    bool enableWindDisturbance = false;
    bool enableMotorDynamics = true;
    bool enableBatteryDynamics = false;
    bool enableIMU = false;
    bool enableNonlinearMotor = true;
    bool enableBladeFlapping = true;
    bool enableVariableMass = false;
    bool enableAdvancedAero = true;
    
    double groundZ = 0.0;
    double groundRestitution = 0.3;
    double groundFriction = 0.5;
};

struct MotorState {
    double rpm[4];
    double current[4];
    double temperature[4];
    
    constexpr MotorState() noexcept
        : rpm{0,0,0,0}, current{0,0,0,0}, temperature{25,25,25,25} {}
};

class IntegrationGuard {
public:
    explicit IntegrationGuard(std::atomic<bool>& flag) noexcept : flag_(flag) {
        flag_.store(true, std::memory_order_release);
    }
    ~IntegrationGuard() noexcept {
        flag_.store(false, std::memory_order_release);
    }
    IntegrationGuard(const IntegrationGuard&) = delete;
    IntegrationGuard& operator=(const IntegrationGuard&) = delete;
private:
    std::atomic<bool>& flag_;
};

class alignas(64) Quadrotor {
public:
    Quadrotor() noexcept;
    explicit Quadrotor(const QuadrotorConfig& config) noexcept;
    
    void* operator new(std::size_t size) {
        void* ptr = aligned_malloc(size, 64);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }
    
    void* operator new[](std::size_t size) {
        void* ptr = aligned_malloc(size, 64);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }
    
    void operator delete(void* ptr) noexcept {
        aligned_free(ptr);
    }
    
    void operator delete[](void* ptr) noexcept {
        aligned_free(ptr);
    }
    
    void step(const MotorCommand& command, double dt);
    void stepWithSubStepping(const MotorCommand& command, double agentDt);
    void stepAdaptive(const MotorCommand& command, double& dt);
    void reset() noexcept;
    
    State getState() const noexcept;
    MotorState getMotorState() const noexcept;
    IMUReading getIMUReading() const noexcept;
    
    void setState(const State& state);
    void setPosition(const Vec3& pos);
    void setVelocity(const Vec3& vel);
    void setOrientation(const Quaternion& q);
    void setAngularVelocity(const Vec3& omega);
    
    void setMotorConfiguration(MotorConfiguration config) noexcept;
    void setIntegrationMethod(IntegrationMethod method) noexcept;
    void setSubStepConfig(const SubStepConfig& config) noexcept;
    
    void setWind(const Vec3& meanWind) noexcept;
    void enableFeature(const char* feature, bool enable) noexcept;
    
    const QuadrotorConfig& getConfig() const noexcept;
    Vec3 getForces() const noexcept;
    Vec3 getTorques() const noexcept;
    double getSimulationTime() const noexcept;
    double getCurrentMass() const noexcept;
    double getCurrentFuel() const noexcept;
    int getSubStepCount() const noexcept;
    
    bool isIntegrating() const noexcept;

private:
    static constexpr double GRAVITY = 9.80665;
    
    QuadrotorConfig config_;
    State state_;
    MotorState motorState_;
    MotorDynamicsState motorDynamics_;
    BladeFlappingState flappingState_;
    DynamicMassState massState_;
    IMUReading imuReading_;
    
    PhysicsEngine physics_;
    DrydenWindModel windModel_;
    IMUSimulator imuSim_;
    
    double simulationTime_;
    double totalFuelConsumed_;
    int lastSubStepCount_;
    Vec3 lastForce_;
    Vec3 lastTorque_;
    Vec3 windVelocity_;
    
    Mat3 cachedInertiaInverse_;
    Mat3 currentInertia_;
    
    mutable std::atomic<bool> integrating_;
    
    void initialize() noexcept;
    void checkNotIntegrating() const;
    void updateInertia() noexcept;
    void stepInternal(const MotorCommand& command, double dt);
    
    RigidBodyDerivative computeDerivative(const RigidBodyState& rbState, double t) const;
    
    void updateMotorDynamics(const MotorCommand& command, double dt) noexcept;
    void updateBladeFlapping(double dt) noexcept;
    void updateMassDynamics(double dt) noexcept;
    Vec3 computeTotalThrust() const noexcept;
    Vec3 computeTotalTorque() const noexcept;
    Vec3 computeAerodynamicForces(const Vec3& velocityBody) const noexcept;
    Vec3 computeGyroscopicTorque() const noexcept;
    
    void enforceGroundConstraint() noexcept;
    void updateIMU() noexcept;
    void updateWind(double dt) noexcept;
    void updateBattery() noexcept;
};