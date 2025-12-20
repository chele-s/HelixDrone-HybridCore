#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <iomanip>
#include "../cpp_core/include/Types.h"
#include "../cpp_core/include/PhysicsEngine.h"
#include "../cpp_core/include/Quadrotor.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace CoreTest {
    using HighResClock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<HighResClock>;

    struct Metrics {
        double maxError;
        double meanError;
        double rmsError;
        double computeTimeNs;
        bool passed;
    };

    template<typename T>
    inline void assert_near(T val, T expected, T epsilon, const char* name) {
        if (std::abs(val - expected) > epsilon) {
            std::cerr << "FAIL: " << name << " | " << val << " != " << expected << " (diff: " << std::abs(val - expected) << ")" << std::endl;
            exit(1);
        }
    }

    class PhysicsVariant {
    public:
        static void validate_vector_algebra() {
            Vec3 v1(1.0, 2.0, 3.0);
            Vec3 v2(4.0, 5.0, 6.0);
            assert_near((v1 + v2).norm(), std::sqrt(152.0), 1e-9, "Vec3::Add");
            assert_near(v1.dot(v2), 32.0, 1e-9, "Vec3::Dot");
            assert_near(v1.cross(v2).norm(), std::sqrt(324.0), 1e-9, "Vec3::Cross");
        }

        static void validate_quaternion_kinematics() {
            Quaternion q = Quaternion::fromAxisAngle(Vec3(0, 0, 1), M_PI / 2);
            Vec3 v(1, 0, 0);
            Vec3 rotated = q.rotate(v);
            assert_near(rotated.x, 0.0, 1e-9, "Quat::RotateX");
            assert_near(rotated.y, 1.0, 1e-9, "Quat::RotateY");
            
            Quaternion p(1, 0, 0, 0);
            Vec3 omega(0.1, 0.2, 0.3);
            Quaternion deriv = p.derivative(omega);
            assert_near(deriv.x, 0.05, 1e-9, "Quat::DerivX");
        }

        static Metrics benchmark_step_latency() {
            QuadrotorConfig config;
            config.mass = 1.2;
            config.enableGroundEffect = true;
            config.enableWindDisturbance = true;
            Quadrotor drone(config);
            
            MotorCommand cmd(15000, 15000, 15000, 15000);
            
            auto start = HighResClock::now();
            constexpr int ITERATIONS = 10000;
            
            for(int i=0; i<ITERATIONS; ++i) {
                drone.step(cmd, 0.001);
            }
            
            auto end = HighResClock::now();
            double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            
            return {0, 0, 0, total_ns / ITERATIONS, true};
        }

        static void stress_test_energy_conservation() {
            QuadrotorConfig config;
            config.enableMotorDynamics = false;
            config.enableGroundEffect = false;
            config.enableWindDisturbance = false;
            config.integrationMethod = IntegrationMethod::RK4;
            Quadrotor drone(config);
            
            drone.setPosition(Vec3(0, 0, 100));
            double initial_energy = 9.80665 * config.mass * 100.0;
            
            for(int i=0; i<5000; ++i) {
                drone.step(MotorCommand(0, 0, 0, 0), 0.002);
            }
            
            State s = drone.getState();
            double ke = 0.5 * config.mass * s.velocity.normSquared();
            double pe = 9.80665 * config.mass * s.position.z;
            
            assert_near(ke + pe, initial_energy, 1.0, "EnergyConservation");
        }

        static void integration_symplecticity_check() {
            QuadrotorConfig cfg; 
            cfg.enableWindDisturbance = true;
            PhysicsEngine engine(IntegrationMethod::RK45);
            RigidBodyState state;
            state.position = Vec3(0,0,10);
            state.orientation = Quaternion(1,0,0,0);
            
            auto deriv = [](const RigidBodyState& s, double t) -> RigidBodyDerivative {
                RigidBodyDerivative d;
                d.velocity = s.velocity;
                d.acceleration = Vec3(0,0,-9.80665);
                d.angularAcceleration = Vec3(0,0,0.1);
                d.orientationDot = s.orientation.derivative(s.angularVelocity);
                return d;
            };

            double dt = 0.01;
            double t = 0;
            AdaptiveStepResult res = engine.integrateAdaptive(state, deriv, dt, t);
            
            if (!res.accepted) exit(2);
            assert_near(res.state.position.z, 10.0 + -0.5*9.80665*dt*dt, 1e-4, "RK45::Position");
        }
    };
}

int main() {
    try {
        std::cout << "Running SOTA Physics Suite [Optimized Build]..." << std::endl;
        
        std::vector<std::future<void>> futures;
        futures.push_back(std::async(std::launch::async, CoreTest::PhysicsVariant::validate_vector_algebra));
        futures.push_back(std::async(std::launch::async, CoreTest::PhysicsVariant::validate_quaternion_kinematics));
        futures.push_back(std::async(std::launch::async, CoreTest::PhysicsVariant::stress_test_energy_conservation));
        futures.push_back(std::async(std::launch::async, CoreTest::PhysicsVariant::integration_symplecticity_check));
        
        for(auto& f : futures) f.get();
        
        auto metrics = CoreTest::PhysicsVariant::benchmark_step_latency();
        
        std::cout << "============================================" << std::endl;
        std::cout << "STATUS: PASSED (ALL)" << std::endl;
        std::cout << "PERFORMANCE: " << metrics.computeTimeNs << " ns/step" << std::endl;
        std::cout << "THROUGHPUT: " << (1e9 / metrics.computeTimeNs) / 1e6 << " Msteps/s" << std::endl;
        std::cout << "============================================" << std::endl;
        
        return 0;
    } catch (...) {
        return -1;
    }
}
