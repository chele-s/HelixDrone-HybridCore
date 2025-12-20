#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include "Quadrotor.h"
#include "PhysicsEngine.h"

namespace py = pybind11;

PYBIND11_MODULE(drone_core, m) {
    m.doc() = "State-of-the-Art Quadrotor Physics Engine";
    
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        })
        .def("norm", &Vec3::norm)
        .def("norm_squared", &Vec3::normSquared)
        .def("normalized", &Vec3::normalized)
        .def("dot", &Vec3::dot)
        .def("cross", &Vec3::cross)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(-py::self)
        .def_static("zero", &Vec3::zero)
        .def_static("unit_x", &Vec3::unitX)
        .def_static("unit_y", &Vec3::unitY)
        .def_static("unit_z", &Vec3::unitZ);
    
    py::class_<Quaternion>(m, "Quaternion")
        .def(py::init<>())
        .def(py::init<double, double, double, double>())
        .def_readwrite("w", &Quaternion::w)
        .def_readwrite("x", &Quaternion::x)
        .def_readwrite("y", &Quaternion::y)
        .def_readwrite("z", &Quaternion::z)
        .def("__repr__", [](const Quaternion& q) {
            return "Quaternion(" + std::to_string(q.w) + ", " + std::to_string(q.x) + 
                   ", " + std::to_string(q.y) + ", " + std::to_string(q.z) + ")";
        })
        .def("norm", &Quaternion::norm)
        .def("normalized", &Quaternion::normalized)
        .def("conjugate", &Quaternion::conjugate)
        .def("inverse", &Quaternion::inverse)
        .def("rotate", &Quaternion::rotate)
        .def("inverse_rotate", &Quaternion::inverseRotate)
        .def("to_euler_zyx", &Quaternion::toEulerZYX)
        .def(py::self * py::self)
        .def_static("from_axis_angle", &Quaternion::fromAxisAngle)
        .def_static("from_euler_zyx", &Quaternion::fromEulerZYX)
        .def_static("slerp", &Quaternion::slerp);
    
    py::class_<Mat3>(m, "Mat3")
        .def(py::init<>())
        .def("__mul__", [](const Mat3& m, const Vec3& v) { return m * v; })
        .def("inverse", &Mat3::inverse)
        .def("transpose", &Mat3::transpose)
        .def("determinant", &Mat3::determinant)
        .def_static("diagonal", &Mat3::diagonal)
        .def_static("zero", &Mat3::zero)
        .def_static("from_quaternion", &Mat3::fromQuaternion)
        .def_static("skew", &Mat3::skew);
    
    py::class_<MotorCommand>(m, "MotorCommand")
        .def(py::init<>())
        .def(py::init<double, double, double, double>())
        .def("__repr__", [](const MotorCommand& cmd) {
            return "MotorCommand(" + std::to_string(cmd.rpm[0]) + ", " + std::to_string(cmd.rpm[1]) + 
                   ", " + std::to_string(cmd.rpm[2]) + ", " + std::to_string(cmd.rpm[3]) + ")";
        })
        .def("__getitem__", [](const MotorCommand& cmd, int i) { return cmd[i]; })
        .def("__setitem__", [](MotorCommand& cmd, int i, double v) { cmd[i] = v; })
        .def_static("hover", &MotorCommand::hover);
    
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("position", &State::position)
        .def_readwrite("velocity", &State::velocity)
        .def_readwrite("orientation", &State::orientation)
        .def_readwrite("angular_velocity", &State::angularVelocity)
        .def_readwrite("battery_voltage", &State::batteryVoltage)
        .def_readwrite("time", &State::time)
        .def_property_readonly("motor_rpm", [](const State& s) {
            return std::vector<double>(s.motorRPM, s.motorRPM + 4);
        })
        .def("__repr__", [](const State& s) {
            return "State(pos=(" + std::to_string(s.position.x) + "," + 
                   std::to_string(s.position.y) + "," + std::to_string(s.position.z) + 
                   "), t=" + std::to_string(s.time) + ")";
        });
    
    py::class_<MotorState>(m, "MotorState")
        .def(py::init<>())
        .def_property_readonly("rpm", [](const MotorState& s) {
            return std::vector<double>(s.rpm, s.rpm + 4);
        })
        .def_property_readonly("current", [](const MotorState& s) {
            return std::vector<double>(s.current, s.current + 4);
        })
        .def_property_readonly("temperature", [](const MotorState& s) {
            return std::vector<double>(s.temperature, s.temperature + 4);
        });
    
    py::class_<IMUReading>(m, "IMUReading")
        .def(py::init<>())
        .def_readwrite("accelerometer", &IMUReading::accelerometer)
        .def_readwrite("gyroscope", &IMUReading::gyroscope)
        .def_readwrite("magnetometer", &IMUReading::magnetometer)
        .def_readwrite("barometer", &IMUReading::barometer)
        .def_readwrite("timestamp", &IMUReading::timestamp);
    
    py::class_<WindField>(m, "WindField")
        .def(py::init<>())
        .def_readwrite("mean_velocity", &WindField::meanVelocity)
        .def_readwrite("turbulence", &WindField::turbulence)
        .def_readwrite("gust_magnitude", &WindField::gustMagnitude)
        .def_readwrite("gust_frequency", &WindField::gustFrequency)
        .def("get_velocity_at", &WindField::getVelocityAt);
    
    py::enum_<IntegrationMethod>(m, "IntegrationMethod")
        .value("EULER", IntegrationMethod::EULER)
        .value("SEMI_IMPLICIT_EULER", IntegrationMethod::SEMI_IMPLICIT_EULER)
        .value("RK4", IntegrationMethod::RK4)
        .value("RK45_ADAPTIVE", IntegrationMethod::RK45_ADAPTIVE)
        .value("VELOCITY_VERLET", IntegrationMethod::VELOCITY_VERLET);
    
    py::enum_<MotorConfiguration>(m, "MotorConfiguration")
        .value("PLUS", MotorConfiguration::PLUS)
        .value("X", MotorConfiguration::X);
    
    py::class_<RotorConfig>(m, "RotorConfig")
        .def(py::init<>())
        .def_readwrite("radius", &RotorConfig::radius)
        .def_readwrite("chord", &RotorConfig::chord)
        .def_readwrite("pitch_angle", &RotorConfig::pitchAngle)
        .def_readwrite("lift_slope", &RotorConfig::liftSlope)
        .def_readwrite("drag_coeff", &RotorConfig::dragCoeff)
        .def_readwrite("inflow_ratio", &RotorConfig::inflowRatio);
    
    py::class_<MotorConfig>(m, "MotorConfig")
        .def(py::init<>())
        .def_readwrite("kv", &MotorConfig::kv)
        .def_readwrite("resistance", &MotorConfig::resistance)
        .def_readwrite("torque_constant", &MotorConfig::torqueConstant)
        .def_readwrite("friction_coeff", &MotorConfig::frictionCoeff)
        .def_readwrite("inertia", &MotorConfig::inertia)
        .def_readwrite("max_current", &MotorConfig::maxCurrent)
        .def_readwrite("efficiency", &MotorConfig::efficiency);
    
    py::class_<BatteryConfig>(m, "BatteryConfig")
        .def(py::init<>())
        .def_readwrite("nominal_voltage", &BatteryConfig::nominalVoltage)
        .def_readwrite("max_voltage", &BatteryConfig::maxVoltage)
        .def_readwrite("min_voltage", &BatteryConfig::minVoltage)
        .def_readwrite("capacity", &BatteryConfig::capacity)
        .def_readwrite("internal_resistance", &BatteryConfig::internalResistance);
    
    py::class_<AeroConfig>(m, "AeroConfig")
        .def(py::init<>())
        .def_readwrite("air_density", &AeroConfig::airDensity)
        .def_readwrite("ground_effect_coeff", &AeroConfig::groundEffectCoeff)
        .def_readwrite("ground_effect_height", &AeroConfig::groundEffectHeight)
        .def_readwrite("parasitic_drag_area", &AeroConfig::parasiticDragArea)
        .def_readwrite("induced_drag_factor", &AeroConfig::inducedDragFactor);
    
    py::class_<QuadrotorConfig>(m, "QuadrotorConfig")
        .def(py::init<>())
        .def_readwrite("mass", &QuadrotorConfig::mass)
        .def_readwrite("arm_length", &QuadrotorConfig::armLength)
        .def_readwrite("motor_config", &QuadrotorConfig::motorConfig)
        .def_readwrite("integration_method", &QuadrotorConfig::integrationMethod)
        .def_readwrite("rotor", &QuadrotorConfig::rotor)
        .def_readwrite("motor", &QuadrotorConfig::motor)
        .def_readwrite("battery", &QuadrotorConfig::battery)
        .def_readwrite("aero", &QuadrotorConfig::aero)
        .def_readwrite("enable_ground_effect", &QuadrotorConfig::enableGroundEffect)
        .def_readwrite("enable_wind_disturbance", &QuadrotorConfig::enableWindDisturbance)
        .def_readwrite("enable_motor_dynamics", &QuadrotorConfig::enableMotorDynamics)
        .def_readwrite("enable_battery_dynamics", &QuadrotorConfig::enableBatteryDynamics)
        .def_readwrite("enable_imu", &QuadrotorConfig::enableIMU)
        .def_readwrite("ground_z", &QuadrotorConfig::groundZ)
        .def_readwrite("ground_restitution", &QuadrotorConfig::groundRestitution)
        .def_readwrite("ground_friction", &QuadrotorConfig::groundFriction);
    
    py::class_<Quadrotor>(m, "Quadrotor")
        .def(py::init<>())
        .def(py::init<const QuadrotorConfig&>())
        .def("step", &Quadrotor::step)
        .def("step_adaptive", &Quadrotor::stepAdaptive)
        .def("reset", &Quadrotor::reset)
        .def("get_state", &Quadrotor::getState)
        .def("get_motor_state", &Quadrotor::getMotorState)
        .def("get_imu_reading", &Quadrotor::getIMUReading)
        .def("set_state", &Quadrotor::setState)
        .def("set_position", &Quadrotor::setPosition)
        .def("set_velocity", &Quadrotor::setVelocity)
        .def("set_orientation", &Quadrotor::setOrientation)
        .def("set_angular_velocity", &Quadrotor::setAngularVelocity)
        .def("set_motor_configuration", &Quadrotor::setMotorConfiguration)
        .def("set_integration_method", &Quadrotor::setIntegrationMethod)
        .def("set_wind", &Quadrotor::setWind)
        .def("enable_feature", &Quadrotor::enableFeature)
        .def("get_config", &Quadrotor::getConfig, py::return_value_policy::reference)
        .def("get_forces", &Quadrotor::getForces)
        .def("get_torques", &Quadrotor::getTorques)
        .def("get_simulation_time", &Quadrotor::getSimulationTime);
    
    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def(py::init<>())
        .def(py::init<IntegrationMethod>())
        .def("set_integration_method", &PhysicsEngine::setIntegrationMethod)
        .def("get_integration_method", &PhysicsEngine::getIntegrationMethod)
        .def("set_adaptive_tolerance", &PhysicsEngine::setAdaptiveTolerance)
        .def("set_min_max_step", &PhysicsEngine::setMinMaxStep)
        .def_static("compute_inertia_from_mass", &PhysicsEngine::computeInertiaFromMass)
        .def_static("integrate_quaternion", &PhysicsEngine::integrateQuaternion);
    
    py::class_<BladeElementTheory>(m, "BladeElementTheory")
        .def_static("compute_thrust", &BladeElementTheory::computeThrust)
        .def_static("compute_torque", &BladeElementTheory::computeTorque)
        .def_static("compute_power", &BladeElementTheory::computePower)
        .def_static("compute_inflow", &BladeElementTheory::computeInflow);
    
    py::class_<GroundEffect>(m, "GroundEffect")
        .def_static("get_coefficient", &GroundEffect::getCoefficient);
    
    py::class_<DrydenWindModel>(m, "DrydenWindModel")
        .def(py::init<>())
        .def("set_altitude", &DrydenWindModel::setAltitude)
        .def("set_wind_speed", &DrydenWindModel::setWindSpeed)
        .def("update", &DrydenWindModel::update);
    
    py::class_<IMUSimulator>(m, "IMUSimulator")
        .def(py::init<>())
        .def("set_accel_noise", &IMUSimulator::setAccelNoise)
        .def("set_gyro_noise", &IMUSimulator::setGyroNoise)
        .def("set_accel_bias", &IMUSimulator::setAccelBias)
        .def("set_gyro_bias", &IMUSimulator::setGyroBias)
        .def("simulate", &IMUSimulator::simulate);
}