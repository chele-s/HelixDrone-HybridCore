#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "Quadrotor.h"
#include "PhysicsEngine.h"
#include "ReplayBuffer.h"
#include "SOTAActuator.h"
#include "StateEstimator.h"
#include "ActuatorFailure.h"
#include "CollisionWorld.h"
#include "PayloadDynamics.h"

namespace py = pybind11;

PYBIND11_MODULE(drone_core, m) {
    m.doc() = "State-of-the-Art Quadrotor Physics Engine with SIMD, Blade Flapping, Variable Mass, and Sub-Stepping";
    
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
        .def_readonly("position", &State::position)
        .def_readonly("velocity", &State::velocity)
        .def_readonly("orientation", &State::orientation)
        .def_readonly("angular_velocity", &State::angularVelocity)
        .def_readonly("battery_voltage", &State::batteryVoltage)
        .def_readonly("time", &State::time)
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
        .def_readonly("accelerometer", &IMUReading::accelerometer)
        .def_readonly("gyroscope", &IMUReading::gyroscope)
        .def_readonly("magnetometer", &IMUReading::magnetometer)
        .def_readonly("barometer", &IMUReading::barometer)
        .def_readonly("timestamp", &IMUReading::timestamp);
    
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
    
    py::enum_<PropulsionType>(m, "PropulsionType")
        .value("ELECTRIC", PropulsionType::ELECTRIC)
        .value("COMBUSTION", PropulsionType::COMBUSTION)
        .value("HYBRID", PropulsionType::HYBRID);
    
    py::class_<SubStepConfig>(m, "SubStepConfig")
        .def(py::init<>())
        .def_readwrite("physics_sub_steps", &SubStepConfig::physicsSubSteps)
        .def_readwrite("enable_sub_stepping", &SubStepConfig::enableSubStepping)
        .def_readwrite("min_sub_step_dt", &SubStepConfig::minSubStepDt)
        .def_readwrite("max_sub_step_dt", &SubStepConfig::maxSubStepDt);
    
    py::class_<ESCConfig>(m, "ESCConfig")
        .def(py::init<>())
        .def_readwrite("base_response_time", &ESCConfig::baseResponseTime)
        .def_readwrite("min_response_time", &ESCConfig::minResponseTime)
        .def_readwrite("max_response_time", &ESCConfig::maxResponseTime)
        .def_readwrite("nonlinear_gamma", &ESCConfig::nonlinearGamma)
        .def_readwrite("voltage_scale_factor", &ESCConfig::voltageScaleFactor)
        .def_readwrite("current_limit_softness", &ESCConfig::currentLimitSoftness)
        .def_readwrite("thermal_coeff", &ESCConfig::thermalCoeff)
        .def_readwrite("pwm_frequency", &ESCConfig::pwmFrequency);
    
    py::class_<BladeFlappingConfig>(m, "BladeFlappingConfig")
        .def(py::init<>())
        .def_readwrite("hinge_offset", &BladeFlappingConfig::hingeOffset)
        .def_readwrite("lock_number", &BladeFlappingConfig::lockNumber)
        .def_readwrite("flap_frequency", &BladeFlappingConfig::flapFrequency)
        .def_readwrite("pitch_flap_coupling", &BladeFlappingConfig::pitchFlapCoupling)
        .def_readwrite("roll_flap_coupling", &BladeFlappingConfig::rollFlapCoupling)
        .def_readwrite("advance_ratio_threshold", &BladeFlappingConfig::advanceRatioThreshold)
        .def_readwrite("enabled", &BladeFlappingConfig::enabled);
    
    py::class_<FuelConfig>(m, "FuelConfig")
        .def(py::init<>())
        .def_readwrite("initial_fuel_mass", &FuelConfig::initialFuelMass)
        .def_readwrite("current_fuel_mass", &FuelConfig::currentFuelMass)
        .def_readwrite("specific_fuel_consumption", &FuelConfig::specificFuelConsumption)
        .def_readwrite("fuel_density", &FuelConfig::fuelDensity)
        .def_readwrite("variable_mass_enabled", &FuelConfig::variableMassEnabled)
        .def_static("combustion", &FuelConfig::combustion);
    
    py::class_<RotorConfig>(m, "RotorConfig")
        .def(py::init<>())
        .def_readwrite("radius", &RotorConfig::radius)
        .def_readwrite("chord", &RotorConfig::chord)
        .def_readwrite("pitch_angle", &RotorConfig::pitchAngle)
        .def_readwrite("lift_slope", &RotorConfig::liftSlope)
        .def_readwrite("drag_coeff", &RotorConfig::dragCoeff)
        .def_readwrite("inflow_ratio", &RotorConfig::inflowRatio)
        .def_readwrite("flapping", &RotorConfig::flapping);
    
    py::class_<MotorConfig>(m, "MotorConfig")
        .def(py::init<>())
        .def_readwrite("kv", &MotorConfig::kv)
        .def_readwrite("resistance", &MotorConfig::resistance)
        .def_readwrite("torque_constant", &MotorConfig::torqueConstant)
        .def_readwrite("friction_coeff", &MotorConfig::frictionCoeff)
        .def_readwrite("inertia", &MotorConfig::inertia)
        .def_readwrite("max_current", &MotorConfig::maxCurrent)
        .def_readwrite("efficiency", &MotorConfig::efficiency)
        .def_readwrite("thermal_mass", &MotorConfig::thermalMass)
        .def_readwrite("thermal_resistance", &MotorConfig::thermalResistance)
        .def_readwrite("max_rpm", &MotorConfig::maxRpm)
        .def_readwrite("esc", &MotorConfig::esc);
    
    py::class_<BatteryConfig>(m, "BatteryConfig")
        .def(py::init<>())
        .def_readwrite("nominal_voltage", &BatteryConfig::nominalVoltage)
        .def_readwrite("max_voltage", &BatteryConfig::maxVoltage)
        .def_readwrite("min_voltage", &BatteryConfig::minVoltage)
        .def_readwrite("capacity", &BatteryConfig::capacity)
        .def_readwrite("internal_resistance", &BatteryConfig::internalResistance)
        .def_readwrite("soc_curve_alpha", &BatteryConfig::socCurveAlpha)
        .def_readwrite("soc_curve_beta", &BatteryConfig::socCurveBeta)
        .def_readwrite("temperature_coeff", &BatteryConfig::temperatureCoeff);
    
    py::class_<AeroConfig>(m, "AeroConfig")
        .def(py::init<>())
        .def_readwrite("air_density", &AeroConfig::airDensity)
        .def_readwrite("ground_effect_coeff", &AeroConfig::groundEffectCoeff)
        .def_readwrite("ground_effect_height", &AeroConfig::groundEffectHeight)
        .def_readwrite("parasitic_drag_area", &AeroConfig::parasiticDragArea)
        .def_readwrite("induced_drag_factor", &AeroConfig::inducedDragFactor)
        .def_readwrite("rotational_drag_coeff", &AeroConfig::rotationalDragCoeff)
        .def_readwrite("advance_ratio_drag_scale", &AeroConfig::advanceRatioDragScale);
    
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
        .def_readwrite("fuel", &QuadrotorConfig::fuel)
        .def_readwrite("sub_step", &QuadrotorConfig::subStep)
        .def_readwrite("enable_ground_effect", &QuadrotorConfig::enableGroundEffect)
        .def_readwrite("enable_wind_disturbance", &QuadrotorConfig::enableWindDisturbance)
        .def_readwrite("enable_motor_dynamics", &QuadrotorConfig::enableMotorDynamics)
        .def_readwrite("enable_battery_dynamics", &QuadrotorConfig::enableBatteryDynamics)
        .def_readwrite("enable_imu", &QuadrotorConfig::enableIMU)
        .def_readwrite("enable_nonlinear_motor", &QuadrotorConfig::enableNonlinearMotor)
        .def_readwrite("enable_blade_flapping", &QuadrotorConfig::enableBladeFlapping)
        .def_readwrite("enable_variable_mass", &QuadrotorConfig::enableVariableMass)
        .def_readwrite("enable_advanced_aero", &QuadrotorConfig::enableAdvancedAero)
        .def_readwrite("ground_z", &QuadrotorConfig::groundZ)
        .def_readwrite("ground_restitution", &QuadrotorConfig::groundRestitution)
        .def_readwrite("ground_friction", &QuadrotorConfig::groundFriction);
    
    py::class_<Quadrotor>(m, "Quadrotor")
        .def(py::init<>())
        .def(py::init<const QuadrotorConfig&>())
        .def("step", &Quadrotor::step)
        .def("step_with_sub_stepping", &Quadrotor::stepWithSubStepping)
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
        .def("set_sub_step_config", &Quadrotor::setSubStepConfig)
        .def("set_wind", &Quadrotor::setWind)
        .def("enable_feature", &Quadrotor::enableFeature)
        .def("get_config", &Quadrotor::getConfig, py::return_value_policy::reference)
        .def("get_forces", &Quadrotor::getForces)
        .def("get_torques", &Quadrotor::getTorques)
        .def("get_simulation_time", &Quadrotor::getSimulationTime)
        .def("get_current_mass", &Quadrotor::getCurrentMass)
        .def("get_current_fuel", &Quadrotor::getCurrentFuel)
        .def("get_sub_step_count", &Quadrotor::getSubStepCount)
        .def("is_integrating", &Quadrotor::isIntegrating);
    
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
    
    py::class_<NonlinearMotorModel>(m, "NonlinearMotorModel")
        .def_static("compute_nonlinear_response", &NonlinearMotorModel::computeNonlinearResponse)
        .def_static("compute_esc_delay", &NonlinearMotorModel::computeESCDelay)
        .def_static("compute_soft_current_limit", &NonlinearMotorModel::computeSoftCurrentLimit)
        .def_static("compute_thermal_derating", &NonlinearMotorModel::computeThermalDerating);
    
    py::class_<BladeFlappingModel>(m, "BladeFlappingModel")
        .def_static("compute_advance_ratio", &BladeFlappingModel::computeAdvanceRatio);
    
    py::class_<VariableMassModel>(m, "VariableMassModel")
        .def_static("compute_inertia_with_fuel", &VariableMassModel::computeInertiaWithFuel);
    
    py::class_<helix::SumTree>(m, "SumTree")
        .def(py::init<size_t>())
        .def("update", &helix::SumTree::update)
        .def("add", &helix::SumTree::add)
        .def("get", [](const helix::SumTree& self, double value) {
            size_t treeIdx, dataIdx;
            double priority;
            self.get(value, treeIdx, priority, dataIdx);
            return py::make_tuple(treeIdx, priority, dataIdx);
        })
        .def("total_priority", &helix::SumTree::totalPriority)
        .def("max_priority", &helix::SumTree::maxPriority)
        .def("min_priority", &helix::SumTree::minPriority)
        .def("capacity", &helix::SumTree::capacity);
    
    py::class_<helix::PrioritizedReplayBuffer>(m, "PrioritizedReplayBuffer")
        .def(py::init<size_t, size_t, size_t, double, double, size_t, double>(),
            py::arg("capacity"),
            py::arg("state_dim"),
            py::arg("action_dim"),
            py::arg("alpha") = 0.6,
            py::arg("beta_start") = 0.4,
            py::arg("beta_frames") = 100000,
            py::arg("epsilon") = 1e-6)
        .def("beta", &helix::PrioritizedReplayBuffer::beta)
        .def("push", [](helix::PrioritizedReplayBuffer& self, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> state, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> action, 
                        float reward, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> next_state, 
                        float done) {
            self.push(state.data(), action.data(), reward, next_state.data(), done);
        })
        .def("push_batch", [](helix::PrioritizedReplayBuffer& self,
                              py::array_t<float, py::array::c_style | py::array::forcecast> states,
                              py::array_t<float, py::array::c_style | py::array::forcecast> actions,
                              py::array_t<float, py::array::c_style | py::array::forcecast> rewards,
                              py::array_t<float, py::array::c_style | py::array::forcecast> next_states,
                              py::array_t<float, py::array::c_style | py::array::forcecast> dones) {
            self.pushBatch(states.data(), actions.data(), rewards.data(), 
                          next_states.data(), dones.data(), static_cast<size_t>(states.shape(0)));
        })
        .def("sample", [](helix::PrioritizedReplayBuffer& self, size_t batchSize) {
            auto result = self.sample(batchSize);
            
            py::array_t<float> states({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(result.stateDim)});
            py::array_t<float> actions({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(result.actionDim)});
            py::array_t<float> rewards({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(1)});
            py::array_t<float> next_states({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(result.stateDim)});
            py::array_t<float> dones({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(1)});
            py::array_t<float> weights({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(1)});
            py::array_t<int32_t> tree_indices({static_cast<py::ssize_t>(batchSize)});
            
            std::memcpy(states.mutable_data(), result.states.data(), batchSize * result.stateDim * sizeof(float));
            std::memcpy(actions.mutable_data(), result.actions.data(), batchSize * result.actionDim * sizeof(float));
            std::memcpy(next_states.mutable_data(), result.nextStates.data(), batchSize * result.stateDim * sizeof(float));
            std::memcpy(tree_indices.mutable_data(), result.treeIndices.data(), batchSize * sizeof(int32_t));
            
            float* rPtr = rewards.mutable_data();
            float* dPtr = dones.mutable_data();
            float* wPtr = weights.mutable_data();
            for (size_t i = 0; i < batchSize; ++i) {
                rPtr[i] = result.rewards[i];
                dPtr[i] = result.dones[i];
                wPtr[i] = result.weights[i];
            }
            
            return py::make_tuple(states, actions, rewards, next_states, dones, weights, tree_indices);
        })
        .def("update_priorities", [](helix::PrioritizedReplayBuffer& self,
                                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> tree_indices,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> td_errors) {
            self.updatePriorities(tree_indices.data(), td_errors.data(), static_cast<size_t>(tree_indices.size()));
        })
        .def("size", &helix::PrioritizedReplayBuffer::size)
        .def("capacity", &helix::PrioritizedReplayBuffer::capacity)
        .def("is_ready", &helix::PrioritizedReplayBuffer::isReady)
        .def("__len__", &helix::PrioritizedReplayBuffer::size);
    
    py::class_<helix::UniformReplayBuffer>(m, "UniformReplayBuffer")
        .def(py::init<size_t, size_t, size_t>(),
            py::arg("capacity"),
            py::arg("state_dim"),
            py::arg("action_dim"))
        .def("push", [](helix::UniformReplayBuffer& self, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> state, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> action, 
                        float reward, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> next_state, 
                        float done) {
            self.push(state.data(), action.data(), reward, next_state.data(), done);
        })
        .def("push_batch", [](helix::UniformReplayBuffer& self,
                              py::array_t<float, py::array::c_style | py::array::forcecast> states,
                              py::array_t<float, py::array::c_style | py::array::forcecast> actions,
                              py::array_t<float, py::array::c_style | py::array::forcecast> rewards,
                              py::array_t<float, py::array::c_style | py::array::forcecast> next_states,
                              py::array_t<float, py::array::c_style | py::array::forcecast> dones) {
            self.pushBatch(states.data(), actions.data(), rewards.data(), 
                          next_states.data(), dones.data(), static_cast<size_t>(states.shape(0)));
        })
        .def("sample", [](helix::UniformReplayBuffer& self, size_t batchSize, size_t stateDim, size_t actionDim) {
            auto result = self.sample(batchSize);
            
            py::array_t<float> states({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(stateDim)});
            py::array_t<float> actions({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(actionDim)});
            py::array_t<float> rewards({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(1)});
            py::array_t<float> next_states({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(stateDim)});
            py::array_t<float> dones({static_cast<py::ssize_t>(batchSize), static_cast<py::ssize_t>(1)});
            
            std::memcpy(states.mutable_data(), result.states.data(), batchSize * stateDim * sizeof(float));
            std::memcpy(actions.mutable_data(), result.actions.data(), batchSize * actionDim * sizeof(float));
            std::memcpy(next_states.mutable_data(), result.nextStates.data(), batchSize * stateDim * sizeof(float));
            
            float* rPtr = rewards.mutable_data();
            float* dPtr = dones.mutable_data();
            for (size_t i = 0; i < batchSize; ++i) {
                rPtr[i] = result.rewards[i];
                dPtr[i] = result.dones[i];
            }
            
            return py::make_tuple(states, actions, rewards, next_states, dones);
        })
        .def("size", &helix::UniformReplayBuffer::size)
        .def("capacity", &helix::UniformReplayBuffer::capacity)
        .def("is_ready", &helix::UniformReplayBuffer::isReady)
        .def("__len__", &helix::UniformReplayBuffer::size);
    
    py::class_<SOTAActuatorConfig>(m, "SOTAActuatorConfig")
        .def(py::init<>())
        .def_readwrite("delay_ms", &SOTAActuatorConfig::delayMs)
        .def_readwrite("tau_spin_up", &SOTAActuatorConfig::tauSpinUp)
        .def_readwrite("tau_spin_down", &SOTAActuatorConfig::tauSpinDown)
        .def_readwrite("rotor_inertia", &SOTAActuatorConfig::rotorInertia)
        .def_readwrite("voltage_sag_factor", &SOTAActuatorConfig::voltageSagFactor)
        .def_readwrite("max_rpm", &SOTAActuatorConfig::maxRpm)
        .def_readwrite("min_rpm", &SOTAActuatorConfig::minRpm)
        .def_readwrite("hover_rpm", &SOTAActuatorConfig::hoverRpm)
        .def_readwrite("max_slew_rate", &SOTAActuatorConfig::maxSlewRate)
        .def_readwrite("process_noise_std", &SOTAActuatorConfig::processNoiseStd)
        .def_readwrite("active_braking_gain", &SOTAActuatorConfig::activeBrakingGain)
        .def_readwrite("thermal_time_constant", &SOTAActuatorConfig::thermalTimeConstant)
        .def_readwrite("nominal_voltage", &SOTAActuatorConfig::nominalVoltage);
    
    py::class_<SOTAActuatorState>(m, "SOTAActuatorState")
        .def(py::init<>())
        .def_property_readonly("current_rpm", [](const SOTAActuatorState& s) {
            return std::vector<double>(s.currentRpm, s.currentRpm + 4);
        })
        .def_property_readonly("filtered_rpm", [](const SOTAActuatorState& s) {
            return std::vector<double>(s.filteredRpm, s.filteredRpm + 4);
        })
        .def_property_readonly("motor_temperature", [](const SOTAActuatorState& s) {
            return std::vector<double>(s.motorTemperature, s.motorTemperature + 4);
        })
        .def_property_readonly("total_current", [](const SOTAActuatorState& s) {
            return std::vector<double>(s.totalCurrent, s.totalCurrent + 4);
        })
        .def_readonly("delay_steps", &SOTAActuatorState::delaySteps)
        .def_readonly("accumulated_sag", &SOTAActuatorState::accumulatedSag);
    
    py::class_<SOTAActuatorModel>(m, "SOTAActuatorModel")
        .def(py::init<>())
        .def(py::init<const SOTAActuatorConfig&>())
        .def("reset", &SOTAActuatorModel::reset)
        .def("step", [](SOTAActuatorModel& self,
                        py::array_t<double, py::array::c_style | py::array::forcecast> commanded_rpm,
                        double dt, double voltage) {
            py::array_t<double> output(4);
            self.step(commanded_rpm.data(), dt, voltage, output.mutable_data());
            return output;
        })
        .def("step_normalized", [](SOTAActuatorModel& self,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> action,
                                   double dt, double voltage) {
            py::array_t<double> output(4);
            self.stepNormalized(action.data(), dt, voltage, output.mutable_data());
            return output;
        })
        .def("get_voltage_sag_factor", &SOTAActuatorModel::getVoltageSagFactor)
        .def("get_effective_tau", &SOTAActuatorModel::getEffectiveTau)
        .def("get_state", &SOTAActuatorModel::getState, py::return_value_policy::reference)
        .def("get_config", &SOTAActuatorModel::getConfig, py::return_value_policy::reference)
        .def("set_config", &SOTAActuatorModel::setConfig);
    
    py::class_<SOTASecondOrderFilter>(m, "SOTASecondOrderFilter")
        .def(py::init<>())
        .def(py::init<double, double, double>(),
             py::arg("cutoff_hz"), py::arg("damping"), py::arg("dt"))
        .def("configure", &SOTASecondOrderFilter::configure)
        .def("filter", &SOTASecondOrderFilter::filter)
        .def("reset", &SOTASecondOrderFilter::reset);
    
    py::class_<SOTAButterworthFilter>(m, "SOTAButterworthFilter")
        .def(py::init<>())
        .def(py::init<int, double, double>(),
             py::arg("order"), py::arg("cutoff_hz"), py::arg("sample_rate"))
        .def("configure", &SOTAButterworthFilter::configure)
        .def("filter", &SOTAButterworthFilter::filter)
        .def("reset", &SOTAButterworthFilter::reset);
    
    py::class_<SensorNoise>(m, "SensorNoise")
        .def(py::init<>())
        .def_readwrite("accel_std", &SensorNoise::accel_std)
        .def_readwrite("gyro_std", &SensorNoise::gyro_std)
        .def_readwrite("gps_pos_std", &SensorNoise::gps_pos_std)
        .def_readwrite("gps_vel_std", &SensorNoise::gps_vel_std)
        .def_readwrite("baro_std", &SensorNoise::baro_std)
        .def_readwrite("mag_std", &SensorNoise::mag_std)
        .def_readwrite("gps_update_rate", &SensorNoise::gps_update_rate)
        .def_readwrite("baro_update_rate", &SensorNoise::baro_update_rate)
        .def_readwrite("mag_update_rate", &SensorNoise::mag_update_rate)
        .def_readwrite("accel_bias_stability", &SensorNoise::accel_bias_stability)
        .def_readwrite("gyro_bias_stability", &SensorNoise::gyro_bias_stability)
        .def_readwrite("cable_angle_std", &SensorNoise::cable_angle_std)
        .def_readwrite("cable_angle_update_rate", &SensorNoise::cable_angle_update_rate)
        .def_readwrite("cable_sensor_enabled", &SensorNoise::cable_sensor_enabled);
    
    py::class_<EKFState>(m, "EKFState")
        .def(py::init<>())
        .def_readwrite("position", &EKFState::position)
        .def_readwrite("velocity", &EKFState::velocity)
        .def_readwrite("orientation", &EKFState::orientation)
        .def_readwrite("accel_bias", &EKFState::accel_bias)
        .def_readwrite("gyro_bias", &EKFState::gyro_bias);
    
    py::class_<SensorReading>(m, "SensorReading")
        .def(py::init<>())
        .def_readwrite("accelerometer", &SensorReading::accelerometer)
        .def_readwrite("gyroscope", &SensorReading::gyroscope)
        .def_readwrite("gps_position", &SensorReading::gps_position)
        .def_readwrite("gps_velocity", &SensorReading::gps_velocity)
        .def_readwrite("barometer", &SensorReading::barometer)
        .def_readwrite("magnetometer", &SensorReading::magnetometer)
        .def_readwrite("timestamp", &SensorReading::timestamp)
        .def_readwrite("gps_valid", &SensorReading::gps_valid)
        .def_readwrite("baro_valid", &SensorReading::baro_valid)
        .def_readwrite("mag_valid", &SensorReading::mag_valid)
        .def_readwrite("cable", &SensorReading::cable);
    
    py::class_<CableSensorReading>(m, "CableSensorReading")
        .def(py::init<>())
        .def_readwrite("theta_x", &CableSensorReading::theta_x)
        .def_readwrite("theta_y", &CableSensorReading::theta_y)
        .def_readwrite("tension", &CableSensorReading::tension)
        .def_readwrite("timestamp", &CableSensorReading::timestamp)
        .def_readwrite("valid", &CableSensorReading::valid);
    
    py::class_<RobustnessConfig>(m, "RobustnessConfig")
        .def(py::init<>())
        .def_readwrite("enable_chi_square_gating", &RobustnessConfig::enable_chi_square_gating)
        .def_readwrite("enable_fault_detection", &RobustnessConfig::enable_fault_detection)
        .def_readwrite("enable_adaptive_noise", &RobustnessConfig::enable_adaptive_noise)
        .def_readwrite("enable_state_dependent_noise", &RobustnessConfig::enable_state_dependent_noise)
        .def_readwrite("chi_square_threshold_1dof", &RobustnessConfig::chi_square_threshold_1dof)
        .def_readwrite("chi_square_threshold_3dof", &RobustnessConfig::chi_square_threshold_3dof)
        .def_readwrite("max_consecutive_rejections", &RobustnessConfig::max_consecutive_rejections)
        .def_readwrite("adaptive_alpha", &RobustnessConfig::adaptive_alpha)
        .def_readwrite("gyro_noise_scale_coeff", &RobustnessConfig::gyro_noise_scale_coeff)
        .def_readwrite("accel_noise_scale_coeff", &RobustnessConfig::accel_noise_scale_coeff)
        .def_readwrite("enable_external_force_estimation", &RobustnessConfig::enable_external_force_estimation)
        .def_readwrite("external_force_process_noise", &RobustnessConfig::external_force_process_noise);
    
    py::class_<ConsistencyMetrics>(m, "ConsistencyMetrics")
        .def(py::init<>())
        .def_readonly("current_nis", &ConsistencyMetrics::current_nis)
        .def_readonly("avg_nis", &ConsistencyMetrics::avg_nis)
        .def_readonly("nees", &ConsistencyMetrics::nees)
        .def_readonly("total_updates", &ConsistencyMetrics::total_updates)
        .def_readonly("rejected_updates", &ConsistencyMetrics::rejected_updates)
        .def_readonly("rejection_rate", &ConsistencyMetrics::rejection_rate);
    
    py::class_<ExtendedKalmanFilter>(m, "ExtendedKalmanFilter")
        .def(py::init<>())
        .def(py::init<const SensorNoise&>())
        .def("reset", py::overload_cast<>(&ExtendedKalmanFilter::reset))
        .def("reset", py::overload_cast<const Vec3&, const Vec3&, const Quaternion&>(&ExtendedKalmanFilter::reset))
        .def("predict", &ExtendedKalmanFilter::predict)
        .def("update_gps_position", &ExtendedKalmanFilter::updateGPSPosition)
        .def("update_gps_velocity", &ExtendedKalmanFilter::updateGPSVelocity)
        .def("update_barometer", &ExtendedKalmanFilter::updateBarometer)
        .def("update_magnetometer", &ExtendedKalmanFilter::updateMagnetometer)
        .def("get_state", &ExtendedKalmanFilter::getState)
        .def("get_position", &ExtendedKalmanFilter::getPosition)
        .def("get_velocity", &ExtendedKalmanFilter::getVelocity)
        .def("get_orientation", &ExtendedKalmanFilter::getOrientation)
        .def("get_accel_bias", &ExtendedKalmanFilter::getAccelBias)
        .def("get_gyro_bias", &ExtendedKalmanFilter::getGyroBias)
        .def("get_position_uncertainty", &ExtendedKalmanFilter::getPositionUncertainty)
        .def("get_orientation_uncertainty", &ExtendedKalmanFilter::getOrientationUncertainty)
        .def("set_process_noise", &ExtendedKalmanFilter::setProcessNoise)
        .def("set_measurement_noise", &ExtendedKalmanFilter::setMeasurementNoise)
        .def("set_robustness_config", &ExtendedKalmanFilter::setRobustnessConfig)
        .def("is_sensor_healthy", &ExtendedKalmanFilter::isSensorHealthy)
        .def("get_current_nis", &ExtendedKalmanFilter::getCurrentNIS)
        .def("get_consistency_metrics", &ExtendedKalmanFilter::getConsistencyMetrics)
        .def("is_yaw_observable", &ExtendedKalmanFilter::isYawObservable)
        .def("is_accel_valid", &ExtendedKalmanFilter::isAccelValid)
        .def("compute_nees", &ExtendedKalmanFilter::computeNEES);
    
    py::class_<SensorSimulator>(m, "SensorSimulator")
        .def(py::init<>())
        .def(py::init<const SensorNoise&>())
        .def("set_noise", &SensorSimulator::setNoise)
        .def("simulate", &SensorSimulator::simulate)
        .def("simulate_cable_angle", &SensorSimulator::simulateCableAngle)
        .def("reset", &SensorSimulator::reset)
        .def("inject_gps_failure", &SensorSimulator::injectGPSFailure)
        .def("inject_baro_failure", &SensorSimulator::injectBaroFailure)
        .def("inject_mag_failure", &SensorSimulator::injectMagFailure)
        .def("inject_cable_failure", &SensorSimulator::injectCableFailure)
        .def("inject_gps_spoof", &SensorSimulator::injectGPSSpoof)
        .def("clear_all_failures", &SensorSimulator::clearAllFailures);
    
    py::class_<StateEstimator>(m, "StateEstimator")
        .def(py::init<>())
        .def(py::init<const SensorNoise&>())
        .def("reset", py::overload_cast<>(&StateEstimator::reset))
        .def("reset", py::overload_cast<const State&>(&StateEstimator::reset))
        .def("update", &StateEstimator::update)
        .def("get_estimated_state", &StateEstimator::getEstimatedState)
        .def("get_last_sensor_reading", &StateEstimator::getLastSensorReading)
        .def("set_noise", &StateEstimator::setNoise)
        .def("get_noise", &StateEstimator::getNoise)
        .def("get_position_error", &StateEstimator::getPositionError)
        .def("get_velocity_error", &StateEstimator::getVelocityError)
        .def("get_orientation_error", &StateEstimator::getOrientationError)
        .def("is_initialized", &StateEstimator::isInitialized);
    
    py::enum_<FailureType>(m, "FailureType")
        .value("NONE", FailureType::NONE)
        .value("STUCK", FailureType::STUCK)
        .value("REDUCED_POWER", FailureType::REDUCED_POWER)
        .value("OSCILLATING", FailureType::OSCILLATING)
        .value("DELAYED_RESPONSE", FailureType::DELAYED_RESPONSE)
        .value("RANDOM_CUTOUT", FailureType::RANDOM_CUTOUT)
        .value("THERMAL_DEGRADATION", FailureType::THERMAL_DEGRADATION)
        .value("PARTIAL_LOSS", FailureType::PARTIAL_LOSS)
        .value("COMPLETE_LOSS", FailureType::COMPLETE_LOSS);
    
    py::enum_<FailureMode>(m, "FailureMode")
        .value("PERMANENT", FailureMode::PERMANENT)
        .value("INTERMITTENT", FailureMode::INTERMITTENT)
        .value("PROGRESSIVE", FailureMode::PROGRESSIVE);
    
    py::class_<FailureConfig>(m, "FailureConfig")
        .def(py::init<>())
        .def_readwrite("type", &FailureConfig::type)
        .def_readwrite("mode", &FailureConfig::mode)
        .def_readwrite("motor_id", &FailureConfig::motor_id)
        .def_readwrite("severity", &FailureConfig::severity)
        .def_readwrite("stuck_rpm", &FailureConfig::stuck_rpm)
        .def_readwrite("oscillation_freq", &FailureConfig::oscillation_freq)
        .def_readwrite("oscillation_amplitude", &FailureConfig::oscillation_amplitude)
        .def_readwrite("delay_factor", &FailureConfig::delay_factor)
        .def_readwrite("cutout_probability", &FailureConfig::cutout_probability)
        .def_readwrite("thermal_rate", &FailureConfig::thermal_rate)
        .def_readwrite("intermittent_on_time", &FailureConfig::intermittent_on_time)
        .def_readwrite("intermittent_off_time", &FailureConfig::intermittent_off_time)
        .def_readwrite("progressive_rate", &FailureConfig::progressive_rate)
        .def_readwrite("start_time", &FailureConfig::start_time)
        .def_readwrite("duration", &FailureConfig::duration);
    
    py::class_<ActuatorFailureInjector>(m, "ActuatorFailureInjector")
        .def(py::init<>())
        .def("inject_failure", py::overload_cast<const FailureConfig&>(&ActuatorFailureInjector::injectFailure))
        .def("inject_failure", py::overload_cast<int, FailureType, double>(&ActuatorFailureInjector::injectFailure))
        .def("clear_failure", &ActuatorFailureInjector::clearFailure)
        .def("clear_all_failures", &ActuatorFailureInjector::clearAllFailures)
        .def("schedule_failure", &ActuatorFailureInjector::scheduleFailure)
        .def("apply_failures", [](ActuatorFailureInjector& self,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> rpm,
                                  double dt) {
            self.applyFailures(rpm.mutable_data(), dt);
        })
        .def("reset", &ActuatorFailureInjector::reset)
        .def("has_active_failure", py::overload_cast<>(&ActuatorFailureInjector::hasActiveFailure, py::const_))
        .def("has_active_failure", py::overload_cast<int>(&ActuatorFailureInjector::hasActiveFailure, py::const_))
        .def("get_failure_type", &ActuatorFailureInjector::getFailureType)
        .def("get_failure_severity", &ActuatorFailureInjector::getFailureSeverity)
        .def("set_random_failure_probability", &ActuatorFailureInjector::setRandomFailureProbability)
        .def("enable_random_failures", &ActuatorFailureInjector::enableRandomFailures);
    
    py::enum_<FailureScenarioGenerator::Scenario>(m, "FailureScenario")
        .value("SINGLE_MOTOR_LOSS", FailureScenarioGenerator::Scenario::SINGLE_MOTOR_LOSS)
        .value("DUAL_MOTOR_LOSS", FailureScenarioGenerator::Scenario::DUAL_MOTOR_LOSS)
        .value("OPPOSITE_MOTOR_LOSS", FailureScenarioGenerator::Scenario::OPPOSITE_MOTOR_LOSS)
        .value("ADJACENT_MOTOR_LOSS", FailureScenarioGenerator::Scenario::ADJACENT_MOTOR_LOSS)
        .value("PROGRESSIVE_DEGRADATION", FailureScenarioGenerator::Scenario::PROGRESSIVE_DEGRADATION)
        .value("RANDOM_INTERMITTENT", FailureScenarioGenerator::Scenario::RANDOM_INTERMITTENT)
        .value("THERMAL_RUNWAY", FailureScenarioGenerator::Scenario::THERMAL_RUNWAY)
        .value("CASCADING_FAILURE", FailureScenarioGenerator::Scenario::CASCADING_FAILURE);
    
    py::class_<FailureScenarioGenerator>(m, "FailureScenarioGenerator")
        .def_static("generate_scenario", &FailureScenarioGenerator::generateScenario,
                    py::arg("injector"), py::arg("scenario"), py::arg("severity") = 1.0)
        .def_static("single_motor_loss", &FailureScenarioGenerator::singleMotorLoss)
        .def_static("dual_motor_loss", &FailureScenarioGenerator::dualMotorLoss)
        .def_static("opposite_motor_loss", &FailureScenarioGenerator::oppositeMotorLoss)
        .def_static("adjacent_motor_loss", &FailureScenarioGenerator::adjacentMotorLoss)
        .def_static("progressive_degradation", &FailureScenarioGenerator::progressiveDegradation)
        .def_static("random_intermittent", &FailureScenarioGenerator::randomIntermittent)
        .def_static("thermal_runaway", &FailureScenarioGenerator::thermalRunaway)
        .def_static("cascading_failure", &FailureScenarioGenerator::cascadingFailure);
    
    py::enum_<ColliderType>(m, "ColliderType")
        .value("SPHERE", ColliderType::SPHERE)
        .value("AABB", ColliderType::AABB)
        .value("CYLINDER", ColliderType::CYLINDER)
        .value("PLANE", ColliderType::PLANE);
    
    py::class_<AABB>(m, "AABB")
        .def(py::init<>())
        .def(py::init<const Vec3&, const Vec3&>())
        .def_static("from_center_size", &AABB::fromCenterSize)
        .def_readwrite("min", &AABB::min)
        .def_readwrite("max", &AABB::max)
        .def("center", &AABB::center)
        .def("size", &AABB::size)
        .def("contains", &AABB::contains)
        .def("intersects", &AABB::intersects)
        .def("distance_to", &AABB::distanceTo)
        .def("closest_point", &AABB::closestPoint)
        .def("expanded", &AABB::expanded);
    
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def(py::init<const Vec3&, double>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def("contains", &Sphere::contains)
        .def("intersects", py::overload_cast<const Sphere&>(&Sphere::intersects, py::const_))
        .def("distance_to", &Sphere::distanceTo);
    
    py::class_<CollisionResult>(m, "CollisionResult")
        .def(py::init<>())
        .def_readwrite("collision", &CollisionResult::collision)
        .def_readwrite("collider_id", &CollisionResult::collider_id)
        .def_readwrite("distance", &CollisionResult::distance)
        .def_readwrite("closest_point", &CollisionResult::closest_point)
        .def_readwrite("normal", &CollisionResult::normal)
        .def_readwrite("penetration", &CollisionResult::penetration);
    
    py::class_<RaycastResult>(m, "RaycastResult")
        .def(py::init<>())
        .def_readwrite("hit", &RaycastResult::hit)
        .def_readwrite("collider_id", &RaycastResult::collider_id)
        .def_readwrite("distance", &RaycastResult::distance)
        .def_readwrite("hit_point", &RaycastResult::hit_point)
        .def_readwrite("normal", &RaycastResult::normal);
    
    py::class_<CollisionWorld>(m, "CollisionWorld")
        .def(py::init<>())
        .def("add_sphere", &CollisionWorld::addSphere)
        .def("add_aabb", &CollisionWorld::addAABB)
        .def("add_cylinder", &CollisionWorld::addCylinder)
        .def("add_ground_plane", &CollisionWorld::addGroundPlane, py::arg("height") = 0.0)
        .def("add_building", &CollisionWorld::addBuilding)
        .def("add_tree", &CollisionWorld::addTree)
        .def("add_pole", &CollisionWorld::addPole)
        .def("check_collision", &CollisionWorld::checkCollision)
        .def("has_collision", &CollisionWorld::hasCollision)
        .def("distance_to_nearest", &CollisionWorld::distanceToNearest)
        .def("nearest_collider_id", &CollisionWorld::nearestColliderId)
        .def("raycast", &CollisionWorld::raycast, py::arg("origin"), py::arg("direction"), py::arg("max_distance") = 1000.0)
        .def("line_of_sight", &CollisionWorld::lineOfSight)
        .def("get_colliders_in_radius", &CollisionWorld::getCollidersInRadius)
        .def("generate_random_obstacles", &CollisionWorld::generateRandomObstacles)
        .def("generate_urban_environment", &CollisionWorld::generateUrbanEnvironment)
        .def("generate_forest_environment", &CollisionWorld::generateForestEnvironment)
        .def("get_collider_count", &CollisionWorld::getColliderCount)
        .def("clear_colliders", &CollisionWorld::clearColliders)
        .def("reset", &CollisionWorld::reset);
    
    py::enum_<CableState>(m, "CableState")
        .value("SLACK", CableState::SLACK)
        .value("TENSIONED", CableState::TENSIONED)
        .value("STRETCHED", CableState::STRETCHED);
    
    py::class_<CableConfig>(m, "CableConfig")
        .def(py::init<>())
        .def_readwrite("rest_length", &CableConfig::rest_length)
        .def_readwrite("compliance", &CableConfig::compliance)
        .def_readwrite("damping", &CableConfig::damping)
        .def_readwrite("linear_density", &CableConfig::linear_density)
        .def_readwrite("num_segments", &CableConfig::num_segments)
        .def_readwrite("drag_coefficient", &CableConfig::drag_coefficient)
        .def_readwrite("diameter", &CableConfig::diameter)
        .def_readwrite("enable_drag", &CableConfig::enable_drag)
        .def_readwrite("enable_catenary", &CableConfig::enable_catenary)
        .def_readwrite("max_strain", &CableConfig::max_strain);
    
    py::class_<PayloadConfig>(m, "PayloadConfig")
        .def(py::init<>())
        .def_readwrite("mass", &PayloadConfig::mass)
        .def_readwrite("inertia", &PayloadConfig::inertia)
        .def_readwrite("drag_area", &PayloadConfig::drag_area)
        .def_readwrite("drag_coefficient", &PayloadConfig::drag_coefficient)
        .def_readwrite("center_of_mass", &PayloadConfig::center_of_mass)
        .def_readwrite("restitution", &PayloadConfig::restitution)
        .def_readwrite("friction", &PayloadConfig::friction);
    
    py::class_<PayloadState>(m, "PayloadState")
        .def(py::init<>())
        .def(py::init<const Vec3&, const Vec3&>())
        .def_readwrite("position", &PayloadState::position)
        .def_readwrite("velocity", &PayloadState::velocity)
        .def_readwrite("orientation", &PayloadState::orientation)
        .def_readwrite("angular_velocity", &PayloadState::angular_velocity);
    
    py::class_<CableForces>(m, "CableForces")
        .def(py::init<>())
        .def_readonly("tension_force", &CableForces::tension_force)
        .def_readonly("damping_force", &CableForces::damping_force)
        .def_readonly("drag_force", &CableForces::drag_force)
        .def_readonly("tension_magnitude", &CableForces::tension_magnitude)
        .def_readonly("strain", &CableForces::strain)
        .def_readonly("state", &CableForces::state)
        .def("total", &CableForces::total);
    
    py::class_<PayloadForces>(m, "PayloadForces")
        .def(py::init<>())
        .def_readonly("gravity", &PayloadForces::gravity)
        .def_readonly("cable_tension", &PayloadForces::cable_tension)
        .def_readonly("aerodynamic_drag", &PayloadForces::aerodynamic_drag)
        .def_readonly("ground_reaction", &PayloadForces::ground_reaction)
        .def_readonly("total_force", &PayloadForces::total_force)
        .def_readonly("total_torque", &PayloadForces::total_torque);
    
    py::class_<DronePayloadCoupling>(m, "DronePayloadCoupling")
        .def(py::init<>())
        .def_readonly("force_on_drone", &DronePayloadCoupling::force_on_drone)
        .def_readonly("torque_on_drone", &DronePayloadCoupling::torque_on_drone)
        .def_readonly("total_tension", &DronePayloadCoupling::total_tension)
        .def_readonly("payload_attached", &DronePayloadCoupling::payload_attached)
        .def_readonly("total_energy", &DronePayloadCoupling::total_energy);
    
    py::enum_<IntegratorType>(m, "IntegratorType")
        .value("VERLET", IntegratorType::VERLET)
        .value("RK4", IntegratorType::RK4)
        .value("XPBD", IntegratorType::XPBD);
    
    py::class_<UnifiedXPBDSystem>(m, "XPBDCable")
        .def(py::init<>())
        .def(py::init<const CableConfig&, const PayloadConfig&>())
        .def("initialize", &UnifiedXPBDSystem::initialize)
        .def("prestabilize", &UnifiedXPBDSystem::prestabilize)
        .def("step", &UnifiedXPBDSystem::step, py::arg("anchor_pos"), py::arg("anchor_vel"), 
             py::arg("dt"), py::arg("wind") = Vec3(), py::arg("ground_height") = 0.0)
        .def("get_anchor_force", &UnifiedXPBDSystem::getAnchorForce)
        .def("get_state", &UnifiedXPBDSystem::getState)
        .def("get_tension", &UnifiedXPBDSystem::getTension)
        .def("get_strain", &UnifiedXPBDSystem::getStrain)
        .def("get_total_length", &UnifiedXPBDSystem::getTotalLength)
        .def("get_particle_count", &UnifiedXPBDSystem::getParticleCount)
        .def("get_payload_particle_index", &UnifiedXPBDSystem::getPayloadParticleIndex)
        .def("get_payload_position", &UnifiedXPBDSystem::getPayloadPosition)
        .def("get_payload_velocity", &UnifiedXPBDSystem::getPayloadVelocity)
        .def("get_kinetic_energy", &UnifiedXPBDSystem::getKineticEnergy)
        .def("get_potential_energy", &UnifiedXPBDSystem::getPotentialEnergy, py::arg("ground_height") = 0.0)
        .def("get_total_energy", &UnifiedXPBDSystem::getTotalEnergy, py::arg("ground_height") = 0.0)
        .def("reset", &UnifiedXPBDSystem::reset);
    
    py::class_<PayloadDynamics>(m, "PayloadDynamics")
        .def(py::init<>())
        .def(py::init<const PayloadConfig&, const CableConfig&>())
        .def("step", &PayloadDynamics::step, py::arg("drone_position"), py::arg("drone_orientation"),
             py::arg("drone_velocity"), py::arg("drone_angular_velocity"), py::arg("dt"), 
             py::arg("air_density") = 1.225)
        .def("get_coupling_forces", &PayloadDynamics::getCouplingForces)
        .def("attach", &PayloadDynamics::attach, py::arg("initial_offset") = Vec3(0, 0, -1.0))
        .def("detach", &PayloadDynamics::detach)
        .def("is_attached", &PayloadDynamics::isAttached)
        .def("add_attachment_point", &PayloadDynamics::addAttachmentPoint)
        .def("clear_attachment_points", &PayloadDynamics::clearAttachmentPoints)
        .def("get_attachment_count", &PayloadDynamics::getAttachmentCount)
        .def("get_payload_state", &PayloadDynamics::getPayloadState)
        .def("get_payload_forces", &PayloadDynamics::getPayloadForces)
        .def("set_payload_config", &PayloadDynamics::setPayloadConfig)
        .def("set_cable_config", &PayloadDynamics::setCableConfig)
        .def("get_payload_config", &PayloadDynamics::getPayloadConfig)
        .def("get_cable_config", &PayloadDynamics::getCableConfig)
        .def("set_ground_height", &PayloadDynamics::setGroundHeight)
        .def("set_wind_velocity", &PayloadDynamics::setWindVelocity)
        .def("set_integrator", &PayloadDynamics::setIntegrator)
        .def("reset", py::overload_cast<>(&PayloadDynamics::reset))
        .def("reset", py::overload_cast<const PayloadState&>(&PayloadDynamics::reset))
        .def("get_kinetic_energy", &PayloadDynamics::getKineticEnergy)
        .def("get_potential_energy", &PayloadDynamics::getPotentialEnergy)
        .def("get_cable_energy", &PayloadDynamics::getCableEnergy)
        .def("get_total_energy", &PayloadDynamics::getTotalEnergy)
        .def("get_swing_angle", &PayloadDynamics::getSwingAngle)
        .def("get_cable_angle_from_vertical", &PayloadDynamics::getCableAngleFromVertical)
        .def("get_natural_frequency", &PayloadDynamics::getNaturalFrequency);
    
    py::class_<SwingingPayloadController::LQRGains>(m, "LQRGains")
        .def(py::init<>())
        .def_readwrite("max_compensation", &SwingingPayloadController::LQRGains::max_compensation)
        .def_readwrite("damping_ratio", &SwingingPayloadController::LQRGains::damping_ratio);
    
    py::class_<SwingingPayloadController>(m, "SwingingPayloadController")
        .def(py::init<>())
        .def(py::init<const SwingingPayloadController::LQRGains&>())
        .def("compute_compensation", &SwingingPayloadController::computeCompensation)
        .def("compute_input_shaping", &SwingingPayloadController::computeInputShaping)
        .def("compute_energy_based_control", &SwingingPayloadController::computeEnergyBasedControl)
        .def("compute_compensation_from_angles", &SwingingPayloadController::computeCompensationFromAngles)
        .def_static("estimate_payload_from_cable_angles", &SwingingPayloadController::estimatePayloadFromCableAngles)
        .def("set_gains", &SwingingPayloadController::setGains)
        .def("get_gains", &SwingingPayloadController::getGains)
        .def("reset", &SwingingPayloadController::reset);
}