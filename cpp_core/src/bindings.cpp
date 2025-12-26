#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "Quadrotor.h"
#include "PhysicsEngine.h"
#include "ReplayBuffer.h"

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
}