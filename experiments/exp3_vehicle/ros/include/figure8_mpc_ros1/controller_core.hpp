#pragma once

#include "f8.h"
#include "model_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace figure8_mpc_ros1
{

using State = std::array<double, F8_NX>;
using Theta = std::array<double, F8_PARAMETERS>;

struct ControlResult
{
  bool success{false};
  double acceleration{0.0};
  double steering{0.0};
  double next_speed{0.0};
  F8Stats stats{};
  std::string error;
};

inline Theta theta_from_file(const std::string & path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open theta_file: " + path);
  }

  std::string model_id;
  input >> model_id;
  if (model_id != f8_model_id()) {
    throw std::runtime_error("theta_file was trained for a different model");
  }

  Theta theta{};
  for (double & value : theta) {
    input >> value;
    if (!input || !std::isfinite(value) || value < -6.0 || value > 6.0) {
      throw std::runtime_error("theta_file must contain four finite log parameters in [-6, 6]");
    }
  }
  std::string extra;
  if (input >> extra) {
    throw std::runtime_error("theta_file contains extra data");
  }
  return theta;
}

class ControllerCore
{
public:
  ControllerCore(const std::string & assets_directory, const Theta & theta)
  : theta_(theta)
  {
    solver_.reset(f8_create(assets_directory.c_str()));
    if (!solver_) {
      throw std::runtime_error("f8_create failed: " + std::string(f8_last_error()));
    }
    if (std::strcmp(f8_model_id(), F8_MODEL_ID) != 0 || f8_horizon() != F8_N ||
      std::abs(f8_dt() - F8_DT) > 1e-12)
    {
      throw std::runtime_error("native library and model_config.h do not match");
    }
    const int size = f8_trajectory_size();
    if (size != F8_N * (F8_NX + F8_NU) + F8_NX) {
      throw std::runtime_error("unexpected MPC trajectory layout");
    }
    guess_.resize(static_cast<std::size_t>(size));
    trajectory_.resize(static_cast<std::size_t>(size));
  }

  ControlResult solve(State state)
  {
    ControlResult result;
    if (!std::all_of(state.begin(), state.end(), [](double v) {return std::isfinite(v);})) {
      result.error = "non-finite state";
      reset_warm_start();
      return result;
    }
    if (state[3] < F8_SPEED_MIN - 0.01 || state[3] > F8_SPEED_MAX + 0.01) {
      result.error = "measured speed is outside the trained model range";
      reset_warm_start();
      return result;
    }
    state[3] = std::clamp(state[3], double(F8_SPEED_MIN), double(F8_SPEED_MAX));

    if (warm_start_valid_) {
      f8_shift_guess(trajectory_.data(), state.data(), guess_.data());
    } else {
      f8_initial_guess(state.data(), guess_.data());
    }

    const int status = f8_forward(
      solver_.get(), theta_.data(), state.data(), guess_.data(), trajectory_.data(),
      &result.stats);
    if (status != 0) {
      result.error = f8_last_error();
      reset_warm_start();
      return result;
    }

    const double raw_acceleration = trajectory_[F8_NX];
    const double raw_steering = trajectory_[F8_NX + 1];
    if (!std::isfinite(raw_acceleration) || !std::isfinite(raw_steering)) {
      result.error = "MPC returned a non-finite control";
      reset_warm_start();
      return result;
    }

    const double acceleration_lower = std::max(
      -double(F8_ACCELERATION_MAX), (F8_SPEED_MIN - state[3]) / f8_dt());
    const double acceleration_upper = std::min(
      double(F8_ACCELERATION_MAX), (F8_SPEED_MAX - state[3]) / f8_dt());
    result.acceleration = std::clamp(
      raw_acceleration, acceleration_lower, acceleration_upper);
    result.steering = std::clamp(
      raw_steering, -double(F8_STEERING_MAX), double(F8_STEERING_MAX));
    result.next_speed = std::clamp(
      state[3] + f8_dt() * result.acceleration,
      double(F8_SPEED_MIN), double(F8_SPEED_MAX));
    result.success = true;
    warm_start_valid_ = true;
    return result;
  }

  void reset_warm_start() noexcept
  {
    warm_start_valid_ = false;
  }

private:
  struct SolverDeleter
  {
    void operator()(F8Solver * solver) const noexcept
    {
      f8_destroy(solver);
    }
  };

  std::unique_ptr<F8Solver, SolverDeleter> solver_;
  Theta theta_{};
  std::vector<double> guess_;
  std::vector<double> trajectory_;
  bool warm_start_valid_{false};
};

}  // namespace figure8_mpc_ros1
