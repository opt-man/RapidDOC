#include "figure8_mpc_ros1/controller_core.hpp"

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <ros/ros.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>
#include <csignal>

namespace figure8_mpc_ros1
{

namespace
{
volatile std::sig_atomic_t stop_requested = 0;
void request_stop(int) { stop_requested = 1; }

bool quaternion_to_yaw(const geometry_msgs::Quaternion & q, double & yaw)
{
  const double norm_squared = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(norm_squared) || norm_squared < 1e-12) {
    return false;
  }
  const double scale = 1.0 / std::sqrt(norm_squared);
  const double x = scale * q.x;
  const double y = scale * q.y;
  const double z = scale * q.z;
  const double w = scale * q.w;
  yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  return std::isfinite(yaw);
}

}  // namespace

class Figure8Controller
{
public:
  Figure8Controller()
  : private_node_("~")
  {
    private_node_.param<std::string>(
      "pose_topic", pose_topic_, "/vrpn_client_node/limo_car0/pose");
    private_node_.param<std::string>(
      "velocity_topic", velocity_topic_, "/vrpn_client_node/limo_car0/twist");
    private_node_.param<std::string>("command_topic", command_topic_, "/cmd_vel");
    private_node_.param<std::string>("theta_file", theta_file_, "");
    private_node_.param("input_scale", input_scale_, 0.001);
    private_node_.param("state_timeout_s", state_timeout_s_, 0.25);
    private_node_.param("yaw_hold_speed_mps", yaw_hold_speed_mps_, 0.02);
    private_node_.param("reference_origin_x", reference_origin_x_, 0.0);
    private_node_.param("reference_origin_y", reference_origin_y_, 0.0);
    private_node_.param("initial_phase", initial_phase_, 0.0);
    private_node_.param("x_limit", x_limit_, 1.25);
    private_node_.param("y_limit", y_limit_, 0.75);
    std::string log_file;
    private_node_.param<std::string>("log_file", log_file, "");
    if (log_file.empty() || !std::isfinite(x_limit_) || !std::isfinite(y_limit_) ||
      x_limit_ <= 0 || y_limit_ <= 0) {
      throw std::invalid_argument("log_file and positive position limits are required");
    }
    if (std::ifstream(log_file).good()) {
      throw std::invalid_argument("log_file already exists");
    }
    log_.open(log_file);
    if (!log_) {throw std::runtime_error("cannot open log file");}
    log_ << std::setprecision(17)
         << "time,x,y,yaw,speed,phase,ref_x,ref_y,ref_yaw,ref_speed,a,delta,cmd_v,cmd_w,solve_s,status\n";
    ROS_INFO("Saving measured trajectory to %s", log_file.c_str());

    if (theta_file_.empty()) {
      throw std::invalid_argument(
              "~theta_file is required; use the theta_final.txt produced by offline training");
    }
    if (!std::isfinite(input_scale_) || input_scale_ <= 0.0 ||
      !std::isfinite(state_timeout_s_) || state_timeout_s_ <= 0.0 ||
      !std::isfinite(yaw_hold_speed_mps_) || yaw_hold_speed_mps_ < 0.0 ||
      !std::isfinite(reference_origin_x_) || !std::isfinite(reference_origin_y_) ||
      !std::isfinite(initial_phase_))
    {
      throw std::invalid_argument("invalid numeric ROS parameter");
    }

    core_.reset(new ControllerCore("", theta_from_file(theta_file_)));
    command_publisher_ = node_.advertise<geometry_msgs::Twist>(command_topic_, 1);
    pose_subscriber_ = node_.subscribe(
      pose_topic_, 1, &Figure8Controller::on_pose, this,
      ros::TransportHints().tcpNoDelay());
    velocity_subscriber_ = node_.subscribe(
      velocity_topic_, 1, &Figure8Controller::on_velocity, this,
      ros::TransportHints().tcpNoDelay());

    ROS_INFO(
      "figure8 MPC ready: model=%s, rate=%.1f Hz, theta=%s",
      f8_model_id(), 1.0 / f8_dt(), theta_file_.c_str());
    ROS_INFO(
      "topics: %s + %s -> %s; input_scale=%.9g",
      pose_topic_.c_str(), velocity_topic_.c_str(), command_topic_.c_str(), input_scale_);
  }

  ~Figure8Controller()
  {
    if (!command_publisher_.getTopic().empty()) {
      publish_stop();
    }
  }

  void run()
  {
    ros::WallRate rate(1.0 / f8_dt());

    while (ros::ok() && !stop_requested) {
      ros::spinOnce();
      const ros::WallTime now = ros::WallTime::now();
      if (halted_) {publish_stop(); rate.sleep(); continue;}
      if (!state_is_fresh(now)) {
        core_->reset_warm_start();
        publish_stop();
        ROS_WARN_THROTTLE(2.0, "waiting for fresh pose and velocity topics");
        if (reference_started_) {
          halted_ = true;
          record((now-reference_start_).toSec(), last_phase_, ControlResult{}, 0, 0, "stale_state_stop");
          ROS_ERROR("State lost during trial. Stopped; restart for a new trial.");
        }
        rate.sleep();
        continue;
      }

      if (!reference_started_) {
        double ref[5], ref_u[2];
        f8_reference(initial_phase_, ref, ref_u);
        const double yaw_error = std::atan2(std::sin(yaw_rad_-ref[2]), std::cos(yaw_rad_-ref[2]));
        if (std::hypot(position_x_m_-reference_origin_x_-ref[0], position_y_m_-reference_origin_y_-ref[1]) > 0.10 ||
          std::abs(yaw_error) > 0.15 || speed_mps_ > 0.03) {
          publish_stop();
          ROS_WARN_THROTTLE(2.0, "Waiting at start: x=%.3f y=%.3f yaw=%.3f v=%.3f; target=(0,0,1.009,0)", position_x_m_,position_y_m_,yaw_rad_,speed_mps_);
          rate.sleep(); continue;
        }
        reference_start_ = now;
        reference_started_ = true;
      }

      const double phase = initial_phase_ + (now - reference_start_).toSec() * F8_PHASE_RATE;
      last_phase_ = phase;
      const double elapsed = (now-reference_start_).toSec();
      if (std::abs(position_x_m_) >= x_limit_ || std::abs(position_y_m_) >= y_limit_) {
        publish_stop(); halted_ = true;
        record(elapsed, phase, ControlResult{}, 0, 0, "boundary_stop");
        ROS_ERROR("Position limit reached. Trial incomplete; holding zero until restart.");
        continue;
      }
      if (phase >= initial_phase_ + 2.0 * std::acos(-1.0)) {
        publish_stop();
        ROS_INFO_ONCE("One figure-eight reference period completed; holding zero command.");
        record(elapsed, phase, ControlResult{}, 0, 0, "reference_period_complete");
        halted_ = true;
        rate.sleep();
        continue;
      }
      State state{{
        position_x_m_ - reference_origin_x_,
        position_y_m_ - reference_origin_y_,
        yaw_rad_,
        speed_mps_,
        phase}};

      const ControlResult result = core_->solve(state);
      if (!result.success) {
        publish_stop();
        ROS_ERROR_THROTTLE(1.0, "MPC solve failed: %s", result.error.c_str());
        record(elapsed, phase, result, 0, 0, "solve_failed_stop");
        halted_ = true;
        rate.sleep();
        continue;
      }

      geometry_msgs::Twist command;
      command.linear.x = result.next_speed;
      command.angular.z = result.next_speed * std::tan(result.steering) / F8_WHEELBASE;
      command_publisher_.publish(command);
      record(elapsed, phase, result, command.linear.x, command.angular.z, "running");
      ROS_INFO_THROTTLE(
        1.0, "x=%.3f y=%.3f yaw=%.3f v=%.3f cmd_v=%.3f cmd_w=%.3f solve=%.3f ms",
        position_x_m_, position_y_m_, yaw_rad_, speed_mps_, command.linear.x,
        command.angular.z, 1000.0 * result.stats.forward_solve_s);
      rate.sleep();
    }

    publish_stop();
    if (reference_started_ && !halted_) {
      record((ros::WallTime::now()-reference_start_).toSec(), last_phase_, ControlResult{}, 0, 0, "interrupted");
    }
    // With NoSigintHandler, Ctrl+C leaves ROS alive long enough to send zeros.
    for (int i=0; i<5 && ros::ok(); ++i) {
      publish_stop(); ros::WallDuration(0.05).sleep();
    }
  }

private:
  void record(double time, double phase, const ControlResult & result, double v, double w, const char * status)
  {
    double ref[5], u[2]; f8_reference(phase, ref, u);
    log_ << time << ',' << position_x_m_ << ',' << position_y_m_ << ',' << yaw_rad_ << ','
         << speed_mps_ << ',' << phase << ',' << ref[0]+reference_origin_x_ << ','
         << ref[1]+reference_origin_y_ << ',' << ref[2] << ',' << ref[3] << ','
         << result.acceleration << ',' << result.steering << ',' << v << ',' << w << ','
         << result.stats.forward_solve_s << ',' << status << '\n';
    log_.flush();
    if (!log_) {throw std::runtime_error("trajectory log write failed");}
  }
  void on_pose(const geometry_msgs::PoseStamped::ConstPtr & message)
  {
    position_x_m_ = input_scale_ * message->pose.position.x;
    position_y_m_ = input_scale_ * message->pose.position.y;
    pose_yaw_valid_ = quaternion_to_yaw(message->pose.orientation, pose_yaw_rad_);
    yaw_valid_ = pose_yaw_valid_;
    if (pose_yaw_valid_) {
      yaw_rad_ = pose_yaw_rad_;
    }
    pose_received_ = std::isfinite(position_x_m_) && std::isfinite(position_y_m_);
    pose_received_at_ = ros::WallTime::now();
  }

  void on_velocity(const geometry_msgs::TwistStamped::ConstPtr & message)
  {
    const double vx = input_scale_ * message->twist.linear.x;
    const double vy = input_scale_ * message->twist.linear.y;
    const double speed = std::hypot(vx, vy);
    if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(speed)) {
      velocity_received_ = false;
      return;
    }
    speed_mps_ = speed;
    velocity_received_ = true;
    velocity_received_at_ = ros::WallTime::now();
  }

  bool state_is_fresh(const ros::WallTime & now) const
  {
    return pose_received_ && velocity_received_ && yaw_valid_ &&
           (now - pose_received_at_).toSec() <= state_timeout_s_ &&
           (now - velocity_received_at_).toSec() <= state_timeout_s_;
  }

  void publish_stop()
  {
    command_publisher_.publish(geometry_msgs::Twist{});
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Subscriber pose_subscriber_;
  ros::Subscriber velocity_subscriber_;
  ros::Publisher command_publisher_;
  std::unique_ptr<ControllerCore> core_;
  std::ofstream log_;
  double x_limit_{1.25}, y_limit_{0.75}, last_phase_{0};
  bool halted_{false};

  std::string pose_topic_;
  std::string velocity_topic_;
  std::string command_topic_;
  std::string theta_file_;
  double input_scale_{0.001};
  double state_timeout_s_{0.25};
  double yaw_hold_speed_mps_{0.02};
  double reference_origin_x_{0.0};
  double reference_origin_y_{0.0};
  double initial_phase_{0.0};
  double position_x_m_{0.0};
  double position_y_m_{0.0};
  double speed_mps_{0.0};
  double yaw_rad_{0.0};
  double pose_yaw_rad_{0.0};
  bool pose_received_{false};
  bool velocity_received_{false};
  bool yaw_valid_{false};
  bool pose_yaw_valid_{false};
  bool reference_started_{false};
  ros::WallTime pose_received_at_;
  ros::WallTime velocity_received_at_;
  ros::WallTime reference_start_;
};

}  // namespace figure8_mpc_ros1

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "figure8_mpc_controller", ros::init_options::NoSigintHandler);
  std::signal(SIGINT, figure8_mpc_ros1::request_stop);
  std::signal(SIGTERM, figure8_mpc_ros1::request_stop);
  try {
    figure8_mpc_ros1::Figure8Controller controller;
    controller.run();
  } catch (const std::exception & error) {
    ROS_FATAL("figure8 MPC terminated: %s", error.what());
    return 1;
  }
  return 0;
}
