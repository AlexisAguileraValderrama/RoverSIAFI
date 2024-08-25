#include <gazebo/gazebo.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/Model.hh> // Ensure this header is included
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

namespace gazebo
{
  class ModelRelocatorPlugin : public WorldPlugin
  {
    public:
      ModelRelocatorPlugin() : WorldPlugin() {}

      void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) override
      {
        this->world = _world;

        // Initialize ROS 2 node
        rclcpp::init(0, nullptr);
        this->node = std::make_shared<rclcpp::Node>("model_relocator_plugin");

        // Create a ROS 2 subscriber to listen to pose messages
        this->pose_sub = this->node->create_subscription<geometry_msgs::msg::Pose>(
          "/model_pose", 10,
          [this](const geometry_msgs::msg::Pose::SharedPtr msg) {
            this->OnPoseMessage(msg);
          });

        // Launch a separate thread to run ROS 2 spinning
        this->ros_spin_thread = std::thread([this]() {
          rclcpp::spin(this->node);
        });

        RCLCPP_INFO(this->node->get_logger(), "Model Relocator Plugin Loaded");
      }

      ~ModelRelocatorPlugin() override
      {
        rclcpp::shutdown();
        if (this->ros_spin_thread.joinable())
        {
          this->ros_spin_thread.join();
        }
      }

      void OnPoseMessage(const geometry_msgs::msg::Pose::SharedPtr msg)
      {
        std::string model_name = "your_model_name";  // Replace with your model name
        auto model = this->world->ModelByName(model_name);
        if (model)
        {
          // Set the model's new pose
          ignition::math::Pose3d new_pose(
            msg->position.x,
            msg->position.y,
            msg->position.z,
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z,
            msg->orientation.w
          );
          model.SetWorldPose(new_pose);
          RCLCPP_INFO(this->node->get_logger(), "Model %s moved to new pose", model_name.c_str());
        }
        else
        {
          RCLCPP_WARN(this->node->get_logger(), "Model %s not found", model_name.c_str());
        }
      }

    private:
      physics::WorldPtr world;
      std::shared_ptr<rclcpp::Node> node;
      rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub;
      std::thread ros_spin_thread;
  };

  GZ_REGISTER_WORLD_PLUGIN(ModelRelocatorPlugin)
}

