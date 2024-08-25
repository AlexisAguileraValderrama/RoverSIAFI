
#include <gazebo/gazebo.hh>
#include <rclcpp/rclcpp.hpp>
#include <gazebo_msgs/msg/entity_state.hpp>
#include <gazebo/physics/World.hh>
#include <std_msgs/msg/string.hpp>
#include <gazebo/physics/Model.hh>  // Include this header for Model

namespace gazebo
{
  class WorldPluginExample : public WorldPlugin
  {
  public:
    WorldPluginExample() : WorldPlugin()
    {
      // Constructor
    }

    void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) override
    {
      if (!rclcpp::ok())
      {
        rclcpp::init(0, nullptr);
      }
      node_ = rclcpp::Node::make_shared("world_plugin_node");

      auto callback = [this](const gazebo_msgs::msg::EntityState::SharedPtr msg) {
        this->RelocateModel(msg);
      };
      
      auto callback_remove = [this](const std_msgs::msg::String::SharedPtr msg) {
        this->OnRemoveModel(msg);
      };
      
      auto callback_spawn = [this](const std_msgs::msg::String::SharedPtr msg) {
        this->OnSpawnModel(msg);
      };
      
      auto qos = rclcpp::QoS(rclcpp::KeepAll());
      qos.reliable();
      
      subscriber_ = node_->create_subscription<gazebo_msgs::msg::EntityState>(
        "/gazebo/mgp/fast_relocate", qos, callback);

      rem_subscriber_ = node_->create_subscription<std_msgs::msg::String>(
        "/gazebo/mgp/fast_remove", qos, callback_remove);

      spawn_subscriber_ = node_->create_subscription<std_msgs::msg::String>(
        "/gazebo/mgp/fast_spawn", qos, callback_spawn);

      spin_thread_ = std::thread([this]() {
        rclcpp::spin(this->node_);
      });

      world_ = _world;
      
      std::cout << "MyWorldPlugin loaded!" << std::endl;
    }

    ~WorldPluginExample()
    {
      rclcpp::shutdown();
      if (spin_thread_.joinable())
      {
        spin_thread_.join();
      }
    }

  private:
    void RelocateModel(const gazebo_msgs::msg::EntityState::SharedPtr msg)
    {
      // Check if model name is provided
      if (msg->name.empty())
      {
        RCLCPP_WARN(node_->get_logger(), "Model name is empty in EntityState message.");
        return;
      }

      // Get the model from the world
      auto model = world_->ModelByName(msg->name);
      if (!model)
      {
        RCLCPP_WARN(node_->get_logger(), "Model with name '%s' not found.", msg->name.c_str());
        return;
      }

      // Extract position and orientation from the message
      ignition::math::Pose3d new_pose(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z,
        msg->pose.orientation.w
      );

      // Update the model's pose
      model->SetWorldPose(new_pose);
      RCLCPP_INFO(node_->get_logger(), "Relocated model '%s' to new position.", msg->name.c_str());
    }
    
    
    void OnRemoveModel(const std_msgs::msg::String::SharedPtr msg)
    {
      std::string model_name = msg->data;
      auto model = world_->ModelByName(model_name);
      if (model)
      {
        world_->RemoveModel(model);
        RCLCPP_INFO(node_->get_logger(), "Removed model: %s", model_name.c_str());
      }
      else
      {
        RCLCPP_WARN(node_->get_logger(), "Model not found: %s", model_name.c_str());
      }
    }
    
    void OnSpawnModel(const std_msgs::msg::String::ConstSharedPtr &msg)
    {
      std::string sdf_string = msg->data;

      // Convert SDF string to a SDF element
      sdf::SDFPtr sdf(new sdf::SDF());
      sdf::init(sdf);
      sdf::readString(sdf_string, sdf);

	    // Extract the model element
	    sdf::ElementPtr modelElem = sdf->Root()->GetElement("model");

	    if (modelElem)
	    {
		// Get the name of the model
		std::string modelName = modelElem->Get<std::string>("name");

		// Insert the model into the world
		world_->InsertModelSDF(*sdf);
		RCLCPP_INFO(node_->get_logger(), "Spawning model with name:  %s", modelName.c_str());
	    }
	    else
	    {
		RCLCPP_WARN(node_->get_logger(), "No <model> element found in the SDF string.");
	    }
      
    }

    physics::WorldPtr world_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<gazebo_msgs::msg::EntityState>::SharedPtr subscriber_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr rem_subscriber_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr spawn_subscriber_;
    std::thread spin_thread_;
  };

  GZ_REGISTER_WORLD_PLUGIN(WorldPluginExample)
}

