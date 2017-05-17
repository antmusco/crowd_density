#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>
#include <random>
#include <cstdlib>
#include <math.h>

#define MEAN 0.0
#define VAR  18.0
#define GAMMA_SHAPE 4
#define GAMMA_SCALE 3

namespace gazebo {

  // Convert a 3-D rotation matrix to a quaternion.
  inline math::Quaternion mat3toQuaternion(math::Matrix3& rot) {
    double w = sqrt(1.0 + rot[0][0] + rot[1][1] + rot[2][2]) / 2.0;
    double w4 = 4 * w;
    double x = (rot[2][1] - rot[1][2]) / w4;
    double y = (rot[0][2] - rot[2][0]) / w4;
    double z = (rot[1][0] - rot[0][1]) / w4;
    return math::Quaternion(w, x, y, z);
  }
  
  // Camera plugin that relocates every
  class HeisenCamera : public ModelPlugin {

    // Default constructor.
    public: HeisenCamera();

    // Loads the plugin for the model.
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/);

    // Called by the world update start event
    public: void OnUpdate(const common::UpdateInfo & /*_info*/);

    // Places the model randomly (gaussian) around the origin.
    public: void updatePose();

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;

    // Random number generator.
    private: std::default_random_engine       rand_generator;

    // Gaussian distribution for camera distance.
    private: std::normal_distribution<double> normal_dist;

    // Gamma distribution for camera height.
    private: std::gamma_distribution<double>  gamma_dist;

  };

}
