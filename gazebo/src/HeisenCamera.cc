#include "Factory.hh"
#include "HeisenCamera.hh"

using namespace gazebo;

// Convert a 3-D rotation matrix to a quaternion.
inline math::Quaternion mat3toQuaternion(math::Matrix3& rot) {
  double w = sqrt(1.0 + rot[0][0] + rot[1][1] + rot[2][2]) / 2.0;
  double w4 = 4 * w;
  double x = (rot[2][1] - rot[1][2]) / w4;
  double y = (rot[0][2] - rot[2][0]) / w4;
  double z = (rot[1][0] - rot[0][1]) / w4;
  return math::Quaternion(w, x, y, z);
}

HeisenCamera::HeisenCamera() : 
    rand_generator(std::random_device{}()), 
    normal_dist(MEAN, VAR),
    gamma_dist(GAMMA_SHAPE, GAMMA_SCALE) {

  // Empty.

}

void HeisenCamera::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) {

  printf("Loading heisen_camera plugin..\n");
  // Store the pointer to the model
  this->model = _parent;
  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
      boost::bind(&HeisenCamera::OnUpdate, this, _1));

  // Update the factory camera.
  Factory::camera = this;

  // Initialzie the pose randomly.
  this->updatePose();

}

// Called by the world update start event
void HeisenCamera::OnUpdate(const common::UpdateInfo & /*_info*/) {
  // Do nothing.
}

// Places the model randomly (gaussian) around the origin.
void HeisenCamera::updatePose() {

  // Generate x, y, and z coordinates.
  double x = this->normal_dist(this->rand_generator);
  double y = this->normal_dist(this->rand_generator);
  double z = this->gamma_dist(this->rand_generator);

  // Calculate quaternion and rotation angle to look at 
  // origin.
  math::Vector3 pose(x, y, z);
  math::Vector3 x_axis, y_axis, z_axis(0, 0, 1);
  x_axis = -pose;
  x_axis = x_axis.Normalize();
  y_axis = z_axis.Cross(x_axis);
  z_axis = x_axis.Cross(y_axis);
  // Construct rotation matrix.
  math::Matrix3 rot_mat;
  rot_mat.SetFromAxes(x_axis, y_axis, z_axis);
  // Generate quaternion.
  math::Quaternion quat = mat3toQuaternion(rot_mat);
  math::Pose newPose(pose, quat);
  this->model->SetWorldPose(newPose, true, true);

}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(HeisenCamera)
