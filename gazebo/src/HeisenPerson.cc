#include "HeisenPerson.hh"

using namespace gazebo;

/**
 * Default constructor.
 */
HeisenPerson::HeisenPerson() : 
    rand_generator(std::random_device{}()),
    x_dist(X_MEAN, X_VAR),
    y_dist(Y_MEAN, Y_VAR),
    w_dist(W_LO,   W_HI),
    pos(0.0, 0.0, 0.0),
    factory_index(0u) {

  // Empty.

}

/**
 * Loads the HeisenPerson plugin for the model.
 */
void HeisenPerson::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) {

  // Store the pointer to the model
  this->model = _parent;

  // Listen to the update event. This event is broadcast every simulation
  // iteration.
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
      boost::bind(&HeisenPerson::OnUpdate, this, _1));

  // Extract the name of this person from the SDF model. Then, get the index
  // of this person in the Factory::persons array.
  sdf::ElementPtr person_model  = _parent->GetSDF();
  sdf::ElementPtr person_link   = person_model->GetElement("link");
  sdf::ElementPtr person_visual = person_link->GetElement("visual");
  
  std::string my_model_name  = person_model->GetAttribute("name")->GetAsString();
  std::string my_link_name   = person_link->GetAttribute("name")->GetAsString();
  std::string my_visual_name = person_visual->GetAttribute("name")->GetAsString();
  std::string my_name = my_model_name + "::" + my_link_name + "::" + my_visual_name;
  for(; this->factory_index < NUM_PERSONS; ++this->factory_index) {
    if(Factory::person_visuals[this->factory_index] == my_name) {
      break;
    }
  }

  // If we found somthing, this is a person.
  if(this->factory_index < NUM_PERSONS) {
    Factory::persons[this->factory_index] = this;
  // Otherwise, this is an object.
  } else {
    Factory::objects.push_back(this);
  }

  this->model->SetRelativePose(ignition::math::Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  this->model->SetInitialRelativePose(ignition::math::Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

  // Initialzie the pose randomly.
  this->updatePose();

}

/**
 * Called by the world on every update event.
 */
void HeisenPerson::OnUpdate(const common::UpdateInfo & /*_info*/) {
  // Do nothing.
}

/**
 * Returns the 3-D world coordinates of the marker centered at the waist of
 * this person. 
 */
ignition::math::Vector3d HeisenPerson::getMarker() const {
  return ignition::math::Vector3d(pos.X(), pos.Y(), 1.25);
}

/**
 * Places the model randomly around the origin, with a random orientation.
 */
void HeisenPerson::updatePose() {
  pos.X() = this->x_dist(this->rand_generator);
  pos.Y() = this->y_dist(this->rand_generator);
  pos.Z() = this->w_dist(this->rand_generator); // Really W
  ignition::math::Pose3d newPose(pos.X(), pos.Y(), 0.0, 0.0, 0.0, pos.Z());
  this->model->SetWorldPose(newPose, true, true);
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(HeisenPerson)
