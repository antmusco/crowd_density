#ifndef FACTORY_PLUGIN_H
#define FACTORY_PLUGIN_H

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"
#include <string>
#include <vector>

#define NUM_PERSONS 50

namespace gazebo {

// Forward declare HeisenPerson.
class HeisenPerson;
class HeisenCamera;

struct ObjectQuantity {
  unsigned int num;
  float        scale;
  const char*  name;
  const char*  mesh;
};


class Factory : public WorldPlugin {

  public: Factory();
  public: ~Factory();
  public: void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/);
  // Static array of visuals.
  public: static std::string person_visuals[];
  public: static std::vector<HeisenPerson*> persons;
  public: static std::vector<HeisenPerson*> objects;
  public: static HeisenCamera* camera;

};

}

#endif
