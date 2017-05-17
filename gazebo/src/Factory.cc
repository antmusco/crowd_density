#include "Factory.hh"
#include <random>
#include <fstream>

using namespace gazebo;

// List of object quantities to place in the scene.
#define OBJ_TYPE_COUNT 3
const struct ObjectQuantity OBJS[] = {
  {20, 12, "cone", "construction_cone/meshes/construction_cone.dae"},
  {10,  1, "blue", "drc_practice_blue_cylinder/meshes/cylinder.dae"},
  { 5,  1, "dump", "dumpster/meshes/dumpster.dae"},
};

// Initialize static arrays.
std::string Factory::person_visuals[NUM_PERSONS];
std::vector<HeisenPerson*> Factory::persons(NUM_PERSONS, nullptr);
std::vector<HeisenPerson*> Factory::objects;
HeisenCamera* Factory::camera = nullptr;

// Default constructor.
Factory::Factory() {
  // Empty.
}

// Default destructor
Factory::~Factory() {
  // Empty.
}

/**
 * Loads the factory plugin into gazebo.
 */
void Factory::Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/) {

  ////////////////////////////////
  // Insert people into the world
  ////////////////////////////////

  // Load the model sdf string.
  std::ifstream person_ifs("gazebo/models/heisen_person/model.sdf");
  std::string person_sdf_str(
      (std::istreambuf_iterator<char>(person_ifs)),
      (std::istreambuf_iterator<char>())
  );
  
  // Extract the appropriate pointers for the model and the visual.
  sdf::SDF person_sdf;
  person_sdf.SetFromString(person_sdf_str);
  sdf::ElementPtr person_model  = person_sdf.Root()->GetElement("model");
  sdf::ElementPtr person_link   = person_model->GetElement("link");
  sdf::ElementPtr person_visual = person_link->GetElement("visual");

  // Models, visuals, and links will be named by number prefixed with these
  // strings.
  std::string base_model_name("heisen_person_");
  std::string base_visual_name("heisen_person_visual_");
  std::string link_name = person_link->GetAttribute("name")->GetAsString();
  printf("Loaded all persons.\n");

  // Insert NUM_PERSONS models into the gazebo.
  for(int i = 0; i < NUM_PERSONS; ++i) {
    // Construct the appropriate names.
    std::string index       = std::to_string(i+1);
    std::string model_name  = base_model_name + index;
    std::string visual_name = base_visual_name + index;
    // Update the model and visuall name for this person.
    person_model->GetAttribute("name")->SetFromString(model_name);
    person_visual->GetAttribute("name")->SetFromString(visual_name);
    // Add the fully qualified visual name to the list.
    person_visuals[i] = model_name + "::" + link_name + "::" + visual_name;
    // Add the model to the world.
    _parent->InsertModelSDF(person_sdf);
  }

  ////////////////////////////////////////
  // Insert random objects into the world
  ////////////////////////////////////////

  // Collison mesh and visual mesh will be named by number prefixed with these
  // strings.
  sdf::ElementPtr collision_name = person_link->GetElement("collision");
  sdf::ElementPtr collision_mesh = collision_name->GetElement("geometry")
      ->GetElement("mesh");
  sdf::ElementPtr visual_mesh = person_link->GetElement("visual")
      ->GetElement("geometry")->GetElement("mesh");

  std::string mesh_base = "model://";

  // For each object type, introduce the indicated count number into the scene.
  for(int i = 0; i < OBJ_TYPE_COUNT; ++i) {
    for(int j = 0; j < OBJS[i].num; ++j) {
      int index = j+1;
      // Construct the model name.
      std::string model_name = std::string(OBJS[i].name) + std::to_string(index);
      // Set the model, visual, and collision names.
      person_model->GetAttribute("name")->SetFromString(model_name);
      person_visual->GetAttribute("name")->SetFromString(model_name + "_visual");
      collision_name->GetAttribute("name")->SetFromString(model_name + "_collision");
      // Set the collision mesh uri.
      collision_mesh->GetElement("uri")->Set(std::string(mesh_base + OBJS[i].mesh));
      visual_mesh->GetElement("uri")->Set(std::string(mesh_base + OBJS[i].mesh));
      if(OBJS[i].scale > 0) {
        std::string scale = std::to_string(OBJS[i].scale);
        sdf::ElementPtr collision_scale = (collision_mesh->HasElement("scale")) ?
          collision_mesh->GetElement("scale"):
          collision_mesh->AddElement("scale");
        sdf::ElementPtr visual_scale = (visual_mesh->HasElement("scale")) ?
          visual_mesh->GetElement("scale"):
          visual_mesh->AddElement("scale");
        // Set the visual mesh uri and scale.
        collision_scale->Set(scale + " " + scale + " " + scale);
        visual_scale->Set(scale + " " + scale + " " + scale);
      }
      // Add the model to the world.
      _parent->InsertModelSDF(person_sdf);
    }
  }

  ////////////////////////////////////////
  // Insert single camera into the world.
  ////////////////////////////////////////

  // Load camera sdf as a string.
  std::ifstream cam_ifs("gazebo/models/heisen_camera/model.sdf");
  std::string cam_sdf_str(
      (std::istreambuf_iterator<char>(cam_ifs)),
      (std::istreambuf_iterator<char>())
  );
  // Construct the camera sdf from the string.
  sdf::SDF camera_sdf;
  camera_sdf.SetFromString(cam_sdf_str);
  // Update the cameral model name.
  sdf::ElementPtr camera_model = camera_sdf.Root()->GetElement("model");
  camera_model->GetAttribute("name")->SetFromString("heisen_camera");
  // Add the camera to the world.
  _parent->InsertModelSDF(camera_sdf);

}

// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(Factory)
