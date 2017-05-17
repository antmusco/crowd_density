#include "HeisenPerson.hh"
#include "HeisenCamera.hh"
#include "Snapshot.hh"
#include "Factory.hh"
#include <OGRE/OgreVector3.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include <CImg.h>

using namespace gazebo;

#define CONFIG_FILE "config/ccnn_cfg.yml"

// Constructor.
Snapshot::Snapshot() : 
    SensorPlugin(), 
    width(0), 
    height(0), 
    depth(0), 
    frame_no(0), 
    pic_no(0),
    start(std::clock()) {
  
  // Open and parse file.
  YAML::Node node  = YAML::LoadFile(CONFIG_FILE);
  // Extract the image folder and output name list.
  this->img_num    = node["IMG_NUM"].as<long>();
  this->img_dir    = node["IMG_DIR"].as<std::string>();
  this->img_fmt    = node["IMG_FMT"].as<std::string>();
  this->img_ext    = node["IMG_EXT"].as<std::string>();
  this->dot_ext    = node["DOT_EXT"].as<std::string>();
  this->train_per_10 = (int)(node["TRAIN_PCT"].as<float>() * 10);
  this->valid_per_10 = (int)(node["VALID_PCT"].as<float>() * 10);
  this->test_per_10 = (int)(node["TEST_PCT"].as<float>() * 10);

  // Open the output streams.
  std::string img_set_fmt = node["IMG_SET_DIR"].as<std::string>() + "/" + 
      node["IMG_SET_FMT"].as<std::string>();
  // Training list.
  sprintf(this->name_buf, img_set_fmt.c_str(), "train");
  this->train_set_os = std::ofstream(this->name_buf, std::ios_base::out);
  // Validation list.
  sprintf(this->name_buf, img_set_fmt.c_str(), "valid");
  this->valid_set_os = std::ofstream(this->name_buf, std::ios_base::out);
  // Test list.
  sprintf(this->name_buf, img_set_fmt.c_str(), "test");
  this->test_set_os = std::ofstream(this->name_buf, std::ios_base::out);

}

// Destructor.
Snapshot::~Snapshot() {
  this->parentSensor.reset();
  this->camera.reset();
  this->train_set_os.close();
  this->valid_set_os.close();
  this->test_set_os.close();
}

/**
 * Loads the Snapshot into Gazebo.
 *
 * @param _sensor The sensor pointed to.
 */
void Snapshot::Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/) {

  if (!_sensor) {
    gzerr << "Invalid sensor pointer.\n";
  }

  // The sensor should be a CameraSensor, but if not error out.
  this->parentSensor = std::dynamic_pointer_cast<
      sensors::CameraSensor>(_sensor);
  if (!this->parentSensor) {
    gzerr << "Snapshot requires a CameraSensor.\n";
  }

  // Grab the camera and the camera image properties.
  this->camera = this->parentSensor->Camera();
  this->width  = this->camera->ImageWidth();
  this->height = this->camera->ImageHeight();
  this->depth  = this->camera->ImageDepth();
  this->format = this->camera->ImageFormat();

  // Bind the update function to the camera.
  this->newFrameConnection = this->camera->ConnectNewImageFrame(
      std::bind(&Snapshot::OnNewFrame, this,
        std::placeholders::_1, std::placeholders::_2, 
        std::placeholders::_3, std::placeholders::_4,
        std::placeholders::_5));

  // Activate the sensor.
  this->parentSensor->SetActive(true);

}

/**
 * On every new frame, we count the number of people we can see in the image.
 * If > 0, save the image.
 */
void Snapshot::OnNewFrame(const unsigned char * _image,
    unsigned int _width, unsigned int _height, unsigned int _depth, 
    const std::string &_format) {

  // Count the number of people we can see. Iterate over every person in Gazebo
  // and project their coordinates into the camera. If the coordinates are
  // within out bounds, we can see the person.
  int visuals_count = 0;
  std::vector<ignition::math::Vector2i> pixelCoords;
  for(auto p = Factory::persons.begin(); p != Factory::persons.end(); ++p) {
    ignition::math::Vector3d worldCoords = (*p)->getMarker();
    ignition::math::Vector2i px = this->camera->Project(worldCoords);
    if(px.X() >= 0 && px.X() < _width &&
       px.Y() >= 0 && px.Y() < _height) {
      visuals_count++;
      pixelCoords.push_back(px);
    }
  }

  // Only keep images where we see someone.
  if(visuals_count > 0) {

    // Construct base filename - to be shared between img and dot_img.
    sprintf(name_buf, this->img_fmt.c_str(), ++(this->pic_no), visuals_count);

    // Save original image.
    std::string img_name = name_buf + this->img_ext;
    rendering::Camera::SaveFrame(_image, this->width, this->height,
        this->depth, this->format, this->img_dir + img_name);

    // Draw dots on another image and save it as the 'dot image'.
             std::string dot_img_name = name_buf + this->dot_ext;
    cimg_library::CImg<unsigned char> dot_img(_height, _width, 1, 1, 0);
    for(auto p = pixelCoords.begin(); p != pixelCoords.end(); ++p) {
      dot_img(p->X(), p->Y(), 0) = 255; // R
    }
    dot_img.save((this->img_dir + dot_img_name).c_str());

    // Add file name to list.
    unsigned int mod = this->pic_no % 10;
    if(mod < this->train_per_10) {
      this->train_set_os << img_name << std::endl;
    } else if(mod < (this->train_per_10 + this->valid_per_10)) {
      printf("Validation image\n");
      this->valid_set_os << img_name << std::endl;
    } else {
      printf("Test image\n");
      this->test_set_os << img_name << std::endl;
    }

    // Update statistics.
    if(mod == 0) {
      double duration = (std::clock() - this->start) / (double) CLOCKS_PER_SEC;
      double rate     = this->pic_no / duration;
      double expected = this->img_num / rate;
      double percent  = duration / expected;
      printf("Pics %d (%3.1f%%) :: %5.3f / %5.3f seconds (projected)\n",
          this->pic_no, (percent * 100.0), duration, expected);
    }
  }

  // We've got enough.
  if(this->pic_no >= this->img_num) {
    gazebo::shutdown();
  // Change it up.
  } else {
    updateAll();
  }
 
}

/**
 * Updates poses for all people and objects, as well as the camera.
 */
void Snapshot::updateAll() {
  // Update person pose.
  for(auto p = Factory::persons.begin(); p != Factory::persons.end(); ++p) {
    (*p)->updatePose();
  }
  // Update object pose.
  for(auto o = Factory::objects.begin(); o != Factory::objects.end(); ++o) {
    (*o)->updatePose();
  }
  // Update camera pose.
  if(Factory::camera) {
    Factory::camera->updatePose();
  }
}

GZ_REGISTER_SENSOR_PLUGIN(Snapshot)
