#ifndef GAZEBO_PLUGINS_SNAPSHOT_HH_
#define GAZEBO_PLUGINS_SNAPSHOT_HH_

#include <string>
#include <ios>
#include <fstream>

#include <ignition/math/Vector2.hh>
#include <ignition/math/Vector3.hh>
#include "gazebo/common/Plugin.hh"
#include "gazebo/sensors/CameraSensor.hh"
#include "gazebo/rendering/Camera.hh"
#include "gazebo/util/system.hh"

namespace gazebo {

/**
 * Plugin which takes photos from a camera every second. This plugin is
 * attached to the HeisenCamera's camera sensor.
 */
class GAZEBO_VISIBLE Snapshot : public SensorPlugin {

  // Constructor/Destructor.
  public: Snapshot();
  public: virtual ~Snapshot();

  // Load the camera plugin.
  public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);

  // On every frame.
  public: virtual void OnNewFrame(const unsigned char *_image, 
    unsigned int _width, unsigned int _height, unsigned int _depth, const
    std::string &_format);

  // Update all people and objects in the scene.
  protected: void updateAll();

  // Width, height, and depth of the image.
  protected: unsigned int width, height, depth;
  // Format of the image.
  protected: std::string format;
  // Frame number (0-30, 30 per second).
  protected: unsigned int frame_no;
  // Number of pictures taken.
  protected: unsigned int pic_no;

  // Pointer to the CameraSensor.
  protected: sensors::CameraSensorPtr parentSensor;
  // Pointer to the Camera.
  protected: rendering::CameraPtr camera;

  // ???
  private: event::ConnectionPtr newFrameConnection;

  // Loaded configuration settings.
  private: std::clock_t  start;
  private: long          img_num;
  private: std::string   img_dir;
  private: std::string   img_fmt;
  private: std::string   img_ext;
  private: std::string   dot_ext;
  private: std::ofstream train_set_os;
  private: std::ofstream valid_set_os;
  private: std::ofstream test_set_os;
  private: unsigned int  train_per_10;
  private: unsigned int  valid_per_10;
  private: unsigned int  test_per_10;
  private: char          name_buf[100];

};

}
#endif
