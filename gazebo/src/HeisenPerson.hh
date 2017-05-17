#ifndef HEISEN_PERSON_PLUGIN_H
#define HEISEN_PERSON_PLUGIN_H

#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <stdio.h>
#include <random>

#include "Factory.hh"

#define X_MEAN 0.0
#define X_VAR  15.0
#define Y_MEAN 0.0
#define Y_VAR  15.0
#define W_LO   0.0
#define W_HI   6.28318

namespace gazebo {
  
/**
 * Person model that spontaneously teleoprts every second.
 */
class HeisenPerson : public ModelPlugin {

  /**
   * Default constructor.
   */
  public: HeisenPerson();

  /**
   * Loads the HeisenPerson plugin for the model.
   */
  public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf);

  // Called by the world update start event
  public: void OnUpdate(const common::UpdateInfo & /*_info*/);

  /// Returns a marker for this person around waist-height.
  public: ignition::math::Vector3d getMarker() const;

  // Places the model randomly around the origin, with a random orientation.
  public: void updatePose();

  // Pointer to the model
  private: physics::ModelPtr model;

  // Pointer to the update event connection
  private: event::ConnectionPtr updateConnection;

  // Random number generator.
  private: std::default_random_engine rand_generator;

  // Normal distribution for positon about the X-Y plane.
  private: std::normal_distribution<double> x_dist;
  private: std::normal_distribution<double> y_dist;

  // Uniform distribution between [0, 2pi) for orientation.
  private: std::uniform_real_distribution<double> w_dist;

  // The position of the person (x, y, w).
  private: ignition::math::Vector3d pos;

  // Index of this person in the static Factory arrays.
  private: unsigned int factory_index;

};

}

#endif
