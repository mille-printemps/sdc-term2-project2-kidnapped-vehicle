/*
 * particle_filter.cpp
 *
 * Originally created by Tiffany Huang on Dec 12, 2016
 * 
 * Original header files are used for the implemenation.
 *
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <chrono>
#include <limits.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Initialize each particle with random values using Gaussian distribution.

  num_particles = 1000;
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine random;
  random.seed(seed);

  normal_distribution<double> n_x(x, std[0]);
  normal_distribution<double> n_y(y, std[1]);
  normal_distribution<double> n_theta(theta, std[2]);

  for (int i=0; i<num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = n_x(random);
    particle.y = n_y(random);
    particle.theta = n_theta(random);
    particle.weight = 1;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  // Calculate the next position for each particle.
  // Different equations have to be used depending on the value of the yaw rate.

  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine random;
  random.seed(seed);

  for (int i=0; i<num_particles; i++) {
    double theta = particles[i].theta;

    if (yaw_rate == 0) {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
      particles[i].theta = theta;
    } else {
      particles[i].theta += yaw_rate * delta_t;

      double coefficient = velocity/yaw_rate;
      double measurement_x = coefficient * (sin(particles[i].theta) - sin(theta));
      double measurement_y = coefficient * (cos(theta) - cos(particles[i].theta));

      normal_distribution<double> n_x(0, std_pos[0]);
      normal_distribution<double> n_y(0, std_pos[1]);
      normal_distribution<double> n_theta(0, std_pos[2]);

      particles[i].x += measurement_x + n_x(random);
      particles[i].y += measurement_y + n_y(random);
      particles[i].theta += n_theta(random);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// This method is not implemented and not used.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  // Calculate the total weight of the observations for each particle.
  // Equation 3.33 is used to transform the vehicle coordinates to the map coordinates.
	// http://planning.cs.uiuc.edu/node99.html

  weights.clear();
  vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
  const int num_landmarks = landmark_list.size();

  for (int i=0; i<num_particles; i++) {
    double theta = particles[i].theta;
    double total_weight = 1;
    const int num_observations = observations.size();

    for (int j=0; j<num_observations; j++) {
      // Transform the landmark observations to the map coordinates
      double measurement_x = observations[j].x * cos(theta) - observations[j].y * sin(theta) + particles[i].x;
      double measurement_y = observations[j].x * sin(theta) + observations[j].y * cos(theta) + particles[i].y;

      // Find the nearest map landmarks based on the distance
      LandmarkObs predicted;
      double min_distance = LONG_MAX;

      for (int k=0; k<num_landmarks; k++) {
        double distance = dist(measurement_x, measurement_y, landmark_list[k].x_f, landmark_list[k].y_f);
        if (distance < min_distance) {
          min_distance = distance;
          predicted.id = landmark_list[k].id_i;
          predicted.x = landmark_list[k].x_f;
          predicted.y = landmark_list[k].y_f;
        }
      }

      // Calculate the weight for the nearest map landmark using Gaussian distribution
      double weight = 0.5/(M_PI * std_landmark[0] * std_landmark[1]) *
                      exp(-0.5 *
                          (pow((measurement_x - predicted.x)/std_landmark[0], 2.0) +
                           pow((measurement_y - predicted.y)/std_landmark[1], 2.0)));

      total_weight *= weight;
    }

    // Update the weight of the particle
    particles[i].weight = total_weight;
    weights.push_back(total_weight);
  }
}

void ParticleFilter::resample() {

  // Resample particles depending on the ratio of the weight.
  // The larger the weight is, The more likely the particle is picked up.

  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine random;
  random.seed(seed);

  discrete_distribution<> d_d(weights.begin(), weights.end());
  vector<Particle> resampled;

  for (int i=0; i<num_particles; i++) {
    int id = d_d(random);
    resampled.push_back(particles[id]);
  }

  particles.clear();
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
