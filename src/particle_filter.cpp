/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
    particles.resize(num_particles);
    weights.resize(num_particles);

    default_random_engine gen;
    normal_distribution<> dist_x(x, std[0]);
    normal_distribution<> dist_y(y, std[1]);
    normal_distribution<> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0;
        weights[i] = 1.0;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    normal_distribution<> dist_x(0, std_pos[0]);
    normal_distribution<> dist_y(0, std_pos[1]);
    normal_distribution<> dist_theta(0, std_pos[2]);

    for (Particle &p: particles) {
        if (fabs(yaw_rate) > 0.0) {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
            p.theta += yaw_rate*delta_t;
        } else {
            p.x += velocity * delta_t * cos(p.theta);
            p.y += velocity * delta_t * sin(p.theta);
        }

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (LandmarkObs &o: observations) {
        double min_dist = numeric_limits<double>::max();
        int id = -1;
        for (LandmarkObs &p: predicted) {
            double d = dist(o.x, o.y, p.x, p.y);
            if (d < min_dist) {
                min_dist = d;
                id = p.id;
            }
        }
        o.id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (Particle &p: particles) {
        const vector<Map::single_landmark_s> &landmarks = map_landmarks.landmark_list;

        // Predicted measurements for this particle within the sensor range
        vector<LandmarkObs> preds;
        for (const Map::single_landmark_s &l: landmarks) {
            if ((fabs(l.x_f - p.x) <= sensor_range) and
                (fabs(l.y_f - p.y) <= sensor_range)) {
                    preds.push_back(LandmarkObs{ l.id_i, l.x_f, l.y_f });
                }
        }

        // Transform observations to the map coordinate system
        vector<LandmarkObs> obs_map;
        for (const LandmarkObs &obs: observations) {
            double x = obs.x*cos(p.theta) - obs.y*sin(p.theta) + p.x;
            double y = obs.x*sin(p.theta) + obs.y*cos(p.theta) + p.y;
            obs_map.push_back(LandmarkObs{ obs.id, x, y });
        }

        dataAssociation(preds, obs_map);

        p.weight = 1.0;

        for (LandmarkObs &o: obs_map) {
            double pred_x, pred_y;
            for (LandmarkObs &pred: preds) {
                if (pred.id == o.id) {
                    pred_x = pred.x;
                    pred_y = pred.y;
                    break;
                }
            }

            double sx = std_landmark[0];
            double sy = std_landmark[1];
            double w = (1 / (2*M_PI*sx*sy)) *
                        exp(-(pow(pred_x-o.x,2)/(2*pow(sx,2)) + (pow(pred_y-o.y,2)/(2*pow(sy,2)))));
            p.weight *= w;
        }
    }

    for (int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    discrete_distribution<> dist_index(weights.begin(), weights.end());

    vector<Particle> new_particles;
    new_particles.resize(num_particles);

    for (int i = 0; i < num_particles; i++) {
        int index = dist_index(gen);
        new_particles[i] = particles[index];
    }

    particles = new_particles;
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
