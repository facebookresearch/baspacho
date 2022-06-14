#include "baspacho/benchmarking/BaAtLarge.h"

#include <cmath>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <set>

namespace ba_at_large {

using namespace std;

static std::ostream& operator<<(std::ostream& os,
                                const ba_at_large::Data::Obs& obs) {
  return os << "Obs[c:" << obs.camIdx << ",p:" << obs.ptIdx << ",at:("
            << obs.imgPos.x() << "," << obs.imgPos.y() << ")]";
}

static std::ostream& operator<<(std::ostream& os,
                                const ba_at_large::Data::Cam& cam) {
  Quat rot = cam.T_W_C.so3().unit_quaternion();
  Vec3 tr = cam.T_W_C.translation();
  return os << "Cam[r:q(" << rot.w() << "," << rot.x() << "," << rot.y() << ","
            << rot.z() << ")"
            << ",t:(" << tr.x() << "," << tr.y() << "," << tr.z() << ")"
            << ",f:" << cam.f_k1_k2[0] << ",k1:" << cam.f_k1_k2[1]
            << ",k2:" << cam.f_k1_k2[2] << "]";
}

static std::ostream& operator<<(std::ostream& os,
                                const ba_at_large::Vec3& vec3) {
  return os << "(" << vec3.x() << "," << vec3.y() << "," << vec3.z() << ")";
}

static Quat rodriguezToQuat(const Vec3& rRot) {
  double angle = rRot.norm();
  if (angle < 1e-12) {
    return Quat(1, 0, 0, 0);
  }
  Vec3 v = rRot / angle;
  double c = cos(angle / 2.0), s = sin(angle / 2.0);
  return Quat(c, s * v.x(), s * v.y(), s * v.z());
}

static Vec3 quatToRodriguez(const Quat& q) {
  Vec3 vec = q.vec();
  double norm = vec.norm();
  if (norm < 1e-12) {
    return Vec3(0, 0, 0);
  }
  double angle = 2.0 * asin(norm);
  return vec * (angle / norm);
}

void Data::removeBadObservations(int64_t maxNumPts) {
  if (maxNumPts > 0) {
    cout << "num Pts: " << points.size() << " -> " << maxNumPts << endl;
    points.resize(maxNumPts);
  }

  std::vector<Obs> newObservations;
  for (auto obs : observations) {
    if (obs.ptIdx > points.size()) {
      continue;
    }
    Vec3 camPt = cameras[obs.camIdx].T_W_C * points[obs.ptIdx];
    if (camPt.squaredNorm() < 0.3 || camPt.z() > -0.1) {
      continue;
    }
    newObservations.push_back(obs);
  }
  cout << "num Obs: " << observations.size() << " -> " << newObservations.size()
       << endl;
  std::swap(observations, newObservations);
}

void Data::load(const std::string& path, bool verbose) {
  ifstream istream(path);
  if (!istream) {
    throw runtime_error("Cannot open file `" + path + "`");
  }

  if (verbose) {
    cout << "Loading file `" << path << "`" << endl;
  }

  int num_cameras, num_points, num_observations;
  istream >> num_cameras >> num_points >> num_observations;

  if (verbose) {
    cout << "N.cams: " << num_cameras << ", N.pts: " << num_points
         << ", N.obs: " << num_observations << endl;
  }

  observations.clear();
  cameras.clear();
  points.clear();

  observations.reserve(num_observations);
  cameras.reserve(num_cameras);
  points.reserve(num_points);

  if (verbose) {
    cout << "Loading observations..." << endl;
  }
  for (int i = 0; i < num_observations; i++) {
    int camIdx, ptIdx;
    double x, y;

    istream >> camIdx >> ptIdx >> x >> y;

    if (camIdx >= num_cameras || camIdx < 0) {
      throw runtime_error(
          "Error loading " + to_string(i) +
          "th observation, invalid camera index: " + to_string(camIdx));
    }
    if (ptIdx >= num_points || ptIdx < 0) {
      throw runtime_error(
          "Error loading " + to_string(i) +
          "th observation, invalid point index: " + to_string(ptIdx));
    }

    observations.emplace_back(camIdx, ptIdx, x, y);

    if (verbose && (i < 3 || i == num_observations - 1)) {
      cout << "observations[" << i << "] = " << observations[i] << endl;
    } else if (verbose && i == 3) {
      cout << "..." << endl;
    }

    if (!istream) {
      throw runtime_error("File error loading " + to_string(i) +
                          "th observation!");
    }
  }

  if (verbose) {
    cout << "Loading cameras..." << endl;
  }
  for (int i = 0; i < num_cameras; i++) {
    Vec3 rRot, tr;
    double f, k1, k2;

    istream >> rRot[0] >> rRot[1] >> rRot[2] >> tr[0] >> tr[1] >> tr[2] >> f >>
        k1 >> k2;
    cameras.emplace_back(rodriguezToQuat(rRot), tr, f, k1, k2);

    if (verbose && (i < 3 || i == num_cameras - 1)) {
      cout << "cameras[" << i << "] = " << cameras[i] << endl;
    } else if (verbose && i == 3) {
      cout << "..." << endl;
    }

    if (!istream) {
      throw runtime_error("File error loading " + to_string(i) + "th camera!");
    }
  }

  if (verbose) {
    cout << "Loading points..." << endl;
  }
  for (int i = 0; i < num_points; i++) {
    double x, y, z;

    istream >> x >> y >> z;
    points.emplace_back(x, y, z);

    if (verbose && (i < 3 || i == num_points - 1)) {
      cout << "points[" << i << "] = " << points[i] << endl;
    } else if (verbose && i == 3) {
      cout << "..." << endl;
    }

    if (!istream) {
      throw runtime_error("File error loading " + to_string(i) + "th point!");
    }
  }

  if (verbose) {
    cout << "Load successful!" << endl;
  }
}

void Data::save(const std::string& path, bool verbose) {
  ofstream ostream(path);
  if (!ostream) {
    throw runtime_error("Cannot open file `" + path + "` for writing");
  }

  if (verbose) {
    cout << "Saving to `" << path << "`..." << endl;
  }

  ostream.precision(numeric_limits<double>::max_digits10 + 2);
  ostream << scientific;

  ostream << cameras.size() << " " << points.size() << " "
          << observations.size() << endl;

  for (Obs& obs : observations) {
    ostream << obs.camIdx << " " << obs.ptIdx << " " << obs.imgPos.x() << " "
            << obs.imgPos.y() << endl;
  }

  for (Cam& cam : cameras) {
    Vec3 rRot = quatToRodriguez(cam.T_W_C.so3().unit_quaternion());
    Vec3 tr = cam.T_W_C.translation();
    for (int i = 0; i < 3; i++) {
      ostream << rRot[i] << endl;
    }
    for (int i = 0; i < 3; i++) {
      ostream << tr[i] << endl;
    }
    ostream << cam.f_k1_k2[0] << endl;
    ostream << cam.f_k1_k2[1] << endl;
    ostream << cam.f_k1_k2[2] << endl;
  }

  for (Vec3& pt : points) {
    for (int i = 0; i < 3; i++) {
      ostream << pt[i] << endl;
    }
  }

  if (verbose) {
    cout << "Save successful!" << endl;
  }
}

void Data::compute_fill_stats() const {
  map<int, vector<int>> ptToCams;

  for (const Obs& obs : observations) {
    ptToCams[obs.ptIdx].push_back(obs.camIdx);
  }

  set<pair<int, int>> camPairs;
  for (const auto& kv : ptToCams) {
    const vector<int>& cams = kv.second;

    for (size_t i = 0; i < cams.size(); i++) {
      for (size_t j = 0; j < i; j++) {
        camPairs.insert(make_pair(i, j));
      }
    }
  }

  size_t cpairs = camPairs.size();
  size_t ncams = cameras.size();
  cout << "connected_pairs: " << cpairs << " / " << (ncams * (ncams - 1) / 2)
       << " (n.cams=" << ncams << ")" << endl;
  cout << "S matrix filling: " << double(cpairs * 2 + ncams) / (ncams * ncams)
       << endl;
}

}  // end namespace ba_at_large
