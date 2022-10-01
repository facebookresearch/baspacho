/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"
#include <array>
#include <cmath>
#include <iomanip>

using namespace std;

string percentageString(double rat, int precision) {
  stringstream ss;
  ss << fixed << setprecision(precision) << (rat * 100) << "%";
  return ss.str();
}

string humanReadableSize(size_t nbytes) {
  stringstream ss;
  static const array<string, 5> suffixes = {"", "K", "M", "G", "T"};
  double num = nbytes;
  unsigned int i = 0;
  while ((num >= 256) && (i < suffixes.size() - 1)) {
    i += 1;
    num /= 1024.0;
  }
  ss << fixed << setprecision(i == 0 ? 0 : (num < 1 ? 2 : 1)) << num << suffixes[i] << "b";
  return ss.str();
}

string microsecondsString(size_t microseconds, int precision) {
  ostringstream os;
  if (microseconds < 1000) {
    os << microseconds << "\u03bcs";
  } else if (microseconds < 100000) {
    os << fixed << setprecision(precision) << (microseconds * 0.001) << "ms";
  } else {
    if (microseconds >= 3600000000L) {
      os << microseconds / 3600000000 << "h";
      microseconds %= 3600000000;
      if (microseconds >= 60000000) {
        os << microseconds / 60000000 << "m";
      }
    } else if (microseconds >= 60000000) {
      os << microseconds / 60000000 << "m";
      microseconds %= 60000000;
      size_t seconds = (size_t)round(microseconds * 0.000001);
      if (seconds > 0) {
        os << seconds << "s";
      }
    } else {
      os << fixed << setprecision(precision) << (microseconds * 0.000001) << "s";
    }
  }
  return os.str();
}
