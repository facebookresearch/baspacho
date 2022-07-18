#pragma once

#include <cxxabi.h>
#include <chrono>
#include <string>

// template introspection util - returns a prettified type name
template <typename T>
std::string prettyTypeName() {
  char* c_str = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  std::string retv(c_str);
  free(c_str);
  return retv;
}

// prints data size in human readable form, 1.2Mb, etc...
std::string humanReadableSize(size_t nbytes);

// convert microseconds into a human readable string
std::string microsecondsString(size_t ms, int precision = 2);

// percentage string, with given precision
std::string percentageString(double rat, int precision = 1);

// convert std::chrono::duration into a human readable string
template <typename Rep, typename Period>
std::string timeString(const std::chrono::duration<Rep, Period>& duration, int precision = 2) {
  return microsecondsString(std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
                            precision);
}