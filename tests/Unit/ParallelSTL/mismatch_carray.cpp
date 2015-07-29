// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

// test mismatch (non-predicated version)
template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp table2[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });
  n = 0;
  std::generate(std::begin(table2), std::end(table2), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, SIZE);

  // randomly pick one element from the input array and make them disagree on the inputs
  int indexRandom = dis(gen);
  table2[indexRandom] = 0;

  // launch kernel with parallel STL mismatch
  using namespace std::experimental::parallel;
  auto result = mismatch(par, std::begin(table), std::end(table), std::begin(table2));

  // verify data
  bool ret = true;
  if (std::distance(std::begin(table), result.first) != indexRandom) {
    ret = false;
  }
  if (std::distance(std::begin(table2), result.second) != indexRandom) {
    ret = false;
  }

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j] << "," << table2[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

// test mismatch (predicated version)
template<typename _Tp, size_t SIZE>
bool test2() {

  _Tp table[SIZE] { 0 };
  _Tp table2[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  // values in table are 0..SIZE-1
  std::generate(std::begin(table), std::end(table), [&] { return n++; });
  // values in table2 are 1..SIZE
  n = 1;
  std::generate(std::begin(table2), std::end(table2), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, SIZE);

  // randomly pick one element from the input array and make them disagree on the inputs
  int indexRandom = dis(gen);
  table2[indexRandom] = table[indexRandom];

  // launch kernel with parallel STL mismatch
  using namespace std::experimental::parallel;
  // use a custom predicate
  auto result = mismatch(par, std::begin(table), std::end(table), std::begin(table2),
                         [](const _Tp& a, const _Tp& b) { return ((a + 1) == b); });

  // verify data
  bool ret = true;
  if (std::distance(std::begin(table), result.first) != indexRandom) {
    ret = false;
  }
  if (std::distance(std::begin(table2), result.second) != indexRandom) {
    ret = false;
  }

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j] << "," << table2[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

int main() {
  bool ret = true;

  // non-predicated mismatch
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  // predicated mismatch
  ret &= test2<int, TEST_SIZE>();
  ret &= test2<unsigned, TEST_SIZE>();
  ret &= test2<float, TEST_SIZE>();
  ret &= test2<double, TEST_SIZE>();

  return !(ret == true);
}

