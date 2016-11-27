
// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <hc.hpp>
#include <hc_am.hpp>

#include <iostream>
#include <random>

#include <time.h>

#define DISPATCH_COUNT (1000)

#define TEST_DEBUG (0)

// A test which measures time spent in dispatching empty kernels with grid size 2048
bool test1() {
  bool ret = true;

  long time_spent = 0, time_spent_once;
  struct timespec begin;
  struct timespec end;
  for (int i = 0; i < DISPATCH_COUNT; ++i) {
    // launch kernel
    hc::extent<1> e(1024);
    clock_gettime(CLOCK_REALTIME, &begin);
    hc::completion_future fut = hc::parallel_for_each(
      e,
      [=](hc::index<1> idx) restrict(amp) {
    });
    clock_gettime(CLOCK_REALTIME, &end);
    time_spent_once = ((end.tv_sec - begin.tv_sec) * 1000 * 1000) + ((end.tv_nsec - begin.tv_nsec) / 1000);
    time_spent += time_spent_once;

    fut.wait();
    ret &= (fut.is_ready() == true);
  }

  std::cout << "Enqueued " << DISPATCH_COUNT << " empty kernels\n";
  std::cout << "Average time per kernel: " << ((double)time_spent / DISPATCH_COUNT) << "us\n";

  return ret;
}

// A test which measures time spent in dispatching vector addition kernels with grid size 2048
bool test2() {
  bool ret = true;

  // define inputs and output
  const int vecSize = 2048;

  hc::accelerator acc = hc::accelerator();
  int *p_a = (int*)hc::am_alloc(vecSize*sizeof(int),acc,1);
  int *p_b = (int*)hc::am_alloc(vecSize*sizeof(int),acc,1);
  int *p_c = (int*)hc::am_alloc(vecSize*sizeof(int),acc,1);

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    p_a[i] = int_dist(rd);
    p_b[i] = int_dist(rd);
  }

  long time_spent = 0, time_spent_once;
  struct timespec begin;
  struct timespec end;
  for (int i = 0; i < DISPATCH_COUNT; ++i) {
    // launch kernel
    hc::extent<1> e(vecSize);
    clock_gettime(CLOCK_REALTIME, &begin);
    hc::completion_future fut = hc::parallel_for_each(
      e,
      [=](hc::index<1> idx) restrict(amp) {
        p_c[idx[0]] = p_a[idx[0]] + p_b[idx[0]];
  
    });
    clock_gettime(CLOCK_REALTIME, &end);
    time_spent_once = ((end.tv_sec - begin.tv_sec) * 1000 * 1000) + ((end.tv_nsec - begin.tv_nsec) / 1000);
    time_spent += time_spent_once;

    fut.wait();
    ret &= (fut.is_ready() == true);
  }

  std::cout << "Enqueued " << DISPATCH_COUNT << " vector addition kernels\n";
  std::cout << "Average time per kernel: " << ((double)time_spent / DISPATCH_COUNT) << "us\n";

  hc::am_free(p_a);
  hc::am_free(p_b);
  hc::am_free(p_c);

  return ret;
}

void init() {
    // launch an empty kernel to initialize everything
    hc::extent<1> e(1024);
    hc::completion_future fut = hc::parallel_for_each(
      e,
      [=](hc::index<1> idx) restrict(amp) {
    });
    fut.wait();
}

int main() {
  bool ret = true;

  init();
  ret &= test1();
  ret &= test2();

  return !(ret == true);
}


