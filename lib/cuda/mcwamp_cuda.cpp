//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Kalmar Runtime implementation (Cuda)

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <cuda.h>

#include <kalmar_runtime.h>
#include <kalmar_aligned_alloc.h>

#define KALMAR_DEBUG 0

#define KALMAR_PTX_JIT_ERRLOG_SIZE 8192
// threshold to clean up finished kernel in HSAQueue.asyncOps
// default set as 1024
#define ASYNCOPS_VECTOR_GC_SIZE (1024)

// TODO: not safe to turn off this option for it is hard to determine dependency of object with offset
#define DISABLE_ASYNC_MEMORY_WRITE_AND_COPY 1

extern "C" void PushArgImpl(void *k_, int idx, size_t sz, const void *s);
extern "C" void PushArgPtrImpl(void *k_, int idx, size_t sz, const void *s);

#define CheckCudaError(rt) { __checkCuda((rt), __FILE__, __LINE__); }
static void __checkCuda(CUresult err, const char* file, const int line) {
  if (err != CUDA_SUCCESS) {
    char const * error_name;
    cuGetErrorName (err, &error_name);
    printf("Cuda Error: %s, Line: %d, File: %s\n", error_name, line, file);
    exit(1);
  }
}

#if KALMAR_DEBUG
#define OutPrivate(var) { std::cout << #var << ": " << var << "\n"; }
#else
#define OutPrivate(var)
#endif

namespace Kalmar {

// forward declaration
class CudaDevice;
class CudaQueue;

class CudaCommonAsyncOp : public Kalmar::KalmarAsyncOp
{
private:
  std::shared_future<void>* future;
  Kalmar::CudaQueue* cudaQueue;
  CUevent hEvent;
  Kalmar::hcWaitMode waitMode;

public:
  std::shared_future<void>* getFuture() override { return future; }
 
  CudaCommonAsyncOp() : KalmarAsyncOp(Kalmar::hcCommandInvalid), future(nullptr), cudaQueue(nullptr) {
    CheckCudaError(cuEventCreate(&hEvent, CU_EVENT_DISABLE_TIMING));
  }

  ~CudaCommonAsyncOp() { dispose(); }

  void enqueueAsync(Kalmar::CudaQueue*);

  bool isReady() override {
    CUresult rt = cuEventQuery(hEvent);
    switch(rt) {
      case CUDA_SUCCESS: return true;
      case CUDA_ERROR_NOT_READY: return false;
      default: return false;
    }
  }

  void setWaitMode(hcWaitMode mode) override {
    waitMode = mode;
  }

  void dispose() {
    CheckCudaError(cuEventDestroy(this->hEvent));
    if (future != nullptr) {
      delete future;
      future = nullptr;
    }

  }
}; // End of CudaCommonAsyncOp

class CudaExecution : public Kalmar::KalmarAsyncOp
{
public:
  CudaExecution(CudaDevice* dev, CUmodule m, CUfunction f)
     : KalmarAsyncOp(Kalmar::hcCommandInvalid), device(dev), hmod(m), cf(f),
       future(nullptr), dynamicGroupSize(0), _queue(nullptr), hEvent(nullptr) {
    clearArgs();
    CheckCudaError(cuEventCreate(&hEvent, CU_EVENT_DISABLE_TIMING));
  }

  ~CudaExecution() { dispose(); }
  
  bool isReady() override {
    CUresult rt = cuEventQuery(hEvent);
    switch(rt) {
      case CUDA_SUCCESS: return true;
      case CUDA_ERROR_NOT_READY: return false;
      default: return false;
    }
  }

  void dispose() {
    clearArgs();
    std::vector<uint8_t>().swap(argVec);
    CheckCudaError(cuEventDestroy(hEvent));

    if (future != nullptr) {
      delete future;
      future = nullptr;
    }
  }

  void clearArgs() {
    argVec.clear();
    argCount = 0;
  }

  // Populate the kernel args
  void appendArg(int idx, void* arg, size_t size) {
#if KALMAR_DEBUG
  //std::cout << "arg index : " << idx << "\n";
  //std::cout << "arg size : " << size << "\n";
#endif
    int now = argVec.size();
    // align up
    int later = (now + size - 1) & (~(size -1));
    for (int i = 0; i < (later - now); ++i)
      argVec.push_back((uint8_t)0x00);

    uint8_t* p = static_cast<uint8_t*>(arg);
    for (int i = 0; i < size; ++i)
      argVec.push_back(p[i]);

    argCount++;
  }
  
  void appendPtrArg(int idx, void* arg, size_t size) {
    appendArg(idx, arg, size);
  }

  // Set kernel attributes
  void setStructures(size_t nr_dim, size_t* global_size, size_t* local_size);

  // Blocking
  void launch(CudaQueue* queue);

  // Non-blocking
  void launchAsync(CudaQueue* queue);

  std::shared_future<void>* getFuture() override { return future; }

  void waitComplete(void* e);

  void setDynamicGroupSegment(size_t dynamicGroupSize) {
    this->dynamicGroupSize = dynamicGroupSize;
  }

private:
  CudaDevice* device;
  CUmodule   hmod;
  CUfunction cf;
  int launchDim;
  unsigned int gridDim[3];
  unsigned int blockDim[3];
  std::string location;
  std::vector<uint8_t> argVec;  // Kernel args
  size_t argCount;              // Kernel arg count
  std::shared_future<void>* future;
  CudaQueue* _queue;
  CUevent hEvent;

  size_t dynamicGroupSize;
}; // End of CudaExecution

class CudaQueue: public KalmarQueue
{
public:
  CudaQueue(KalmarDevice* pDev, CUcontext context, execute_order order)
        : KalmarQueue(pDev, queuing_mode_automatic, order), hStream(nullptr),
        deviceCtx(context), asyncOps(),
        bufferKernelMap(), kernelBufferMap() {
    CheckCudaError(cuCtxGetStreamPriorityRange(&leastPri, &greatestPri));
    isPriSupport = (leastPri || greatestPri);
    OutPrivate(leastPri);
    OutPrivate(greatestPri);
    if (isPriSupport) {
      CheckCudaError(cuStreamCreateWithPriority(&hStream, CU_STREAM_NON_BLOCKING, greatestPri));
    } else {
      CheckCudaError(cuStreamCreate(&hStream, CU_STREAM_NON_BLOCKING));
    }
  }

  void setCurrent() override { CheckCudaError(cuCtxSetCurrent(deviceCtx)); }

  void flush() override {
  }

  void wait(hcWaitMode mode = hcWaitModeBlocked) override {
    // wait on all previous async operations to complete
    for (int i = 0; i < asyncOps.size(); ++i) {
      if (asyncOps[i] != nullptr) {
        auto asyncOp = asyncOps[i];
        // wait on valid futures only
        std::shared_future<void>* future = asyncOp->getFuture();
        if (future->valid()) {
          future->wait();
        }
      }
    }
    // clear async operations table
    asyncOps.clear();
  }

  // Push array/array_view device pointer to kernel args
  void Push(void *k_, int idx, void* dm, bool modify) override {
    PushArgImpl(k_, idx, sizeof(void*), &dm);
        /// store const informantion for each opencl memory object
        /// after kernel launches, const data don't need to wait for kernel finish
    if (modify) {
      kernelBufferMap[k_].push_back(dm);
    }
  }

  void write(void* dm, const void *src, size_t count, size_t offset, bool blocking) override  {
    waitForDependentAsyncOps(dm);
    if (dm == src) return;
#if KALMAR_DEBUG
    std::cout << "Write (to device,offset,host,bytes): " << std::hex << dm
              << ", " << std::dec << offset
              << ", " << std::hex << src
              << ", " << std::dec << count << " bytes\n";
#endif
#if DISABLE_ASYNC_MEMORY_WRITE_AND_COPY
    blocking = true;
#endif

    CheckCudaError(cuCtxSetCurrent(deviceCtx));
    if (getDev()->is_unified()) {
      std::memmove((char*)dm + offset, src, count);
    } else {
      auto ptr = reinterpret_cast<CUdeviceptr>(dm);
      if (blocking) {
        CheckCudaError(cuMemcpyHtoDAsync(ptr + offset, src, count, hStream));
        CheckCudaError(cuStreamSynchronize(hStream));
      } else {
        std::shared_ptr<CudaCommonAsyncOp> op = std::make_shared<CudaCommonAsyncOp>();
        CheckCudaError(cuMemcpyHtoDAsync(ptr + offset, src, count, hStream));
        op->enqueueAsync(this);
        asyncOps.push_back(op);
        // Update dm (dm+offset?) dependency
        bufferKernelMap[dm].clear();
        bufferKernelMap[dm].push_back(op);
      }
    }
  }

  // Read from array/array_view's device pointer to its host pointer
  // or from raw pointer
  void read(void* dm, void* dst, size_t count, size_t offset) override {
    waitForDependentAsyncOps(dm);
    if (dm == dst) return;
#if KALMAR_DEBUG
    std::cout << "Read (from device,offset,host,bytes): " << std::hex << dm
              << ", " << std::dec << offset
              << ", " << std::hex << dst
              << ", " << std::dec << count << " bytes\n";
#endif
    CheckCudaError(cuCtxSetCurrent(deviceCtx));
    if (getDev()->is_unified()) {
      std::memmove(dst, (char*)dm + offset, count);
    } else {
      CheckCudaError(cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(dm) + offset, count, hStream));
      CheckCudaError(cuStreamSynchronize(hStream));
    }
  }

// TODO: peer to peer copy
  void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) override {
    waitForDependentAsyncOps(dst);
    waitForDependentAsyncOps(src);

    if (src == dst) return;
#if KALMAR_DEBUG
    std::cout << "Copy (src,src_offset,dst,dst_offset,bytes): " << std::hex << src
              << ", " << std::dec << src_offset
              << ", " << std::hex << dst
              << ", " << std::dec << dst_offset
              << ", " << std::dec << count << " bytes\n";
#endif

    auto srcdm = reinterpret_cast<CUdeviceptr>(src);
    auto dstdm = reinterpret_cast<CUdeviceptr>(dst);
    CheckCudaError(cuCtxSetCurrent(deviceCtx));

#if DISABLE_ASYNC_MEMORY_WRITE_AND_COPY
    blocking = true;
#endif

    if (getDev()->is_unified()) {
      std::memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
    } else {

      if (blocking) {
        CheckCudaError(cuMemcpyDtoDAsync(dstdm + dst_offset, srcdm + src_offset, count, hStream));
        CheckCudaError(cuStreamSynchronize(hStream));
      } else {
        std::shared_ptr<CudaCommonAsyncOp> op = std::make_shared<CudaCommonAsyncOp>();
        CheckCudaError(cuMemcpyDtoDAsync(dstdm + dst_offset, srcdm + src_offset, count, hStream));
        op->enqueueAsync(this);
        // Update dependencies
        asyncOps.push_back(op);
        //bufferKernelMap[src].clear();
        bufferKernelMap[src].push_back(op);
        //bufferKernelMap[dst].clear();
        bufferKernelMap[dst].push_back(op);
      }
    }
  }

  // for hcc containers constructed without source
  void* map(void* dm, size_t count, size_t offset, bool modify) override {
#if KALMAR_DEBUG
    std::cerr << "map(" << dm << "," << count << "," << offset << "," << modify << ")\n";
#endif
    waitForDependentAsyncOps(dm);
    CheckCudaError(cuCtxSetCurrent(deviceCtx));

    if (!getDev()->is_unified()) {
      void* data = kalmar_aligned_alloc(0x1000, count);
      CheckCudaError(cuMemcpyDtoHAsync(data, reinterpret_cast<CUdeviceptr>(dm) + offset, count, hStream));
      CheckCudaError(cuStreamSynchronize(hStream));
      return data;
    } else {
      return (char*)dm + offset;
    }
  }

  void unmap(void* dm, void* addr, size_t count, size_t offset, bool modify) override {
#if KALMAR_DEBUG
    std::cerr << "unmap(" << dm << "," << addr << "," << count << "," << offset << "," << modify << ")\n";
#endif
    CheckCudaError(cuCtxSetCurrent(deviceCtx));
    if (!getDev()->is_unified()) {
      if (modify) {
        // copy data from host buffer to device buffer
        CheckCudaError(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dm) + offset, addr, count, hStream));
        CheckCudaError(cuStreamSynchronize(hStream));
      }
      // deallocate the host buffer
      kalmar_aligned_free(addr);
    } else {
      // for host memory there's nothing to be done
    }
  }

  void LaunchKernel(void *exec, size_t nr_dim, size_t *global_size, size_t *local_size) override {
    LaunchKernelWithDynamicGroupMemory(exec, nr_dim, global_size, local_size, 0);
  }

  void LaunchKernelWithDynamicGroupMemory(void *exec, size_t nr_dim, size_t *global_size, size_t *local_size, size_t dynamic_group_size) override {
    CudaExecution *thin = reinterpret_cast<CudaExecution*>(exec);
    thin->setStructures(nr_dim, global_size, local_size);
    thin->setDynamicGroupSegment(dynamic_group_size);
    thin->launch(this);
    
    // Cuda driver ensures the completion at this point    
    delete thin;
  }

  std::shared_ptr<KalmarAsyncOp> LaunchKernelAsync(void *exec, size_t nr_dim, size_t *global_size, size_t *local_size) override {
    return LaunchKernelWithDynamicGroupMemoryAsync(exec, nr_dim, global_size, local_size, 0);
  }

  // Must implement
  std::shared_ptr<KalmarAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(void *exec, size_t nr_dim, size_t *global_size, size_t *local_size, size_t dynamic_group_size) override {
    CudaExecution *thin = reinterpret_cast<CudaExecution*>(exec);
    thin->setStructures(nr_dim, global_size, local_size);
    thin->setDynamicGroupSegment(dynamic_group_size);

    // wait for previous kernel dispatches be completed
    std::for_each(std::begin(kernelBufferMap[exec]), std::end(kernelBufferMap[exec]),
            [&] (void* buffer) {
            waitForDependentAsyncOps(buffer);
           });

    thin->launchAsync(this);
    std::shared_ptr<KalmarAsyncOp> op(thin);
    asyncOps.push_back(op);
    // associate all buffers used by the kernel with the kernel dispatch instance
    std::for_each(std::begin(kernelBufferMap[exec]), std::end(kernelBufferMap[exec]),
          [&] (void* buffer) {
          bufferKernelMap[buffer].push_back(op);
         });

    // clear data in kernelBufferMap
    kernelBufferMap[exec].clear();
    
    return op;
  }

  // wait for dependent async operations to complete
    void waitForDependentAsyncOps(void* buffer) {
        auto dependentAsyncOpVector = bufferKernelMap[buffer];
        for (int i = 0; i < dependentAsyncOpVector.size(); ++i) {
          auto dependentAsyncOp = dependentAsyncOpVector[i];
          if (!dependentAsyncOp.expired()) {
            auto dependentAsyncOpPointer = dependentAsyncOp.lock();
            // wait on valid futures only
            std::shared_future<void>* future = dependentAsyncOpPointer->getFuture();
            if (future->valid()) {
              future->wait();
            }
          }
        }
        dependentAsyncOpVector.clear();
    }


  CUstream getCudaStream() { return hStream; }

  // enqueue a barrier packet
  std::shared_ptr<KalmarAsyncOp> EnqueueMarker() override {
    // create shared_ptr instance
    std::shared_ptr<CudaCommonAsyncOp> barrier = std::make_shared<CudaCommonAsyncOp>();

    // enqueue the barrier
    barrier.get()->enqueueAsync(this);

    // associate the barrier with this queue
    asyncOps.push_back(barrier);

    return barrier;
  }

  // remove finished async operation from waiting list
  void removeAsyncOp(KalmarAsyncOp* asyncOp) {
    for (int i = 0; i < asyncOps.size(); ++i) {
      if (asyncOps[i].get() == asyncOp) {
        asyncOps[i] = nullptr;
      }
    }

    // GC for finished kernels
    if (asyncOps.size() > ASYNCOPS_VECTOR_GC_SIZE) {
      asyncOps.erase(std::remove(asyncOps.begin(), asyncOps.end(), nullptr),
                     asyncOps.end());
    }
  }

  // This routine ensures CudaQueue is destroyed ahead of CudaDevice
  void dispose() override {
    if (hStream) {
      // Ensure all works in the stream is complete
      wait();
#if KALMAR_DEBUG
      std::cout << "Destroy stream: " << hStream << "  this: " << this << "\n";
#endif
      // clear bufferKernelMap
      for (auto iter = bufferKernelMap.begin(); iter != bufferKernelMap.end(); ++iter) {
        iter->second.clear();
      }
      bufferKernelMap.clear();
      // clear kernelBufferMap
      for (auto iter = kernelBufferMap.begin(); iter != kernelBufferMap.end(); ++iter) {
        iter->second.clear();
      }
      kernelBufferMap.clear();
      
      CheckCudaError(cuStreamDestroy(hStream));
      hStream = nullptr;
    }
  }

  ~CudaQueue() {
    if (hStream) dispose();
  }

private:
   CUstream hStream;
   int leastPri;
   int greatestPri;
   bool isPriSupport;
   execute_order order;
   std::vector< std::shared_ptr<KalmarAsyncOp> > asyncOps;
   std::map<void*, std::vector< std::weak_ptr<KalmarAsyncOp> > > bufferKernelMap;
   std::map<void*, std::vector<void*> > kernelBufferMap;
   CUcontext deviceCtx;  // equal to device's context for multiple threading
}; // End of CudaQueue

void CudaCommonAsyncOp::enqueueAsync(Kalmar::CudaQueue* queue) {
  this->cudaQueue = queue;
  CUstream hStream = queue->getCudaStream();
  CheckCudaError(cuEventRecord(hEvent, hStream));

  // dynamically allocate a std::shared_future<void> object
  future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
    if (hEvent == nullptr) return;
    CheckCudaError(cuStreamWaitEvent(hStream, this->hEvent, 0));
    if (this->cudaQueue != nullptr) {
      this->cudaQueue->removeAsyncOp(this);
    }
  }).share());
}

void CudaExecution::waitComplete(void* e) {
  if (e == nullptr) return;
  CUevent hEvent = reinterpret_cast<CUevent>(e);
  CheckCudaError(cuStreamWaitEvent(this->_queue->getCudaStream(), hEvent, 0));

  // unregister this async operation from HSAQueue
  if (this->_queue != nullptr) {
    this->_queue->removeAsyncOp(this);
  }
}

void CudaExecution::launch(CudaQueue* queue) {
  this->_queue = queue;
  CUstream hStream = queue->getCudaStream();

  int size = argVec.size();
  void* Extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, argVec.data(),
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &size,
    CU_LAUNCH_PARAM_END
  };
  void** extra = size?Extra:nullptr;
#if KALMAR_DEBUG
  std::cout << "gridDim: " << gridDim[0] << ", " << gridDim[1] << ", "<< gridDim[2] << "\n";
  std::cout << "blockDim: " << blockDim[0] << ", " << blockDim[1] << ", "<< blockDim[2] << "\n";
#endif

  CheckCudaError(cuLaunchKernel(cf, gridDim[0], gridDim[1], gridDim[2],
                 blockDim[0], blockDim[1], blockDim[2],
                 0/*sharedMemBytes*/, hStream, nullptr/*kernelParams*/, extra));
  CheckCudaError(cuStreamSynchronize(hStream));
}

void CudaExecution::launchAsync(CudaQueue* queue) {
  this->_queue = queue;
  CUstream hStream = queue->getCudaStream();

  int size = argVec.size();
  void* Extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, argVec.data(),
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &size,
    CU_LAUNCH_PARAM_END
  };
  void** extra = size?Extra:nullptr;
#if KALMAR_DEBUG
  //int i = 0;
  //std::cout << "int alignment: " << __alignof(i) << "\n";
  //uint64_t j = 0;
  //std::cout << "int64 aligment: " << __alignof(j) << "\n";
  //std::cout << "size: " << size << "\n";
  //std::cout << "arg count: " << argCount << "\n";
  std::cout << "gridDim: " << gridDim[0] << ", " << gridDim[1] << ", "<< gridDim[2] << "\n";
  std::cout << "blockDim: " << blockDim[0] << ", " << blockDim[1] << ", "<< blockDim[2] << "\n";
#endif

  CheckCudaError(cuLaunchKernel(cf, gridDim[0], gridDim[1], gridDim[2],
                 blockDim[0], blockDim[1], blockDim[2],
                 0/*sharedMemBytes*/, hStream, NULL/*kernelParams*/, extra));
  CheckCudaError(cuEventRecord(hEvent, hStream));

  // dynamically allocate a std::shared_future<void> object
  future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
    // wait for completion
    waitComplete(hEvent);
  }).share());

}

class CudaDevice: public KalmarDevice
{
friend class CudaExecution;
public:
  CudaDevice(CUdevice device, int ordinal)
    : KalmarDevice(access_type_none), programs(),
      queues(), queues_mutex(), device(device) {

    char name[256] = {0x0}; 
    CheckCudaError(cuDeviceGetName(name, sizeof(name)/sizeof(char), device)); 

    wchar_t path_wchar[1024] {0};
    swprintf(path_wchar, sizeof(path_wchar)/sizeof(wchar_t),
             L"CUDA gpu %u: %s", ordinal, name);
    path = std::wstring(path_wchar);

    CheckCudaError(cuDeviceGetAttribute(&devMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CheckCudaError(cuDeviceGetAttribute(&devMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    sm = devMajor*10 + devMinor;

    wchar_t description_wchar[1024] {0};
    swprintf(description_wchar, sizeof(description_wchar)/sizeof(wchar_t),
             L", Device Compute Capability:%d.%d", devMajor, devMinor);
    description = path + std::wstring(description_wchar);

#if KALMAR_DEBUG
    std::wcerr << L"Path: " << path << L"\n";
    std::wcerr << L"Description: " << description << L"\n";
#endif

    // Total memory on the device
    CheckCudaError(cuDeviceTotalMem(&deviceMemBytes, device));

    // Other main device properties
    CheckCudaError(cuDeviceGetAttribute(&threadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    OutPrivate(threadsPerBlock);
    CheckCudaError(cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
    OutPrivate(warpSize);
    CheckCudaError(cuDeviceGetAttribute(&maxBlockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
    OutPrivate(maxBlockDim[0]);
    CheckCudaError(cuDeviceGetAttribute(&maxBlockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device));
    OutPrivate(maxBlockDim[1]);
    CheckCudaError(cuDeviceGetAttribute(&maxBlockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device));
    OutPrivate(maxBlockDim[2]);
    CheckCudaError(cuDeviceGetAttribute(&maxGridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    OutPrivate(maxGridDim[0]);
    CheckCudaError(cuDeviceGetAttribute(&maxGridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device));
    OutPrivate(maxGridDim[1]);
    CheckCudaError(cuDeviceGetAttribute(&maxGridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device));
    OutPrivate(maxGridDim[2]);

    CheckCudaError(cuDeviceGetAttribute(&supportUnified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    OutPrivate(supportUnified);

    // Each device is attached to a context
    CheckCudaError(cuCtxCreate(&deviceCtx, 0, device));
    size_t stack_limit;
    CheckCudaError(cuCtxGetLimit(&stack_limit, CU_LIMIT_STACK_SIZE));
    OutPrivate(stack_limit);
  }

  // AMP conformance
  std::wstring get_path() const override { return path; }
  std::wstring get_description() const override { return description; }
  size_t get_mem() const override { return deviceMemBytes; }
  bool is_double() const override { return true; }
  bool is_lim_double() const override { return true; }
  bool is_unified() const override { return supportUnified == 1; }
  bool is_emulated() const override { return false; }
  uint32_t get_version() const override { return ((static_cast<unsigned int>(devMajor) << 16) | devMinor); }

  void BuildProgram(void* size, void* source) override {
    // JIT is used in CreateKernel
  }

  void* CreateKernel(const char* name, void* size, void* source) override {
#if KALMAR_DEBUG
    std::cout << "kernel name : " << name << "\n";
#endif

    // In case in multiple threading, grant the kernel thread a valid CUDA context
    CheckCudaError(cuCtxSetCurrent(deviceCtx));

    std::string str(name);
    CUfunction function;
    CUmodule module = programs[str];
    if (!module) {
      size_t ptx_size = (size_t)((void *)size);
      char *ptx_source = (char*)malloc(ptx_size + 1);
      memcpy(ptx_source, source, ptx_size);
      ptx_source[ptx_size] = '\0';

#if KALMAR_DEBUG
      //std::cout << ptx_source;

      char error_log[KALMAR_PTX_JIT_ERRLOG_SIZE] = {0x0};
      CUjit_option options[] = { 
        CU_JIT_ERROR_LOG_BUFFER, 
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_TARGET_FROM_CUCONTEXT,
        CU_JIT_TARGET
      };

      CUjit_target_enum target = CU_TARGET_COMPUTE_30;
      switch (this->sm) {
        case 10: target = CU_TARGET_COMPUTE_10; break;
        case 11: target = CU_TARGET_COMPUTE_11; break;
        case 12: target = CU_TARGET_COMPUTE_12; break;
        case 13: target = CU_TARGET_COMPUTE_13; break;
        case 20: target = CU_TARGET_COMPUTE_20; break;
        case 21: target = CU_TARGET_COMPUTE_21; break;
        case 30: target = CU_TARGET_COMPUTE_30; break;
        case 35: target = CU_TARGET_COMPUTE_35; break;
        case 37: target = CU_TARGET_COMPUTE_37; break;
        case 50: target = CU_TARGET_COMPUTE_50; break;
        case 52: target = CU_TARGET_COMPUTE_52; break;
        default: assert(0 && "No this target!"); break;
      } 
      void* optionValues[] = {
        (void*)error_log,
        (void*)(uintptr_t)KALMAR_PTX_JIT_ERRLOG_SIZE,
        0,
        (void*)(uintptr_t)target
       };

      CUresult err = cuModuleLoadDataEx(&module, ptx_source,
                                        sizeof(options)/sizeof(CUjit_option),
                                         options, optionValues);
      if (err != CUDA_SUCCESS) {
        free(ptx_source);
        std::cout << "Error in loading. Log: " << error_log << "\n";
        exit(1);
      }
#else
      CheckCudaError(cuModuleLoadDataEx(&module, ptx_source, 0, nullptr, nullptr));
#endif 
      free(ptx_source);
      ptx_source = nullptr;

      programs[str] = module;
    }

    CheckCudaError(cuModuleGetFunction(&function, module, name));
    CudaExecution *thin = new CudaExecution(this, module, function);
    thin->clearArgs();
  
    return thin;
  }

  bool check(size_t* local_size, size_t dim_ext) override {
    // C++ AMP specifications
    // The maximum number of tiles per dimension will be no less than 65535.
    // The maximum number of threads in a tile will be no less than 1024.
    // In 3D tiling, the maximal value of D0 will be no less than 64.
    int threads_per_tile = 1;
    for(int i = 0; local_size && i < dim_ext; ++i) {
      threads_per_tile *= local_size[i];
      // For the following cases, set local_size=NULL and let OpenCL driver arranges it instead
      //  (1) tils number exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES per dimension
      //  (2) threads in a tile exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES
      // Note that the driver can still handle unregular tile_dim, e.g. tile_dim is undivisble by 2
      // So skip this condition ((local_size[i]!=1) && (local_size[i] & 1))
      if(local_size[i] > maxBlockDim[i] || threads_per_tile > maxBlockDim[i])
        return false;
    }

    return true;
  }

  void* create(size_t count, struct rw_info* /* not care */ ) override {
    void* ptr = nullptr;
    if (is_unified()) {
      CheckCudaError(cuMemAllocHost(&ptr, count));
    } else {
      CUdeviceptr dm;
      CheckCudaError(cuMemAlloc(&dm, count));
      ptr = reinterpret_cast<void*>(dm);
    }
#if KALMAR_DEBUG
     std::cout << "Create device/page-locked pointer " << std::hex << ptr
               << " (addr: " << std::hex << &ptr << ") in " << std::dec << count << " bytes\n";
#endif
    return ptr;
  }

  void release(void *dm, struct rw_info* /* not care */ ) override {
#if KALMAR_DEBUG
      std::cout << "Release device pointer: " << std::hex << dm << "\n";
#endif
    if (is_unified()) {
      CheckCudaError(cuMemFreeHost(dm));
    } else {
      auto ptr = reinterpret_cast<CUdeviceptr>(dm);
      CheckCudaError(cuMemFree(ptr));
    }
  }

  std::shared_ptr<KalmarQueue> createQueue(execute_order order = execute_in_order) override {
    std::shared_ptr<KalmarQueue> q = std::shared_ptr<KalmarQueue>(new CudaQueue(this, deviceCtx, order));
    queues_mutex.lock();
    queues.push_back(q);
    queues_mutex.unlock();
    return q;
  }

  ~CudaDevice() {
    // release all queues
    queues_mutex.lock();
    for (auto queue_iterator : queues) {
      if (!queue_iterator.expired()) {
        auto queue = queue_iterator.lock();
        queue->dispose();
      }
    }
    queues.clear();
    queues_mutex.unlock();

    for (auto& it : programs)
      CheckCudaError(cuModuleUnload(it.second));
    programs.clear();
#if KALMAR_DEBUG
    std::cout << "Destroy context: " << deviceCtx << "\n";
#endif
    CheckCudaError(cuCtxDestroy(deviceCtx));
  }

private:
    CUdevice     device;
    CUcontext    deviceCtx;
    std::wstring path;
    std::wstring description;
    size_t deviceMemBytes;   // device memory in bytes
    std::map<std::string, CUmodule> programs;
    std::mutex queues_mutex;
    std::vector< std::weak_ptr<KalmarQueue> > queues;
protected:
    // CUDA device properties
    int threadsPerBlock;
    int warpSize;  // in threads
    int maxBlockDim[3];
    int maxGridDim[3];
    int sm;
    int devMajor;
    int devMinor;
    int supportUnified;
};

template <typename T> inline void deleter(T* ptr) { delete ptr; }

class CudaContext : public KalmarContext
{
public:
  CudaContext() : KalmarContext() {
    // Must initiate CUDA driver before any call of its apis
    cuInit(0);
    int devCount = 0;
    CheckCudaError(cuDeviceGetCount(&devCount));
    for (int ordinal = 0; ordinal < devCount; ++ordinal) {
      CUdevice device;
      CheckCudaError(cuDeviceGet(&device, ordinal)); 
      auto Dev = new CudaDevice(device, ordinal);
      if (ordinal == 0 ) def = Dev;
      Devices.push_back(Dev);
    }
  }

  ~CudaContext() {
    std::for_each(std::begin(Devices), std::end(Devices), deleter<KalmarDevice>);
    Devices.clear();
    def = nullptr;
  }
};

// https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm
static unsigned int gcd(unsigned int a, unsigned int b) {
  unsigned int remainder;
  while (b != 0) {
    remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}
static unsigned int primeFactorization (unsigned int number) {
  unsigned int x_fixed = 2, cycle_size = 2, x = 2, factor = 1;
  while (factor == 1) {
    for (int count = 1; count <= cycle_size && factor <= 1; ++count) {
      x = (x * x + 1 ) % number;
      factor = gcd(x - x_fixed, number);
    }
    cycle_size *= 2;
    x_fixed = x;
  }
  return factor;
}

  // Set kernel attributes
void CudaExecution::setStructures(size_t nr_dim, size_t* global_size, size_t* local_size) {
    assert(nr_dim > 0 && nr_dim < 4);
    launchDim = nr_dim;

    // check if local_size is not valid as input, nullptr otherwise
    //if(device->check(local_size, nr_dim))
    //  local_size = NULL;

    bool dynamic = false;
    size_t local[3] = {1,1,1};
    if (!local_size) {
      local_size = local;
      dynamic = true;
    }

    // ensure the inputs are within hardware limitation
    for (int i = 0; i < nr_dim; ++i) {
      if(global_size[i] > device->maxBlockDim[i] * device->maxGridDim[i]) {
        std::cout << "Exit. level: " << i << ", extent is too large: " << global_size[i] << "\n";
        exit(1);
      }
    }
    gridDim[0] = 1; gridDim[1] = 1; gridDim[2] = 1;
    blockDim[0] = 1; blockDim[1] = 1; blockDim[2] = 1;

    int threads = 1;
    const int maxThreadsPerBlock = device->threadsPerBlock;
    for (int i = 0; i < nr_dim; ++i) {
      blockDim[i] = std::min((int)global_size[i], device->maxBlockDim[i]);
      blockDim[i] = std::min((int)local_size[i], (int)blockDim[i]);
      threads *= blockDim[i];
      // if no local_size specified, blockDim[i] = 1 at this point$
      gridDim[i] = (global_size[i] + blockDim[i] - 1) / blockDim[i];
    }
    if (!dynamic && threads > maxThreadsPerBlock) {
      std::cout << "Exit. User given threads in a block are too many: " << threads << "\n";
      exit(1);
    }


    // if no user specified local_size, let runtime determines grid/block
    if (dynamic) {
      threads = 1;

      int level = 0;
      do {
        threads *= blockDim[level];

        size_t grid = gridDim[level];

        // gridDim needs meet hardware limitation
        while (grid > device->maxGridDim[level]) {
          size_t prime = primeFactorization(grid);
          if (prime != grid) {
            gridDim[level] /= prime;
            blockDim[level] *= prime;
            threads *= prime;
          } else {
            blockDim[level] <<= 1;
            threads *= 2;
            gridDim[level] = (grid + 1)/2;
          }
          grid = gridDim[level];
        }
        level++;
      } while (level < nr_dim);

      if (threads > maxThreadsPerBlock) {
        std::cout << "Exit. Dynamically distributed threads in a block are too many: " << threads << "\n";
        exit(1);
      }

    }

#if KALMAR_DEBUG
    for (int i = 0; i < nr_dim; ++i) {
      std::cout << "global_size: " << global_size[i] << "\n";
      std::cout << "local_size: " << local_size[i] << "\n";
    }
#endif
  }

static CudaContext ctx;

} // namespace Kalmar

extern "C" void *GetContextImpl() {
  return &Kalmar::ctx;
}

// For array & array_view containers
extern "C" void PushArgImpl(void *k_, int idx, size_t sz, const void *s) {
  Kalmar::CudaExecution *thin = reinterpret_cast<Kalmar::CudaExecution*>(k_);
  thin->appendArg(idx, const_cast<void*>(s), sz);
}
// For raw pointers. 
extern "C" void PushArgPtrImpl(void *k_, int idx, size_t sz, const void *s) {
  Kalmar::CudaExecution *thin = reinterpret_cast<Kalmar::CudaExecution*>(k_);
  thin->appendPtrArg(idx, const_cast<void*>(s), sz);
}
