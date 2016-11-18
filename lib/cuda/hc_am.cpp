#include "hc_am.hpp"
#include <cuda.h>

#ifndef KALMAR_DEBUG
#define KALMAR_DEBUG (0)
#endif

#define DB_TRACKER 0

#if DB_TRACKER 
#define mprintf( ...) {\
        fprintf (stderr, __VA_ARGS__);\
        };
#else
#define mprintf( ...) 
#endif

#define CheckCudaError(rt) { __checkCuda((rt), __FILE__, __LINE__); }
static void __checkCuda(CUresult err, const char* file, const int line) {
  if (err != CUDA_SUCCESS) {
    char const * error_name;
    cuGetErrorName (err, &error_name);
    printf("Cuda Error: %s, Line: %d, File: %s\n", error_name, line, file);
    exit(1);
  }
}

//=========================================================================================================
// Pointer Tracker Structures:
//=========================================================================================================
#include <map>
#include <iostream>

namespace hc {
AmPointerInfo & AmPointerInfo::operator= (const AmPointerInfo &other) 
{
  _hostPointer = other._hostPointer;
  _devicePointer = other._devicePointer;
  _sizeBytes = other._sizeBytes;
  _acc = other._acc;
  _isInDeviceMem = other._isInDeviceMem;
  _isAmManaged = other._isAmManaged;
  _appId = other._appId;
  _appAllocationFlags = other._appAllocationFlags;
  _queue = other._queue;

  return *this;
}
}

struct AmMemoryRange {
  const void * _basePointer;
  const void * _endPointer;
  AmMemoryRange(const void *basePointer, size_t sizeBytes) :
        _basePointer(basePointer), _endPointer((const unsigned char*)basePointer + sizeBytes - 1) {};
};

// Functor to compare ranges:
struct AmMemoryRangeCompare {
  // Return true is LHS range is less than RHS - used to order the 
  bool operator()(const AmMemoryRange &lhs, const AmMemoryRange &rhs) const
  {
    return lhs._endPointer < rhs._basePointer;
  }
};

std::ostream &operator<<(std::ostream &os, const hc::AmPointerInfo &ap)
{
  os << "hostPointer:" << ap._hostPointer << " devicePointer:"<< ap._devicePointer << " sizeBytes:" << ap._sizeBytes
     << " isInDeviceMem:" << ap._isInDeviceMem  << " isAmManaged:" << ap._isAmManaged 
     << " appId:" << ap._appId << " appAllocFlags:" << ap._appAllocationFlags;
  return os;
}

//-------------------------------------------------------------------------------------------------
// This structure tracks information for each pointer.
// Uses memory-range-based lookups - so pointers that exist anywhere in the range of hostPtr + size 
// will find the associated AmPointerInfo.
// The insertions and lookups use a self-balancing binary tree and should support O(logN) lookup speed.
// The structure is thread-safe - writers obtain a mutex before modifying the tree.  Multiple simulatenous readers are supported.
class AmPointerTracker {
typedef std::map<AmMemoryRange, hc::AmPointerInfo, AmMemoryRangeCompare> MapTrackerType;
public:

  void insert(void *pointer, const hc::AmPointerInfo &p);
  int remove(void *pointer);

  MapTrackerType::iterator find(const void *hostPtr) ;
    
  MapTrackerType::iterator readerLockBegin() { _mutex.lock(); return _tracker.begin(); } ;
  MapTrackerType::iterator end() { return _tracker.end(); } ;
  void readerUnlock() { _mutex.unlock(); };

  size_t reset (const hc::accelerator &acc);
  //void update_peers (const hc::accelerator &acc, int peerCnt, hsa_agent_t *peerAgents) ;

private:
    MapTrackerType  _tracker;
    std::mutex      _mutex;
    //std::shared_timed_mutex _mut;
};

void AmPointerTracker::insert (void *pointer, const hc::AmPointerInfo &p)
{
  std::lock_guard<std::mutex> l (_mutex);

  mprintf ("insert: %p + %zu\n", pointer, p._sizeBytes);
  _tracker.insert(std::make_pair(AmMemoryRange(pointer, p._sizeBytes), p));
}


// Return 1 if removed or 0 if not found.
int AmPointerTracker::remove (void *pointer)
{
  std::lock_guard<std::mutex> l (_mutex);
  mprintf ("remove: %p\n", pointer);
  return _tracker.erase(AmMemoryRange(pointer,1));
}

AmPointerTracker::MapTrackerType::iterator  AmPointerTracker::find (const void *pointer)
{
  std::lock_guard<std::mutex> l (_mutex);
  auto iter = _tracker.find(AmMemoryRange(pointer,1));
  mprintf ("find: %p\n", pointer);
  return iter;
}

//---
// Remove all tracked locations, and free the associated memory (if the range was originally allocated by AM).
// Returns count of ranges removed.
size_t AmPointerTracker::reset (const hc::accelerator &acc) 
{
  std::lock_guard<std::mutex> l (_mutex);
  mprintf ("reset: \n");

  size_t count = 0;
  // relies on C++11 (erase returns iterator)
  for (auto iter = _tracker.begin() ; iter != _tracker.end(); ) {
    if (iter->second._acc == acc) {
      if (iter->second._isAmManaged) {
        auto dm = reinterpret_cast<CUdeviceptr>(const_cast<void*> (iter->first._basePointer));
        CheckCudaError(cuMemFree(dm));
      }
      count++;
      iter = _tracker.erase(iter);

    } else {
      iter++;
    }
  }

  return count;
}

#if 0
//---
// Remove all tracked locations, and free the associated memory (if the range was originally allocated by AM).
// Returns count of ranges removed.
void AmPointerTracker::update_peers (const hc::accelerator &acc, int peerCnt, hsa_agent_t *peerAgents) 
{
  std::lock_guard<std::mutex> l (_mutex);

  // relies on C++11 (erase returns iterator)
  for (auto iter = _tracker.begin() ; iter != _tracker.end(); ) {
    if (iter->second._acc == acc) {
      if (iter->second._isInDeviceMem) {
        printf ("update peers\n");
        hsa_amd_agents_allow_access(peerCnt, peerAgents, NULL, const_cast<void*> (iter->first._basePointer));
      }
    } 
    iter++;
  }
}
#endif

//=========================================================================================================
// Global var defs:
//=========================================================================================================
AmPointerTracker g_amPointerTracker;  // Track all am pointer allocations.

//=========================================================================================================
// API Definitions.
//=========================================================================================================

namespace hc {
static Kalmar::KalmarQueue* sg_queue;

// Allocate accelerator memory, return NULL if memory could not be allocated:
auto_voidp am_alloc(size_t sizeBytes, hc::accelerator &acc, unsigned flags)
{
  void* ptr = nullptr;
  if (sizeBytes != 0 ) {
    if (acc.get_device_path().find(L"CUDA gpu") != std::wstring::npos) {
      auto kq = Kalmar::getContext()->getDevice(acc.get_device_path())->get_default_queue();
      Kalmar::KalmarQueue* queue = kq.get();
      sg_queue = queue;
      queue->setCurrent();
      if (flags & amHostPinned) {
        CheckCudaError(cuMemAllocHost(&ptr, sizeBytes));
        CUdeviceptr pdptr;
        CheckCudaError(cuMemHostGetDevicePointer(&pdptr, ptr, 0/*Must be 0*/));

        g_amPointerTracker.insert(ptr,
          hc::AmPointerInfo(ptr/*hostPointer*/, reinterpret_cast<void*>(pdptr) /*devicePointer*/, sizeBytes, acc, false/*isDevice*/, true /*isAMManaged*/, queue));
      } else {
        CUdeviceptr dm;
        CheckCudaError(cuMemAlloc(&dm, sizeBytes));
        ptr = reinterpret_cast<void*>(dm);

        // track device pointer
        g_amPointerTracker.insert(ptr,
          hc::AmPointerInfo(NULL/*hostPointer*/, ptr /*devicePointer*/, sizeBytes, acc, true/*isDevice*/, true /*isAMManaged*/, queue));
     }
    }
  }
  return ptr;
}

am_status_t am_free(void* ptr)
{
  am_status_t status = AM_SUCCESS;
  if (ptr != NULL) {
    unsigned int flags;
    if (cuMemHostGetFlags(&flags,ptr) == CUDA_SUCCESS) {
      CheckCudaError(cuMemFreeHost(ptr));
    } else {
      CUdeviceptr base;
      size_t size;
      auto dm = reinterpret_cast<CUdeviceptr>(ptr);
      CUresult result = cuMemGetAddressRange(&base, &size, dm);
      if (result == CUDA_SUCCESS) {
        CheckCudaError(cuMemFree(dm));
      } else {
        // Occasinally input a host raw pointer?
#if KALMAR_DEBUG
        std::cerr << "try to free a host raw pointer:" << dm << " in am_free\n";
#endif
      }
    }

    int numRemoved = g_amPointerTracker.remove(ptr) ;
    if (numRemoved == 0) {
      status = AM_ERROR_MISC;
    }

  }
  return status;
}

am_status_t am_copy(void*  dst, const void*  src, size_t sizeBytes)
{
  CUdeviceptr base;
  size_t size;
  auto d = reinterpret_cast<CUdeviceptr>(dst);
  auto s = reinterpret_cast<CUdeviceptr>(src);
  bool d_host = false;
  bool s_host = false;
  CUresult result = cuMemGetAddressRange(&base, &size, d);
  if (result != CUDA_SUCCESS) { d_host = true; }

  result = cuMemGetAddressRange(&base, &size, s);
  if (result != CUDA_SUCCESS) { s_host = true; }

  // TODO: using default stream block any asynchronization on the same device
  unsigned int flags;

  if (cuMemHostGetFlags(&flags, dst) == CUDA_SUCCESS) { d_host = true; }
  if (cuMemHostGetFlags(&flags, const_cast<void*>(src)) == CUDA_SUCCESS) { s_host = true; }

  // TODO: have multi-threading issue. Need set current context
  if (d_host && !s_host) {
    auto it = g_amPointerTracker.find(dst);
    if (it != g_amPointerTracker.end())
      it->second._queue->setCurrent();

    CUstream st = reinterpret_cast<CUstream>(sg_queue->getStream());
    CheckCudaError(cuMemcpyDtoHAsync(dst, s, sizeBytes, st));
    CheckCudaError(cuStreamSynchronize(st));
  } else if (s_host && !d_host) {
    auto it = g_amPointerTracker.find(src);
    if (it != g_amPointerTracker.end())
      it->second._queue->setCurrent();

    CUstream st = reinterpret_cast<CUstream>(sg_queue->getStream());
    CheckCudaError(cuMemcpyHtoDAsync(d, src, sizeBytes, st));
    CheckCudaError(cuStreamSynchronize(st));
  } else if (!s_host && !d_host) {
    sg_queue->setCurrent();
    CUstream st = reinterpret_cast<CUstream>(sg_queue->getStream());
    CheckCudaError(cuMemcpyDtoDAsync(d, s, sizeBytes, st));
    CheckCudaError(cuStreamSynchronize(st));
  } else {
    std::memmove(dst, src, sizeBytes);
  }

  return AM_SUCCESS;
}

am_status_t am_memtracker_getinfo(hc::AmPointerInfo *info, const void *ptr)
{
  auto infoI = g_amPointerTracker.find(ptr);
  if (infoI != g_amPointerTracker.end()) {
    *info = infoI->second;
    return AM_SUCCESS;
  } else {
    return AM_ERROR_MISC;
  }
}

am_status_t am_memtracker_add(void* ptr, size_t sizeBytes, hc::accelerator &acc, bool isDeviceMem)
{
  if (isDeviceMem) {
    g_amPointerTracker.insert(ptr, hc::AmPointerInfo(ptr/*hostPointer*/,  ptr /*devicePointer*/, sizeBytes, acc, true/*isDevice*/, false /*isAMManaged*/));
  } else {
    g_amPointerTracker.insert(ptr, hc::AmPointerInfo(NULL/*hostPointer*/,  ptr /*devicePointer*/, sizeBytes, acc, false/*isDevice*/, false /*isAMManaged*/));
  }

  return AM_SUCCESS;
}

am_status_t am_memtracker_update(const void* ptr, int appId, unsigned allocationFlags)
{
  auto iter = g_amPointerTracker.find(ptr);
  if (iter != g_amPointerTracker.end()) {
    iter->second._appId              = appId;
    iter->second._appAllocationFlags = allocationFlags;
    return AM_SUCCESS;
  } else {
    return AM_ERROR_MISC;
  }
}


am_status_t am_memtracker_remove(void* ptr)
{
  am_status_t status = AM_SUCCESS;

  int numRemoved = g_amPointerTracker.remove(ptr) ;
  if (numRemoved == 0) {
    status = AM_ERROR_MISC;
  }

  return status;
}

//---
void am_memtracker_print()
{
  std::ostream &os = std::cerr;

  //g_amPointerTracker.print(std::cerr);
  for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
      os << "  " << iter->first._basePointer << "..." << iter->first._endPointer << "::  ";
      os << iter->second << std::endl;
  }

  g_amPointerTracker.readerUnlock();
}


//---
void am_memtracker_sizeinfo(const hc::accelerator &acc, size_t *deviceMemSize, size_t *hostMemSize, size_t *userMemSize)
{
  *deviceMemSize = *hostMemSize = *userMemSize = 0;
  for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
    if (iter->second._acc == acc) {
      size_t sizeBytes = iter->second._sizeBytes;
      if (iter->second._isAmManaged) {
        if (iter->second._isInDeviceMem) {
          *deviceMemSize += sizeBytes;
        } else {
          *hostMemSize += sizeBytes;
        }
      } else {
        *userMemSize += sizeBytes;
      }
    }
  }

  g_amPointerTracker.readerUnlock();
}


//---
size_t am_memtracker_reset(const hc::accelerator &acc)
{
  return g_amPointerTracker.reset(acc);
}

#if 0
void am_memtracker_update_peers (const hc::accelerator &acc, int peerCnt, hsa_agent_t *peerAgents) 
{
  return g_amPointerTracker.update_peers(acc, peerCnt, peerAgents);
}

am_status_t am_map_to_peers(void* ptr, size_t num_peer, const hc::accelerator* peers) 
{
  // check input
  if(nullptr == ptr || 0 == num_peer || nullptr == peers)
    return AM_ERROR_MISC;

  hc::accelerator acc;
  AmPointerInfo info(nullptr, nullptr, 0, acc, false, false);
  auto status = am_memtracker_getinfo(&info, ptr);
  if(AM_SUCCESS != status)
    return status;

  hsa_amd_memory_pool_t* pool = nullptr;
    if(info._isInDeviceMem)
    {
        // get accelerator and pool of device memory
        acc = info._acc;
        pool = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
    }
    else
    {
        //TODO: the ptr is host pointer, it might be allocated through am_alloc, 
        // or allocated by others, but add it to the tracker.
        // right now, only support host pointer which is allocated through am_alloc.
        if(info._isAmManaged)
        {
            // here, accelerator is the device, but used to query system memory pool
            acc = info._acc;
            pool = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_system_region()); 
        }
        else
            return AM_ERROR_MISC;
    }

    const size_t max_agent = hc::accelerator::get_all().size();
    hsa_agent_t agents[max_agent];
  
    int peer_count = 0;

    for(auto i = 0; i < num_peer; i++)
    {
        // if pointer is device pointer, and the accelerator itself is included in the list, ignore it
        auto& a = peers[i];
        if(info._isInDeviceMem)
        {
            if(a == acc)
                continue;
        }

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hsa_amd_memory_pool_access_t access;
        hsa_status_t  status = hsa_amd_agent_memory_pool_get_info(*agent, *pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
        if(HSA_STATUS_SUCCESS != status)
            return AM_ERROR_MISC;

        // check access
        if(HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED == access)
            return AM_ERROR_MISC;

        bool add_agent = true;

        for(int ii = 0; ii < peer_count; ii++)
        {
            if(agent->handle == agents[ii].handle)
                add_agent = false;
        } 

        if(add_agent)
        {
             agents[peer_count] = *agent;
             peer_count++;
        }    
    }

    // allow access to the agents
    if(peer_count)
    {
        hsa_status_t status = hsa_amd_agents_allow_access(peer_count, agents, NULL, ptr);
        return status == HSA_STATUS_SUCCESS ? AM_SUCCESS : AM_ERROR_MISC;
    }
   
    return AM_SUCCESS;
}
#endif

am_status_t am_memory_host_lock(hc::accelerator &ac, void *hostPtr, size_t size, hc::accelerator *visible_ac, size_t num_visible_ac)
{
  am_status_t am_status = AM_ERROR_MISC;
#if 0
  void *devPtr;
  std::vector<hsa_agent_t> agents;
  for(int i=0;i<num_visible_ac;i++)
  {
    agents.push_back(*static_cast<hsa_agent_t*>(visible_ac[i].get_hsa_agent()));
  }
  hsa_status_t hsa_status = hsa_amd_memory_lock(hostPtr, size, &agents[0], num_visible_ac, &devPtr);
  if(hsa_status == HSA_STATUS_SUCCESS)
  {
    g_amPointerTracker.insert(hostPtr, hc::AmPointerInfo(hostPtr, devPtr, size, ac, false, false));
    am_status = AM_SUCCESS;
  }
#endif
  return am_status;
}

am_status_t am_memory_host_unlock(hc::accelerator &ac, void *hostPtr)
{
  am_status_t am_status = AM_ERROR_MISC;
#if 0
  hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, ac, 0, 0);
  am_status = am_memtracker_getinfo(&amPointerInfo, hostPtr);
  if(am_status == AM_SUCCESS)
  {
    hsa_status_t hsa_status = hsa_amd_memory_unlock(hostPtr);
  }
#endif
  return am_status;
}

} // end namespace hc
