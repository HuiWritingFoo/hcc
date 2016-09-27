####################
# CUDA headers
####################
if (HAS_CUDA EQUAL 1)
  include_directories(${CUDA_HEADER})
endif (HAS_CUDA EQUAL 1)

####################
# C++AMP runtime (Cuda implementation)
####################
if (HAS_CUDA EQUAL 1)
add_mcwamp_library_cuda(mcwamp_cuda mcwamp_cuda.cpp)
add_mcwamp_library_cuda(hc_am hc_am.cpp)
install(TARGETS mcwamp_cuda hc_am
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "CUDA available, going to build HCC CUDA Runtime")
else (HAS_CUDA EQUAL 1)
MESSAGE(STATUS "CUDA NOT available, NOT going to build HCC CUDA Runtime")
endif (HAS_CUDA EQUAL 1)
