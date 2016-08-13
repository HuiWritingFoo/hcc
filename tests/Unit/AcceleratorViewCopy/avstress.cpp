// RUN: %hc %s -o %t.out -lhc_am && %t.out
#include<stdlib.h>
#include<iostream>

#include<hc.hpp>
#include<hc_am.hpp>

#include"common.h"

#define N 1024*1024

const size_t size = sizeof(float) * N;

#include "common2.h"


int main(){
    std::vector<hc::accelerator> accs = hc::accelerator::get_all();
    hc::accelerator gpu_acc;
    for(auto& it:accs){
        if(!it.get_is_emulated()){
            gpu_acc = it;
            break;
        }
    }

    Init(gpu_acc);
    hc::accelerator_view av = gpu_acc.get_default_view();


    //unsigned testsToRun = 0xFF;
    unsigned testsToRun = 0x18;
    //int testIters = SHRT_MAX;
    int testIters = 2000;


    if (testsToRun & 0x1) {
        for(uint32_t i=0;i<testIters;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test1 %5d/%5d\n", i, testIters);
            }
            Test1(av);
        }
    }

    if (testsToRun & 0x2) {
        for(uint32_t i=0;i<testIters;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test2 %5d/%5d\n", i, testIters);
            }
            Test2(av);
        }
    }

    if (testsToRun & 0x4) {
        for(uint32_t i=0;i<testIters;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test3 %5d/%5d\n", i, testIters);
            }
            Test3(av);
        }
    }

    if (testsToRun & 0x8) {
        for(uint32_t i=0;i<testIters;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test4 %5d/%5d\n", i, testIters);
            }
            Test4(av);
        }
    }

    if (testsToRun & 0x10) {
        for(uint32_t i=0;i<testIters;i++){
            if ((i%1000 == 0)) {
                printf ("info: running Test5 %5d/%5d\n", i, testIters);
            }
            Test5(av);
        }
    }


    
}
