#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <cuda_runtime.h> 
using namespace std;
using namespace chrono;


inline void checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        // 如果任何CUDA API调用失败，打印错误信息并退出
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main()
{
   
    int deviceCount;
    checkCudaErr(cudaGetDeviceCount(&deviceCount), "Failed to get device count");
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return 1;
    }
    // 选择第一个可用的GPU (设备ID为0)
    checkCudaErr(cudaSetDevice(0), "Failed to set device 0");
    cout << "CUDA device initialized successfully." << endl;
    
   

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    
    auto start_train = system_clock::now();
    q.m.train("/home/s2313715/.vscode-server/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    
    q.init();
    cout << "here" << endl;
    
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    
    while (!q.priority.empty())
    {
        
        q.PopNext();
        
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }
        
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                MD5Hash(pw, state);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    
    checkCudaErr(cudaDeviceReset(), "Failed to reset device");
    

    return 0;
}
