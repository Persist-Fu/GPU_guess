#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h> // MPI: 包含 MPI 头文件

using namespace std;
using namespace chrono;

// MPI 编译指令示例:
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main -O2 
// mpirun -np 4 ./main

int main(int argc, char* argv[]) // MPI: main 函数签名
{
    // MPI: 初始化
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    // --- 1. 模型训练 (所有进程都执行) ---
    if (rank == 0) { // 只有主进程计时
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    } else {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
    MPI_Barrier(MPI_COMM_WORLD); // 同步点

    // --- 2. 加载测试数据 (所有进程都加载一份) ---
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw) {
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000) {
            break;
        }
    }
    
    // MPI: 每个进程维护自己的本地破解数
    int local_cracked = 0;

    // --- 3. 队列初始化 (只有主进程执行) ---
    if (rank == 0) {
        q.init();
    }

    // MPI: 主进程管理全局计数器和计时器
    int curr_num = 0;
    int history = 0;
    // 修复方法
    auto start = (rank == 0) ? system_clock::now() : system_clock::time_point{};    bool continue_looping = true;
    
    // --- 4. 并行主循环 ---
    do
    {
        // 所有进程都调用 PopNext
        q.PopNext();

        // MPI: -------------------------------------------------------------------
        // 所有计时、打印、哈希、破解检查、终止判断的逻辑，都只在主进程(rank 0)中执行
        if (rank == 0)
        {
            q.total_guesses = q.guesses.size(); // 你的原始逻辑
            if (q.total_guesses - curr_num >= 100000)
            {
                curr_num = q.total_guesses; // 你的原始逻辑

                // 检查终止条件
                int generate_n = 10000000;
                if (history + q.total_guesses > generate_n)
                {
                    continue_looping = false; // 标记循环结束
                }
            }
            
            // 检查哈希触发条件
            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();

                // 你的原始并行哈希与破解检查逻辑
                bit32 batch_states[4][4];
                size_t total = q.guesses.size();
                for (size_t i = 0; i < total; i += 4) {
                    std::string batch[4];
                    size_t remain = total - i;
                    size_t batch_size = (remain >= 4) ? 4 : remain;

                    for (size_t j = 0; j < batch_size; ++j) {
                        if (test_set.find(q.guesses[i + j]) != test_set.end()) {
                            // 主进程更新自己的本地破解数
                            local_cracked += 1;
                        }
                        batch[j] = q.guesses[i + j];
                    }
                    for (size_t j = batch_size; j < 4; ++j) {
                        batch[j] = "";
                    }
                    MD5Hash(batch, batch_states);
                }
                
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }

            // 检查队列是否为空作为另一个终止条件
            if (q.priority.empty()) {
                continue_looping = false;
            }
        }
        // MPI: -------------------------------------------------------------------

        // MPI: 主进程将循环决定广播给所有进程
        MPI_Bcast(&continue_looping, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    } while (continue_looping);

    // --- 5. 结果收集与打印 ---
    
    // MPI: 使用 MPI_Reduce 将所有进程的 local_cracked 加总到主进程的 cracked 变量
    int total_cracked = 0;
    MPI_Reduce(&local_cracked, &total_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // MPI: 只有主进程打印最终结果
    if (rank == 0) {
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        
        cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
        cout << "Hash time:" << time_hash << "seconds" << endl;
        cout << "Train time:" << time_train << "seconds" << endl;
        cout << "Cracked:" << total_cracked << endl; // 打印全局总破解数
    }
    
    // MPI: 最终化
    MPI_Finalize();
    return 0;
}