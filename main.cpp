#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h> // MPI: 包含 MPI 头文件

using namespace std;
using namespace chrono;

// MPI 编译指令示例:
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// mpirun -np 4 ./main

int main() // MPI: main 函数需要接收 argc 和 argv
{
    // MPI: 初始化 MPI 环境
	MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	double time_hash = 0;
	double time_guess = 0;
	double time_train = 0;
	PriorityQueue q;

    // --- 模型训练 ---
    // 所有进程都加载和训练自己的模型副本。
	if (rank == 0) { // 只让主进程计时
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    } else {
        // 其他进程也必须加载模型，但不需要计时
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
    
    // MPI: 使用 MPI_Barrier 确保所有进程都完成了模型训练才继续
    MPI_Barrier(MPI_COMM_WORLD);


    // --- 优先队列初始化 ---
    // 只有主进程(rank 0)负责初始化优先队列。
	if (rank == 0) {
        q.init();
        cout << "here" << endl;
    }

	int curr_num = 0;
    // MPI: 计时和 history 只在主进程中有意义
    auto start = (rank == 0) ? system_clock::now() : system_clock::time_point{};
	int history = 0;
    bool continue_looping = true;

    // --- 主循环 ---
    // 循环条件由主进程决定，并广播给其他进程
	do
	{
        // 所有进程都调用 PopNext，它内部有并行的分发、计算、收集逻辑。
		q.PopNext();

        // MPI: ---------------------------------------------------------------
        // 以下所有逻辑（计数、打印、哈希、终止判断）都只在主进程(rank 0)执行
        // 这是对你原始逻辑的直接保留
        if (rank == 0) {
            
            // 你的原始逻辑：更新 total_guesses
            q.total_guesses = q.guesses.size();
		    
            // 你的原始逻辑：检查打印条件
            if (q.total_guesses - curr_num >= 100000)
		    {
			    cout << "Guesses generated: " << history + q.total_guesses << endl;
			    curr_num = q.total_guesses;

                // 你的原始逻辑：检查终止条件
                int generate_n = 10000000;
			    if (history + q.total_guesses > generate_n)
			    {
				    auto end = system_clock::now();
				    auto duration = duration_cast<microseconds>(end - start);
				    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
				    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
				    cout << "Hash time:" << time_hash << "seconds" << endl;
				    cout << "Train time:" << time_train << "seconds" << endl;
                    continue_looping = false; // 标记循环应该结束
			    }
		    }

            // 你的原始逻辑：检查哈希触发条件
		    if (curr_num > 1000000)
		    {
			    auto start_hash = system_clock::now();

                // 你的原始哈希逻辑
			    bit32 batch_states[4][4];
			    size_t total = q.guesses.size();
			    for (size_t i = 0; i < total; i += 4) {
				    std::string batch[4];
				    size_t remain = total - i;
				    size_t batch_size = (remain >= 4) ? 4 : remain;
				    for (size_t j = 0; j < batch_size; ++j) {
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
            
            // 你的原始逻辑：检查队列是否为空作为终止条件
            if (q.priority.empty()) {
                continue_looping = false;
            }
        }
        // MPI: ---------------------------------------------------------------


        // MPI: 主进程将 "是否继续" 的决定广播给所有其他进程
        MPI_Bcast(&continue_looping, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

	} while (continue_looping);


    // MPI: 释放 MPI 资源
	MPI_Finalize();
	return 0;
}