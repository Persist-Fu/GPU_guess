#include "PCFG.h"
using namespace std;
#include <vector>
#include <string>
#include <cuda_runtime.h>
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}
// ===================================================================
// Kernel: 并行生成猜测字符串
// ===================================================================
__global__ void generate_guesses_kernel(
    // 输入数据 (序列化的segment->ordered_values)
    const char* d_ordered_values_chars,
    const int* d_ordered_values_offsets,
    int num_values, // a->ordered_values.size()

    // 输出缓冲区 (用于收集新生成的字符串)
    char* d_output_guesses_chars,
    int* d_output_guesses_offsets,
    int* d_char_counter,
    int* d_string_counter,
    int max_output_chars,
    int max_output_strings
) {
    // 每个线程负责处理 a->ordered_values 中的一个值
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_values) {
        // --- 1. 在Device端获取第idx个原始字符串 ---
        int start = d_ordered_values_offsets[idx];
        int len = d_ordered_values_offsets[idx + 1] - start;
        const char* value_ptr = d_ordered_values_chars + start;

        // 在这个简化场景下, guess就是这个value
        // 在你的else分支中，这里会是 guess_prefix + value_ptr

        // --- 2. 将这个新生成的guess "emplace_back"到输出缓冲区 ---
        
        // a. 为字符串内容预定空间
        int char_write_offset = atomicAdd(d_char_counter, len);

        // b. 为偏移量数组预定一个索引
        int string_write_index = atomicAdd(d_string_counter, 1);

        // c. 边界检查和数据写入
        if (char_write_offset + len < max_output_chars && string_write_index < max_output_strings) {
            d_output_guesses_offsets[string_write_index] = char_write_offset;
            
            // 拷贝字符串内容
            for (int i = 0; i < len; ++i) {
                d_output_guesses_chars[char_write_offset + i] = value_ptr[i];
            }
        }
    }
}
__global__ void generate_guesses_with_prefix_kernel(
    // 输入数据 (a->ordered_values)
    const char* d_ordered_values_chars,
    const int* d_ordered_values_offsets,
    int num_values,

    // 公共前缀
    const char* d_prefix_chars,
    int prefix_len,

    // 输出缓冲区
    char* d_output_guesses_chars,
    int* d_output_guesses_offsets,
    int* d_char_counter,
    int* d_string_counter,
    int max_output_chars,
    int max_output_strings
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_values) {
        // --- 1. 获取第idx个原始字符串 (value) ---
        int value_start = d_ordered_values_offsets[idx];
        int value_len = d_ordered_values_offsets[idx + 1] - value_start;
        const char* value_ptr = d_ordered_values_chars + value_start;
        
        // --- 2. 计算新生成字符串的总长度 (prefix + value) ---
        int final_len = prefix_len + value_len;

        // --- 3. "emplace_back" 到输出缓冲区 ---
        int char_write_offset = atomicAdd(d_char_counter, final_len);
        int string_write_index = atomicAdd(d_string_counter, 1);

        // --- 4. 边界检查和数据写入 ---
        if (char_write_offset + final_len < max_output_chars && string_write_index < max_output_strings) {
            d_output_guesses_offsets[string_write_index] = char_write_offset;
            
            // a. 先拷贝前缀
            for (int i = 0; i < prefix_len; ++i) {
                d_output_guesses_chars[char_write_offset + i] = d_prefix_chars[i];
            }
            // b. 再拷贝value
            for (int i = 0; i < value_len; ++i) {
                d_output_guesses_chars[char_write_offset + prefix_len + i] = value_ptr[i];
            }
        }
    }
}
// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        int num_values_to_process = pt.max_indices[0]; 
        // Host端: 序列化 a->ordered_values ---
        std::string h_input_chars;
        std::vector<int> h_input_offsets;
        int current_offset = 0;
        for (int i = 0; i < num_values_to_process; ++i) {
            const std::string& s = a->ordered_values[i];
            h_input_offsets.push_back(current_offset);
            h_input_chars.append(s);
            current_offset += s.length();
        }
        h_input_offsets.push_back(current_offset);
        
        // Host端: 准备Device内存 (包括输入和输出) 
        
        // 为输入数据分配和传输
        char* d_input_chars;
        int* d_input_offsets;
        cudaMalloc(&d_input_chars, h_input_chars.size());
        cudaMalloc(&d_input_offsets, h_input_offsets.size() * sizeof(int));
        cudaMemcpy(d_input_chars, h_input_chars.c_str(), h_input_chars.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_offsets, h_input_offsets.data(), h_input_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        
        // 为输出数据预分配缓冲区和原子计数器
        // 这里的估算假设新生成的guesses和输入是完全一样的
        const int MAX_OUTPUT_STRINGS = num_values_to_process + 1;
        const int MAX_OUTPUT_CHARS = h_input_chars.size();

        char* d_output_chars;
        int* d_output_offsets;
        int* d_char_counter;
        int* d_string_counter;
        cudaMalloc(&d_output_chars, MAX_OUTPUT_CHARS);
        cudaMalloc(&d_output_offsets, MAX_OUTPUT_STRINGS * sizeof(int));
        cudaMalloc(&d_char_counter, sizeof(int));
        cudaMalloc(&d_string_counter, sizeof(int));
        cudaMemset(d_char_counter, 0, sizeof(int));
        cudaMemset(d_string_counter, 0, sizeof(int));
        
        // Host端: 配置并启动Kernel 
        int threadsPerBlock = 512;
        int blocksPerGrid = (num_values_to_process + threadsPerBlock - 1) / threadsPerBlock;
        
        generate_guesses_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input_chars, d_input_offsets, num_values_to_process,
            d_output_chars, d_output_offsets,
            d_char_counter, d_string_counter,
            MAX_OUTPUT_CHARS, MAX_OUTPUT_STRINGS
        );
        cudaDeviceSynchronize(); // 等待Kernel执行完成
        
        //Host端: 将结果传回并反序列化
        //获取计数器结果
        int h_final_string_count, h_final_char_count;
        cudaMemcpy(&h_final_string_count, d_string_counter, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_final_char_count, d_char_counter, sizeof(int), cudaMemcpyDeviceToHost);
        
        //准备Host容器并拷贝数据
        std::vector<char> h_output_chars(h_final_char_count);
        std::vector<int> h_output_offsets(h_final_string_count + 1);
        cudaMemcpy(h_output_chars.data(), d_output_chars, h_final_char_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_offsets.data(), d_output_offsets, h_final_string_count * sizeof(int), cudaMemcpyDeviceToHost);
        h_output_offsets[h_final_string_count] = h_final_char_count; // 设置最后一个偏移
       
//  反序列化循环 
try { 
    for (int i = 0; i < h_final_string_count; ++i) {
        int start = h_output_offsets[i];
        int len = h_output_offsets[i+1] - start;

        // 在构造字符串之前，打印将要使用的数据
        if (len < 0 || start < 0 || start + len > h_final_char_count) {
            /*
             cout << "ERROR: Invalid slice at i=" << i 
                  << "! start=" << start 
                  << ", len=" << len 
                  << ", total_chars=" << h_final_char_count << endl;
                  */
             // 跳过这次循环
             continue; 
        }

        guesses.emplace_back(h_output_chars.data() + start, len);
    }
} catch (const std::length_error& e) {
    //cout << "FATAL: Caught std::length_error during emplace_back. what(): " << e.what() << endl;
    ;
} catch (const std::exception& e) {
    //cout << "FATAL: Caught an unknown exception: " << e.what() << endl;
    ;
}

        // 更新总数
        total_guesses += h_final_string_count;

        // Host端: 清理所有Device内存 ---
        cudaFree(d_input_chars);
        cudaFree(d_input_offsets);
        cudaFree(d_output_chars);
        cudaFree(d_output_offsets);
        cudaFree(d_char_counter);
        cudaFree(d_string_counter);
        
    }
    else
    {
        //  Host端: 准备公共前缀 'guess' 
        string guess_prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            
            if (pt.content[seg_idx].type == 1) guess_prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2) guess_prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3) guess_prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }

        //  Host端: 准备输入数据 (a->ordered_values) 
        segment* a;
       
        if (pt.content[pt.content.size() - 1].type == 1) a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2) a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3) a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];

        int num_values_to_process = pt.max_indices[pt.content.size() - 1];

       
        std::string h_input_chars;
        std::vector<int> h_input_offsets;
        int current_offset = 0;
        for (int i = 0; i < num_values_to_process; ++i) {
            const std::string& s = a->ordered_values[i];
            h_input_offsets.push_back(current_offset);
            h_input_chars.append(s);
            current_offset += s.length();
        }
        h_input_offsets.push_back(current_offset);

        // Host端: 准备Device内存 
        
        //为输入数据 (ordered_values) 分配和传输
        char* d_input_chars;
        int* d_input_offsets;
        cudaMalloc(&d_input_chars, h_input_chars.size());
        cudaMalloc(&d_input_offsets, h_input_offsets.size() * sizeof(int));
        cudaMemcpy(d_input_chars, h_input_chars.c_str(), h_input_chars.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_offsets, h_input_offsets.data(), h_input_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

        //为'guess_prefix' 分配和传输
        char* d_prefix_chars;
        cudaMalloc(&d_prefix_chars, guess_prefix.length());
        cudaMemcpy(d_prefix_chars, guess_prefix.c_str(), guess_prefix.length(), cudaMemcpyHostToDevice);

        //  为输出数据预分配缓冲区和原子计数器
        // 估算输出大小：每个新字符串 = prefix + 平均value长度
        const int MAX_OUTPUT_STRINGS = num_values_to_process + 1;
        const int MAX_OUTPUT_CHARS = (guess_prefix.length() * num_values_to_process) + h_input_chars.size();
        
        char* d_output_chars;
        int* d_output_offsets;
        int* d_char_counter;
        int* d_string_counter;
        cudaMalloc(&d_output_chars, MAX_OUTPUT_CHARS);
        cudaMalloc(&d_output_offsets, MAX_OUTPUT_STRINGS * sizeof(int));
        cudaMalloc(&d_char_counter, sizeof(int));
        cudaMalloc(&d_string_counter, sizeof(int));
        cudaMemset(d_char_counter, 0, sizeof(int));
        cudaMemset(d_string_counter, 0, sizeof(int));

        //  Host端: 配置并启动新Kernel 
        int threadsPerBlock = 512;
        int blocksPerGrid = (num_values_to_process + threadsPerBlock - 1) / threadsPerBlock;
        
        generate_guesses_with_prefix_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input_chars, d_input_offsets, num_values_to_process,
            d_prefix_chars, guess_prefix.length(),
            d_output_chars, d_output_offsets,
            d_char_counter, d_string_counter,
            MAX_OUTPUT_CHARS, MAX_OUTPUT_STRINGS
        );
        cudaDeviceSynchronize();

        //Host端: 将结果传回并反序列化 
        int h_final_string_count, h_final_char_count;
        cudaMemcpy(&h_final_string_count, d_string_counter, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_final_char_count, d_char_counter, sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<char> h_output_chars(h_final_char_count);
        std::vector<int> h_output_offsets(h_final_string_count + 1);
        cudaMemcpy(h_output_chars.data(), d_output_chars, h_final_char_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_offsets.data(), d_output_offsets, h_final_string_count * sizeof(int), cudaMemcpyDeviceToHost);
        h_output_offsets[h_final_string_count] = h_final_char_count;
        
        try {
    for (int i = 0; i < h_final_string_count; ++i) {
       
        
        //  获取偏移量
        int start = h_output_offsets[i];
        int end = h_output_offsets[i+1];
        
        // 计算长度
        int len = end - start;

        
        //防止 segmentation fault 
        if (start < 0 || len < 0 || (start + len) > h_final_char_count) {
            
            // 跳过这次无效的迭代，继续检查下一个
            continue; 
        }

        
        guesses.emplace_back(h_output_chars.data() + start, len);
    }
} 
catch (const std::length_error& e) {
    // 专门捕获由于长度过大导致的异常
    ;
    /*
    std::cout << "FATAL EXCEPTION: Caught std::length_error during string creation." << std::endl;
    std::cout << "                 what(): " << e.what() << std::endl;
    std::cout << "                 This usually means a calculated length was a huge or negative number." << std::endl;
    */
} 
catch (const std::exception& e) {
    // 捕获其他所有标准异常
    /*
    std::cout << "FATAL EXCEPTION: Caught an unknown std::exception." << std::endl;
    std::cout << "                 what(): " << e.what() << std::endl;
    */
    ;
}
catch (...) {
    // 捕获任何其他未知类型的异常
    //std::cout << "FATAL EXCEPTION: Caught an unknown non-standard exception." << std::endl;
}

        total_guesses += h_final_string_count;

        // --- 7. Host端: 清理所有Device内存 ---
        cudaFree(d_input_chars);
        cudaFree(d_input_offsets);
        cudaFree(d_prefix_chars);
        cudaFree(d_output_chars);
        cudaFree(d_output_offsets);
        cudaFree(d_char_counter);
        cudaFree(d_string_counter);
    }
        /*
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
            */
    
}



