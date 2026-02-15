/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <list>
#include <cuda_runtime.h>
#include <cstdlib>
#include <sys/stat.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"
#include "mem_trace.h"

#include "tool_func/flush_channel.c"

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec

/* LRU file handle cache keyed by (kernel_id, warp_id) pair.
 * Avoids string allocation and string hashing on the hot path.
 * File path is only constructed on cache miss (when opening a new file). */

struct FileKey {
    int kernel_id;
    uint64_t warp_id;
    bool operator==(const FileKey& o) const {
        return kernel_id == o.kernel_id && warp_id == o.warp_id;
    }
};

struct FileKeyHash {
    size_t operator()(const FileKey& k) const {
        size_t h1 = std::hash<int>{}(k.kernel_id);
        size_t h2 = std::hash<uint64_t>{}(k.warp_id);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

class LRUFileCache {
public:
    explicit LRUFileCache(size_t max_open, std::ios_base::openmode mode)
        : max_open_(max_open), open_mode_(mode) {}

    ~LRUFileCache() { close_all(); }

    /* Get an open ofstream for the given (kernel_id, warp_id).
     * path is only used on cache miss to open the file. */
    std::ofstream& get(const FileKey& key, const char* path) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            /* Cache hit — move to front of LRU list */
            lru_.splice(lru_.begin(), lru_, it->second.lru_it);
            return it->second.stream;
        }
        /* Cache miss — evict if full */
        if (cache_.size() >= max_open_) {
            const FileKey& victim = lru_.back();
            cache_[victim].stream.close();
            cache_.erase(victim);
            lru_.pop_back();
        }
        /* Open new file and insert at front */
        lru_.push_front(key);
        auto& entry = cache_[key];
        entry.lru_it = lru_.begin();
        entry.stream.open(path, open_mode_);
        return entry.stream;
    }

    /* Close all cached file handles */
    void close_all() {
        for (auto& kv : cache_) {
            if (kv.second.stream.is_open())
                kv.second.stream.close();
        }
        cache_.clear();
        lru_.clear();
    }

private:
    struct Entry {
        std::ofstream stream;
        std::list<FileKey>::iterator lru_it;
    };
    size_t max_open_;
    std::ios_base::openmode open_mode_;
    std::list<FileKey> lru_;
    std::unordered_map<FileKey, Entry, FileKeyHash> cache_;
};

#define CHANNEL_SIZE (1l << 30)

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev;
    ChannelHost channel_host;

    /* tool module */
    CUmodule tool_module;

    /* flush channel function */
    CUfunction flush_channel_func;

    enum class RecvThreadState { WORKING, STOP, FINISHED };
    // After initialization, set it to WORKING to make recv thread get data,
    // parent thread sets it to STOP to make recv thread stop working.
    // recv thread sets it to FINISHED when it cleans up.
    // parent thread should wait until the state becomes FINISHED to clean up.
    volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

    // whether the context and the channel need a synchronization.
    bool need_sync = false;
};

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t file_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t kernel_begin_interval = 0;
uint32_t kernel_end_interval = UINT32_MAX;
int verbose = 0;
int trace_debug = 0;
int overwrite = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* O(1) lookup tables for opcode/cf_type — built once in nvbit_at_init() */
std::unordered_map<std::string, uint8_t> opcode_short_to_int;
std::unordered_map<std::string, uint8_t> cf_type_to_int;

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

/* Trace file path */
std::string trace_path = "./default/";
std::string sampled_kernel_path = "";
std::string compress_path = "./compress";
std::vector<int> sampled_kernel_ids;

bool file_exists(const std::string& file_path) {
    std::ifstream f(file_path);
    return f.good();
}

/* To distinguish different Kernels */
class UniqueKernelStore {
public:
    int add(const std::string& str) {
        int new_id = kernels.size();
        kernels.push_back(str);
        instr_counts.emplace_back();
        return new_id;
    }

    const std::string& get_string(int id) const {
        return kernels[id];
    }

    /* Write trace.txt and trace_info.txt for each kernel directory.
     * With logical warp IDs, no renaming is needed — IDs are already
     * contiguous within each block. */
    void write_trace_files() {
        for (int i = 0; i < static_cast<int>(kernels.size()); i++) {
            std::string kernel_dir = trace_path + "Kernel" + std::to_string(i);
            if (!file_exists(kernel_dir)) continue;

            std::ofstream file_trace(kernel_dir + "/trace.txt", std::ios_base::app);
            std::ofstream file_info_trace(kernel_dir + "/trace_info.txt", std::ios_base::app);

            /* Collect and sort warp IDs for deterministic output */
            std::vector<uint64_t> sorted_warp_ids;
            sorted_warp_ids.reserve(instr_counts[i].size());
            for (const auto& kv : instr_counts[i]) {
                sorted_warp_ids.push_back(kv.first);
            }
            std::sort(sorted_warp_ids.begin(), sorted_warp_ids.end());

            file_trace << sorted_warp_ids.size() << std::endl;
            for (uint64_t warp_id : sorted_warp_ids) {
                file_trace << warp_id << " " << "0" << std::endl;
                file_info_trace << warp_id << " " << instr_counts[i][warp_id] << std::endl;
            }

            file_trace.close();
            file_info_trace.close();
        }
    }

    /* counting the number of instructions per one trace*.raw */
    std::vector<std::unordered_map<uint64_t, uint64_t>> instr_counts;
    std::vector<std::string> kernels;
};

/* Opcode classification helpers — extract short opcode once */
static std::string get_opcode_short(const std::string& opcode) {
    std::size_t dot_pos = opcode.find('.');
    return opcode.substr(0, dot_pos);
}

bool is_fp(const std::string& opcode_short) {
    return FP_SET.count(opcode_short) > 0;
}

bool is_ld(const std::string& opcode_short) {
    return LD_SET.count(opcode_short) > 0;
}

bool is_st(const std::string& opcode_short) {
    return ST_SET.count(opcode_short) > 0;
}

// Check if the directory exists. If there isn't, make one. 
bool create_a_directory(const std::string& dir_path, bool print) {
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
        if (mkdir(dir_path.c_str(), 0777) != 0) {
            std::cerr << "Error: Failed to create directory: " << dir_path << std::endl;
            return false;
        }
        if (print) std::cout << "Directory " << dir_path << " created." << std::endl;
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: Path is not a directory: " << dir_path << std::endl;
        return false;
    } else {
        if (print) std::cout << "Directory " << dir_path << " already exists." << std::endl;
    }
    return true;
}

// Remove bracket in kernel name 
std::string rm_bracket (std::string kernel_name){
    kernel_name.erase(std::remove(kernel_name.begin(), kernel_name.end(), ' '), kernel_name.end());
    size_t pos_bracket = kernel_name.find('(');
    return kernel_name.substr(0, pos_bracket);
}

std::string cf_type(const std::string& opcode_short){ 
    if (opcode_short == "JMP")
        return "CF_BR";
    else if (opcode_short == "BRA")
        return "CF_CBR";
    else if (opcode_short == "RET")
        return "CF_RET";
    else 
        return "NOT_CF";
}

uint8_t num_dst_reg(const std::string& opcode_short) {
    if (is_st(opcode_short) || NO_DST_SET.count(opcode_short))
        return 0;
    else
        return 1;
}

void src_reg(mem_access_t* ma, uint8_t n_dst, uint16_t* src_reg_){
    for(int i=n_dst, j=0; i<ma->num_regs; i++, j++){
        src_reg_[j] = ma->reg_id[i];
    }
}

void dst_reg(mem_access_t* ma, uint8_t n_dst, uint16_t* dst_reg_){
    for(int i=0; i<n_dst; i++){
        dst_reg_[i] = ma->reg_id[i];
    }
}

/* Compute distinct 128B-aligned sector addresses touched by active threads.
 * Fills the provided vector with sorted, unique sector start addresses.
 * The caller should reuse the vector across calls (clear + refill). */
void compute_coalesced_sectors(
    uint64_t* mem_addrs, uint32_t active_mask,
    std::vector<uint64_t>& out_sectors) {
    out_sectors.clear();
    for (int i = 0; i < 32; i++) {
        if ((active_mask & (1u << i)) && mem_addrs[i] != 0) {
            uint64_t aligned = mem_addrs[i] & ~(uint64_t)127;
            out_sectors.push_back(aligned);
        }
    }
    /* Sort + deduplicate (at most 32 elements, very fast) */
    std::sort(out_sectors.begin(), out_sectors.end());
    out_sectors.erase(std::unique(out_sectors.begin(), out_sectors.end()), out_sectors.end());
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval on each kernel where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval on each kernel where to apply instrumentation");
    GET_VAR_INT(
        kernel_begin_interval, "KERNEL_BEGIN", 0,
        "Beginning of the kernel interval where to generate traces");
    GET_VAR_INT(
        kernel_end_interval, "KERNEL_END", UINT32_MAX,
        "End of the kernel interval where to generate traces");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_STR(trace_path, "TRACE_PATH", "Path to trace file. Default: './default/'");
    GET_VAR_STR(compress_path, "COMPRESSOR_PATH", "Path to the compressor binary file. Default: './compress'");
    GET_VAR_INT(trace_debug, "DEBUG_TRACE", 0, "Generate human-readable debug traces together");
    GET_VAR_INT(overwrite, "OVERWRITE", 0, "Overwrite the previously generated traces in TRACE_PATH directory");
    GET_VAR_STR(sampled_kernel_path, "SAMPLED_KERNEL_INFO", "Path to the file that contains the list of kernels to be sampled. Default: ''");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&file_mutex, &attr);

    trace_path = trace_path + "/";

    /* Build O(1) lookup tables for opcode and cf_type */
    for (int i = 0; i < (int)(sizeof(GPU_NVBIT_OPCODE) / sizeof(GPU_NVBIT_OPCODE[0])); i++) {
        opcode_short_to_int[GPU_NVBIT_OPCODE[i]] = (uint8_t)i;
    }
    for (int i = 0; i < (int)(sizeof(CF_TYPE) / sizeof(CF_TYPE[0])); i++) {
        cf_type_to_int[CF_TYPE[i]] = (uint8_t)i;
    }

    create_a_directory(rm_bracket(trace_path), false);

    if (overwrite != 0){
        if (system(("rm -rf " + trace_path + "Kernel*").c_str()) != 0){
            std::cerr << "Error: Failed to rm -rf " + trace_path + "Kernel*" << std::endl;
            assert(0);
        }
        if (system(("rm -f '" + trace_path + "kernel_config.txt' '" + trace_path + "kernel_names.txt' '" + trace_path + "compress' '" + trace_path + "'sampled*").c_str()) != 0){
            std::cerr << "Error: Failed to rm -f config/names/compress files" << std::endl;
            assert(0);
        }
    }

    std::ofstream file_kernel_config(trace_path + "kernel_config.txt", std::ios_base::app);
    file_kernel_config << "nvbit" << std::endl;
    file_kernel_config << GPU_TRACE_VERSION << std::endl;
    file_kernel_config << "-1" << std::endl;
    file_kernel_config.close();

    // Open the sampled_kernel_path file and ignore the first two lines. The numbers are separated in spaces. 
    // The second number is the number of kernels and the rest are the kernel ids.
    // Every kernel id is stored in a single vector.
    if (sampled_kernel_path != ""){
        std::ifstream file(sampled_kernel_path);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open file for reading: " << sampled_kernel_path << std::endl;
            assert(0);
        }
        std::string line;
        std::getline(file, line);
        std::cout << "Using " << line << std::endl;
        std::getline(file, line);
        std::cout << line << std::endl;

        while (std::getline(file, line)){
            if (line.empty()) 
                break;
            std::istringstream iss(line);
            int skip, num_kernels;
            iss >> skip >> num_kernels;
            int kernel_id, cnt = 0;
            while (iss >> kernel_id) {
                sampled_kernel_ids.push_back(kernel_id);
                cnt++;
            }
            if (num_kernels != cnt) {
                std::cerr << "Error: The number of kernels in the file does not match the actual number of kernels." << std::endl;
                assert(0);
            }
        }
        file.close();

        // Copy the sampled_kernels_info.txt file to the trace_path directory.
        if (system(("cp " + sampled_kernel_path + " " + trace_path + "sampled_kernels_info.txt").c_str()) != 0){
            std::cerr << "Error: Failed to cp " + sampled_kernel_path + " " + trace_path + "sampled_kernels_info.txt" << std::endl;
            assert(0);
        }
    }
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* Kernel - id mapping */
UniqueKernelStore store;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* get vector of instructions of function "f" */
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf(
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }
            if (verbose) {
                instr->printDecoded();
            }

            if (opcode_to_id_map.find(instr->getOpcode()) ==
                opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
            }

            int opcode_id = opcode_to_id_map[instr->getOpcode()];
            std::vector<int> reg_num_list;
            // int mref_idx = 0;
            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const InstrType::operand_t* op = instr->getOperand(i);

                /* count # of regs */
                if (op->type == InstrType::OperandType::REG || 
                    op->type == InstrType::OperandType::PRED || 
                    op->type == InstrType::OperandType::UREG || 
                    op->type == InstrType::OperandType::UPRED) {
                    // for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
                    //     reg_num_list.push_back(op->u.reg.num + reg_idx);
                    // } // for 64-bit-access the instrs, they use two registers. but in this case, we only need the number of regs in the instr itself
                    reg_num_list.push_back(op->u.reg.num);
                }
            }

            nvbit_insert_call(instr, "instrument_trace_info", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            // }
            /* add "space" for kernel function pointer that will be set
                    * at launch time (64 bit value at offset 0 of the dynamic
                    * arguments)*/
            nvbit_add_call_arg_launch_val64(instr, 0);
            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
            /* instruction size */
            nvbit_add_call_arg_const_val32(instr, 16); // 128bit instructions
            /* PC address */
            nvbit_add_call_arg_const_val64(instr, nvbit_get_func_addr(ctx, func) + instr->getOffset());
            /* Branch target address (care about predicates?) */
            uint64_t branchAddrOffset = (std::string(instr->getOpcodeShort()) == "BRA") ? 
                instr->getOperand(instr->getNumOperands()-1)->u.imm_uint64.value + nvbit_get_func_addr(ctx, func) :
                nvbit_get_func_addr(ctx, func) + instr->getOffset() + 0x10;
            nvbit_add_call_arg_const_val64(instr, branchAddrOffset);
            /* MEM access address / reconv(??) address */
            /* LDGSTS (cp.async) has 2 MREFs in SASS order: [shared_dst], [global_src]
             *   MREF id=0 → shared (destination)
             *   MREF id=1 → global (source)
             * We trace only the global memory access and report the space as GLOBAL. */
            std::string opcode_str(instr->getOpcode());
            bool is_ldgsts = (opcode_str.find("LDGSTS") != std::string::npos);
            nvbit_add_call_arg_mref_addr64(instr, is_ldgsts ? 1 : 0);
            /* MEM access size */
            nvbit_add_call_arg_const_val32(instr, (uint8_t)instr->getSize()); 
            /* MEM addr space — override GLOBAL_TO_SHARED → GLOBAL for LDGSTS */
            nvbit_add_call_arg_const_val32(instr,
                is_ldgsts ? (uint8_t)InstrType::MemorySpace::GLOBAL
                          : (uint8_t)instr->getMemorySpace());
            /* how many register values are passed next */
            nvbit_add_call_arg_const_val32(instr, (int)reg_num_list.size());

            for (int num : reg_num_list) {
                /* last parameter tells it is a variadic parameter passed to
                * the instrument function record_reg_val() */
                nvbit_add_call_arg_const_val32(instr, num, true);
            }
            // std::cout << std::endl;
            cnt++;
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    pthread_mutex_lock(&mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&mutex);
        return;
    }
    skip_callback_flag = true;

    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {

        /* Extract launch parameters from the appropriate struct */
        CUfunction func;
        unsigned int gridDimX, gridDimY, gridDimZ;
        unsigned int blockDimX, blockDimY, blockDimZ;
        unsigned int sharedMemBytes;

        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
            gridDimX = p->config->gridDimX;
            gridDimY = p->config->gridDimY;
            gridDimZ = p->config->gridDimZ;
            blockDimX = p->config->blockDimX;
            blockDimY = p->config->blockDimY;
            blockDimZ = p->config->blockDimZ;
            sharedMemBytes = p->config->sharedMemBytes;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
            gridDimX = p->gridDimX;
            gridDimY = p->gridDimY;
            gridDimZ = p->gridDimZ;
            blockDimX = p->blockDimX;
            blockDimY = p->blockDimY;
            blockDimZ = p->blockDimZ;
            sharedMemBytes = p->sharedMemBytes;
        }

        /* Make sure GPU is idle */
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cudaGetLastError() == %s\n", cudaGetErrorString(err));
            fflush(stdout);
            assert(err == cudaSuccess);
        }

        if (!is_exit) {
            ctx_state->need_sync = true;

            /* instrument */
            instrument_function_if_needed(ctx, func);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

            /* get function name and pc */
            uint64_t pc = nvbit_get_func_addr(ctx, func);

            /* set grid launch id at launch time */
            nvbit_set_at_launch(ctx, func, (uint64_t)grid_launch_id);

            /* enable instrumented code to run */
            nvbit_enable_instrumented(ctx, func, false);

            /* Making proper directories for trace files */
            std::string func_name = nvbit_get_func_name(ctx, func); // this function fetches the argument part too..
            int kernel_id = store.add(rm_bracket(func_name));
            
            int numBlocks;
            CUresult result;
            result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func, blockDimX * blockDimY * blockDimZ, sharedMemBytes); 
            if (result != CUDA_SUCCESS) {
                const char* pStr = NULL; // Pointer to store the error string
                cuGetErrorString(result, &pStr);
                printf("[Error] cuOccupancyMaxActiveBlocksPerMultiprocessor() = %s\n", pStr);
                fflush(stdout);
                assert(result == CUDA_SUCCESS);
            }

            // If the sampled_kernel_info file exists, check if the kernel is in the list.
            // If the sampled_kernel_info file doesn't exist but the kernel interval is given, enable the instrumented code.
            bool found = !sampled_kernel_ids.empty() && std::find(sampled_kernel_ids.begin(), sampled_kernel_ids.end(), grid_launch_id) != sampled_kernel_ids.end();
            bool within_range = grid_launch_id >= kernel_begin_interval && grid_launch_id < kernel_end_interval;

            // printf("found: %d, grid_launch_id: %d\n", found, grid_launch_id);

            if ((found || sampled_kernel_ids.empty()) && within_range) {
                std::string kernel_dir = trace_path + "Kernel" + std::to_string(grid_launch_id);
                nvbit_enable_instrumented(ctx, func, true);

                create_a_directory(kernel_dir, false);

                std::ofstream file_trace(kernel_dir + "/" + "trace.txt");
                file_trace << "nvbit" << std::endl;
                file_trace << GPU_TRACE_VERSION << std::endl;
                file_trace << numBlocks << std::endl;
                file_trace.close();

                std::ofstream file_kernel_config(trace_path + "kernel_config.txt", std::ios_base::app);
                file_kernel_config << "./Kernel" + std::to_string(grid_launch_id) + "/trace.txt" << std::endl;
                file_kernel_config.close();

            }

            std::ofstream file_kernel_names(trace_path + "kernel_names.txt", std::ios_base::app);
            file_kernel_names << "Kernel" << grid_launch_id << " name: " << func_name.c_str() << std::endl <<
            "  Grid size: (" << gridDimX << ", " << gridDimY << ", " << gridDimZ << "), " <<
            "Block size: (" << blockDimX << ", " << blockDimY << ", " << blockDimZ << "), " <<
            "maxBlockPerCore: " << numBlocks <<
            ", # of regs: " << nregs << ", static shared mem: " << shmem_static_nbytes << ", dynamic shared mem: " << sharedMemBytes << std::endl;

            /* increment grid launch id for next launch */
            grid_launch_id++;
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    /* LRU file handle caches — avoid open/close syscalls per instruction */
    LRUFileCache raw_cache(16384, std::ios::binary | std::ios_base::app);
    LRUFileCache debug_cache(16384, std::ios_base::app);

    /* Reusable buffers — allocated once, cleared each iteration */
    std::vector<trace_info_nvbit_small_s> children_trace;
    std::vector<uint64_t> sectors;
    children_trace.reserve(32);
    sectors.reserve(32);

    /* Path buffer — avoids heap allocs for string concat */
    char path_buf[512];

    while (ctx_state->recv_thread_done == CTXstate::RecvThreadState::WORKING) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                mem_access_t* ma =
                    (mem_access_t*)&recv_buffer[num_processed_bytes];

                /* when we receive a CTA_id_x it means all the kernels
                 * completed, this is the special token we receive from the
                 * flush channel kernel that is issues at the end of the
                 * context */
                if (ma->cta_id_x == -1) {
                    break;
                }

                int kernel_id = static_cast<int>(ma->grid_launch_id);
                const std::string& opcode = id_to_opcode_map[ma->opcode_id];
                std::string opcode_short = get_opcode_short(opcode);

                /* Increment instruction count for this warp */
                store.instr_counts[kernel_id][ma->warp_id]++;

                /* O(1) opcode lookup */
                uint8_t opcode_int = 255;
                auto opit = opcode_short_to_int.find(opcode_short);
                if (opit != opcode_short_to_int.end()) {
                    opcode_int = opit->second;
                }

                /* O(1) cf_type lookup */
                uint8_t cf_type_int = 255;
                std::string cf_type_str = cf_type(opcode_short);
                auto cfit = cf_type_to_int.find(cf_type_str);
                if (cfit != cf_type_to_int.end()) {
                    cf_type_int = cfit->second;
                }

                /* Cache is_ld / is_st results */
                bool is_load = is_ld(opcode_short);
                bool is_store = is_st(opcode_short);
                bool is_mem = is_load || is_store;

                uint8_t num_dst_reg_ = num_dst_reg(opcode_short);
                uint8_t num_src_reg_ = ma->num_regs - num_dst_reg_;
                if (ma->num_regs <= num_dst_reg_) num_src_reg_ = 0;
                uint16_t src_reg_[MAX_GPU_SRC_NUM];
                uint16_t dst_reg_[MAX_GPU_DST_NUM];
                memset(src_reg_, 0, sizeof(src_reg_));
                memset(dst_reg_, 0, sizeof(dst_reg_));
                src_reg(ma, num_dst_reg_, src_reg_);
                dst_reg(ma, num_dst_reg_, dst_reg_);
                uint8_t inst_size = ma->size;
                uint32_t active_mask = ma->active_mask;
                uint32_t br_taken_mask = 0; // should be added soon
                uint64_t func_addr = ma->func_addr;
                uint64_t br_target_addr = ma->branch_target_addr;
                uint64_t mem_addr = is_mem ? ma->mem_addr : 0;
                uint8_t mem_access_size = ma->mem_access_size;
                uint16_t m_num_barrier_threads = 0;
                uint8_t m_addr_space = ma->m_addr_space;
                const char* m_addr_space_str = MemorySpaceStr[m_addr_space];
                uint8_t m_cache_level = 0;
                uint8_t m_cache_operator = 0;

                trace_info_nvbit_small_s cur_trace;
                cur_trace.m_opcode = opcode_int;
                cur_trace.m_is_fp = is_fp(opcode_short);
                cur_trace.m_is_load = is_load;
                cur_trace.m_cf_type = cf_type_int;
                cur_trace.m_num_read_regs = num_src_reg_;
                cur_trace.m_num_dest_regs = num_dst_reg_;
                memcpy(cur_trace.m_src, src_reg_, sizeof(src_reg_));
                memcpy(cur_trace.m_dst, dst_reg_, sizeof(dst_reg_));
                cur_trace.m_size = inst_size;
                cur_trace.m_active_mask = active_mask;
                cur_trace.m_br_taken_mask = br_taken_mask;
                cur_trace.m_inst_addr = func_addr;
                cur_trace.m_br_target_addr = br_target_addr;
                cur_trace.m_mem_addr = mem_addr;
                cur_trace.m_mem_access_size = mem_access_size;
                cur_trace.m_num_barrier_threads = m_num_barrier_threads;
                cur_trace.m_addr_space = m_addr_space;
                cur_trace.m_cache_level = m_cache_level;
                cur_trace.m_cache_operator = m_cache_operator;

                /* Compute 128B-aligned sectors touched by this warp instruction */
                children_trace.clear();
                if (is_mem) {
                    compute_coalesced_sectors(ma->addrs, active_mask, sectors);
                    int num_sectors = (int)sectors.size();

                    /* Parent trace gets the first sector address */
                    if (num_sectors > 0) {
                        cur_trace.m_mem_addr = sectors[0];
                    }

                    /* Children get remaining sectors */
                    for (int i = 1; i < num_sectors; i++) {
                        trace_info_nvbit_small_s child_trace;
                        memcpy(&child_trace, &cur_trace, sizeof(child_trace));
                        child_trace.m_mem_addr = sectors[i];
                        child_trace.m_is_fp = true;
                        children_trace.push_back(child_trace);
                    }

                    /* Scale parent access size by total number of sectors */
                    if (num_sectors > 1) {
                        cur_trace.m_mem_access_size *= num_sectors;
                        mem_access_size *= num_sectors;
                    }
                }

                pthread_mutex_lock(&file_mutex);
                FileKey file_key = {kernel_id, ma->warp_id};
                if (trace_debug != 0){
                    // Printing debug traces
                    snprintf(path_buf, sizeof(path_buf), "%sKernel%d/bin_trace_%lu.txt",
                             trace_path.c_str(), kernel_id, (unsigned long)ma->warp_id);
                    std::ofstream& file = debug_cache.get(file_key, path_buf);
                    if (!file.is_open()) {
                        std::cerr << "Error: Failed to open file for writing: " << path_buf << std::endl;
                        assert(0);
                    }
                    file << opcode << '\n';
                    file << std::dec << is_fp(opcode_short) << '\n';
                    file << is_load << '\n';
                    file << cf_type_str << '\n';
                    file << (int)num_src_reg_ << '\n';
                    file << (int)num_dst_reg_ << '\n';
                    file << src_reg_[0] << '\n';
                    file << src_reg_[1] << '\n';
                    file << src_reg_[2] << '\n';
                    file << src_reg_[3] << '\n';
                    file << dst_reg_[0] << '\n';
                    file << dst_reg_[1] << '\n';
                    file << dst_reg_[2] << '\n';
                    file << dst_reg_[3] << '\n';
                    file << (int)inst_size << '\n';
                    file << std::hex << active_mask << '\n';
                    file << br_taken_mask << '\n';
                    file << func_addr << '\n';
                    file << br_target_addr << '\n';
                    file << mem_addr << '\n';
                    file << (int)mem_access_size << '\n';
                    file << (int)m_num_barrier_threads << '\n';
                    file << m_addr_space_str << '\n';
                    file << (int)m_cache_level << '\n';
                    file << (int)m_cache_operator << '\n';
                    file << '\n';
                    if(is_mem) {
                        for (int i = 0; i < (int)children_trace.size(); i++){
                            file << opcode << " (child)" << '\n';
                            file << std::dec << is_fp(opcode_short) << '\n';
                            file << is_load << '\n';
                            file << cf_type_str << '\n';
                            file << (int)num_src_reg_ << '\n';
                            file << (int)num_dst_reg_ << '\n';
                            file << src_reg_[0] << '\n';
                            file << src_reg_[1] << '\n';
                            file << src_reg_[2] << '\n';
                            file << src_reg_[3] << '\n';
                            file << dst_reg_[0] << '\n';
                            file << dst_reg_[1] << '\n';
                            file << dst_reg_[2] << '\n';
                            file << dst_reg_[3] << '\n';
                            file << (int)inst_size << '\n';
                            file << std::hex << active_mask << '\n';
                            file << br_taken_mask << '\n';
                            file << func_addr << '\n';
                            file << br_target_addr << '\n';
                            file << children_trace[i].m_mem_addr << '\n';
                            file << (int)mem_access_size << '\n';
                            file << (int)m_num_barrier_threads << '\n';
                            file << m_addr_space_str << '\n';
                            file << (int)m_cache_level << '\n';
                            file << (int)m_cache_operator << '\n';
                            file << '\n';
                        }
                    }
                }

                snprintf(path_buf, sizeof(path_buf), "%sKernel%d/bin_trace_%lu.raw",
                         trace_path.c_str(), kernel_id, (unsigned long)ma->warp_id);
                std::ofstream& file_raw = raw_cache.get(file_key, path_buf);
                if (!file_raw.is_open()) {
                    std::cerr << "Error: Failed to open file for writing: " << path_buf << std::endl;
                    assert(0);
                }
                file_raw.write(reinterpret_cast<const char*>(&cur_trace), sizeof(cur_trace));
                
                if(is_mem) {
                    for (int i = 0; i < (int)children_trace.size(); i++){
                        file_raw.write(reinterpret_cast<const char*>(&children_trace[i]), sizeof(children_trace[i]));
                        auto itt = store.instr_counts[kernel_id].find(ma->warp_id);
                        if (itt != store.instr_counts[kernel_id].end()) {
                            itt->second += 1;
                        }
                    }
                }
                pthread_mutex_unlock(&file_mutex);
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }

    /* Flush and close all cached file handles before writing trace metadata */
    raw_cache.close_all();
    debug_cache.close_all();

    store.write_trace_files();

    free(recv_buffer);
    ctx_state->recv_thread_done = CTXstate::RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
    }
    CTXstate* ctx_state = new CTXstate;
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    ctx_state_map[ctx] = ctx_state;
    pthread_mutex_unlock(&mutex);
}

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Load tool module and find flush_channel function */
    nvbit_load_tool_module(ctx, (const void*)flush_channel_bin,
                           &ctx_state->tool_module);
    nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel",
                                &ctx_state->flush_channel_func);

    ctx_state->recv_thread_done = CTXstate::RecvThreadState::WORKING;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    if (verbose) {
        printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Flush channel if there was a kernel launch before context termination
     * so that any remaining data in the GPU-side buffer is sent to the host
     * recv thread before we shut it down. */
    if (ctx_state->need_sync) {
        void* args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func,
                            1, 1, 1, 1, 1, 1, 0, nullptr, args,
                            nullptr);
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);
    }

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    ctx_state->recv_thread_done = CTXstate::RecvThreadState::STOP;
    while (ctx_state->recv_thread_done != CTXstate::RecvThreadState::FINISHED)
        ;

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;

    if (system(("cp " + compress_path + " " + trace_path).c_str()) != 0){
        std::cout << "cp " + compress_path + " " + trace_path + "was not successful" << std::endl;
    }
    if (system(("cd " + trace_path + " && ./compress").c_str()) != 0){
        std::cout << "cd " + trace_path + " && ./compress was not successful" << std::endl;
    }
    pthread_mutex_unlock(&mutex);
}
