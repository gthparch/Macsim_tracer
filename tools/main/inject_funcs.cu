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

#include <stdint.h>
#include <stdio.h>
#include <cstdarg>
#include <algorithm>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

__inline__ __device__ int get_flat_tid() {
	int tid_b = threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y))); // thread id within a block
	int bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id 
	int tid = tid_b + (bid * blockDim.x * blockDim.y * blockDim.z);
	return tid;
}

__inline__ __device__ uint64_t get_flat_wid() {
	int bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id
	// Use logical warp index within block (deterministic, contiguous)
	// instead of hardware %warpid (non-deterministic SM warp slot)
	int tid_in_block = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
	int warp_in_block = tid_in_block / 32;
	return warp_in_block + (1ull << 16) * bid;
}

extern "C" __device__ __noinline__ void instrument_trace_info(int pred,
                                                       int opcode_id,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev,
                                                       uint8_t size, 
                                                       uint64_t func_addr,
                                                       uint64_t branch_target_addr,
                                                       uint64_t mem_addr,
                                                       uint8_t mem_access_size,
                                                       uint8_t m_addr_space,
                                                       int32_t num_regs...) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = __shfl_sync(active_mask, mem_addr, i);
    }

    int4 cta = get_ctaid();
    ma.active_mask = active_mask;
    ma.grid_launch_id = grid_launch_id;
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_flat_wid();
    ma.opcode_id = opcode_id;

    ma.thread_id = get_flat_tid();
    ma.size = size;
    ma.num_regs = num_regs;
    ma.func_addr = func_addr;
    ma.branch_target_addr = branch_target_addr;
    ma.mem_addr = mem_addr;
    ma.mem_access_size = mem_access_size;
    ma.m_addr_space = m_addr_space;

    memset(ma.reg_id, 0, sizeof(ma.reg_id));

    if (num_regs) {
        va_list vl;
        va_start(vl, num_regs);

        int regs_to_store = (num_regs < 8) ? num_regs : 8;
        for (int i = 0; i < regs_to_store; i++) {
            int val = va_arg(vl, int);
            ma.reg_id[i] = val;
        }
        // Consume remaining args even if we can't store them
        for (int i = regs_to_store; i < num_regs; i++) {
            va_arg(vl, int);
        }
        va_end(vl);
    }

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}
