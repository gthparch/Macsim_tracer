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

#pragma once

#include <stdint.h>

#define MAX_GPU_SRC_NUM 4
#define MAX_GPU_DST_NUM 4

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
  uint64_t grid_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t warp_id;
  int opcode_id;
  uint64_t addrs[32];

  int thread_id;
  uint8_t size;
  int32_t num_regs;
  uint16_t reg_id[8];
  uint32_t active_mask;
  uint64_t func_addr;
  uint64_t branch_target_addr;
  uint64_t mem_addr;
  uint8_t mem_access_size;
  uint8_t m_addr_space;
  /* 32 lanes, each thread can store up to 5 register values */
  // uint32_t reg_vals[32][8];
} mem_access_t;

typedef struct trace_info_nvbit_small_s {
  uint8_t m_opcode;
  bool m_is_fp;
  bool m_is_load;
  uint8_t m_cf_type;
  uint8_t m_num_read_regs;
  uint8_t m_num_dest_regs;
  uint16_t m_src[MAX_GPU_SRC_NUM];
  uint16_t m_dst[MAX_GPU_DST_NUM];
  uint8_t m_size;

  uint32_t m_active_mask;
  uint32_t m_br_taken_mask;
  uint64_t m_inst_addr;
  uint64_t m_br_target_addr;
  union {
    uint64_t m_reconv_inst_addr;
    uint64_t m_mem_addr;
  };
  union {
    uint8_t m_mem_access_size;
    uint8_t m_barrier_id;
  };
  uint16_t m_num_barrier_threads;
  union {
    uint8_t m_addr_space;  // for loads, stores, atomic, prefetch(?)
    uint8_t m_level;  // for membar
  };
  uint8_t m_cache_level;  // for prefetch?
  uint8_t m_cache_operator;  // for loads, stores, atomic, prefetch(?)
//   uint64_t
//     m_next_inst_addr;  // next pc address, not present in raw trace fo

} trace_info_nvbit_small_s;