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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>

/* GPU trace format version */
static const int GPU_TRACE_VERSION = 14;

enum class MemorySpace {
    NONE,
    LOCAL,             // local memory operation
    GENERIC,           // generic memory operation
    GLOBAL,            // global memory operation
    SHARED,            // shared memory operation
    CONSTANT,          // constant memory operation
    GLOBAL_TO_SHARED,  // read from global memory then write to shared memory
    SURFACE,   // surface memory operation
    TEXTURE,   // texture memory operation
};
static const char* const MemorySpaceStr[] = {
    "NONE", "LOCAL", "GENERIC", "GLOBAL", "SHARED", "CONSTANT",
    "GLOBAL_TO_SHARED", "SURFACE", "TEXTURE",
};

static const std::string CF_TYPE[] = {
    "NOT_CF",
    "CF_BR",
    "CF_CBR",
    "CF_CALL",
    "CF_IBR",
    "CF_ICALL",
    "CF_ICO",
    "CF_RET",
    "CF_MITE"
};

static const std::string GPU_NVBIT_OPCODE[] = {
    /* Turing architecture instructions */
    "FADD",
    "FADD32I",
    "FCHK",
    "FFMA32I",
    "FFMA",
    "FMNMX",
    "FMUL",
    "FMUL32I",
    "FSEL",
    "FSET",
    "FSETP",
    "FSWZADD",
    "MUFU",
    "HADD2",
    "HADD2_32I",
    "HFMA2",
    "HFMA2_32I",
    "HMMA",
    "HMUL2",
    "HMUL2_32I",
    "HSET2",
    "HSETP2",
    "DADD",
    "DFMA",
    "DMUL",
    "DSETP",
    "BMMA",
    "BMSK",
    "BREV",
    "FLO",
    "IABS",
    "IADD",
    "IADD3",
    "IADD32I",
    "IDP",
    "IDP4A",
    "IMAD",
    "IMMA",
    "IMNMX",
    "IMUL",
    "IMUL32I",
    "ISCADD",
    "ISCADD32I",
    "ISETP",
    "LEA",
    "LOP",
    "LOP3",
    "LOP32I",
    "POPC",
    "SHF",
    "SHL",
    "SHR",
    "VABSDIFF",
    "VABSDIFF4",
    "F2F",
    "F2I",
    "I2F",
    "I2I",
    "I2IP",
    "FRND",
    "MOV",
    "MOV32I",
    "MOVM",
    "PRMT",
    "SEL",
    "SGXT",
    "SHFL",
    "PLOP3",
    "PSETP",
    "P2R",
    "R2P",
    "LD",
    "LDC",
    "LDG",
    "LDL",
    "LDS",
    "LDSM",
    "ST",
    "STG",
    "STL",
    "STS",
    "MATCH",
    "QSPC",
    "ATOM",
    "ATOMS",
    "ATOMG",
    "RED",
    "CCTL",
    "CCTLL",
    "ERRBAR",
    "MEMBAR",
    "CCTLT",
    "R2UR",
    "S2UR",
    "UBMSK",
    "UBREV",
    "UCLEA",
    "UFLO",
    "UIADD3",
    "UIADD3_64",
    "UIMAD",
    "UISETP",
    "ULDC",
    "ULEA",
    "ULOP",
    "ULOP3",
    "ULOP32I",
    "UMOV",
    "UP2UR",
    "UPLOP3",
    "UPOPC",
    "UPRMT",
    "UPSETP",
    "UR2UP",
    "USEL",
    "USGXT",
    "USHF",
    "USHL",
    "USHR",
    "VOTEU",
    "TEX",
    "TLD",
    "TLD4",
    "TMML",
    "TXD",
    "TXQ",
    "SUATOM",
    "SULD",
    "SURED",
    "SUST",
    "BMOV",
    "BPT",
    "BRA",
    "BREAK",
    "BRX",
    "BRXU",
    "BSSY",
    "BSYNC",
    "CALL",
    "EXIT",
    "JMP",
    "JMX",
    "JMXU",
    "KILL",
    "NANOSLEEP",
    "RET",
    "RPCMOV",
    "RTT",
    "WARPSYNC",
    "YIELD",
    "B2R",
    "BAR",
    "CS2R",
    "DEPBAR",
    "GETLMEMBASE",
    "LEPC",
    "NOP",
    "PMTRIG",
    "R2B",
    "S2R",
    "SETCTAID",
    "SETLMEMBASE",
    "VOTE",

    /* Hopper architecture instructions */
    "HMNMX2",
    "DMMA",
    "VHMNMX",
    "VIADD",
    "VIADDMNMX",
    "VIMNMX",
    "VIMNMX3",
    "I2FP",
    "F2IP",
    "FENCE",
    "LDGDEPBAR",
    "LDGMC",
    "LDGSTS",   
    "STSM",   
    "SYNCS",
    "REDAS",
    "REDG",
    "REDUX",
    "UCGABAR_ARV",
    "UCGABAR_WAIT",
    "UF2FP",
    "ULEPC",
    "USETMAXREG",
    "BGMMA",
    "HGMMA",
    "IGMMA",
    "QGMMA",
    "WARPGROUP",
    "WARPGROUPSET",
    "UBLKCP",
    "UBLKPF",
    "UBLKRED",
    "UTMACCTL",
    "UTMACMDFLUSH",
    "UTMALDG",
    "UTMAPF",
    "UTMAREDG",
    "UTMASTG",
    "ACQBULK",
    "CGAERRBAR",
    "ELECT",
    "ENDCOLLECTIVE",
    "PREEXIT",
};

/* OPCODE classification sets — using unordered_set for O(1) lookup */
static const std::unordered_set<std::string> FP_SET = {
    "FADD", "FADD32I", "FCHK", "FFMA32I", "FFMA", "FMNMX",
    "FMUL", "FMUL32I", "FSEL", "FSET", "FSETP", "FSWZADD",
    "MUFU", "HADD2", "HADD2_32I", "HFMA2", "HFMA2_32I", "HMMA",
    "HMUL2", "HMUL2_32I", "HSET2", "HSETP2",
    "DADD", "DFMA", "DMUL", "DSETP",
    "HMNMX2", "DMMA",
};

static const std::unordered_set<std::string> LD_SET = {
    "LD", "LDC", "LDG", "LDL", "LDS", "LDSM",
    /* Ampere */
    "LDGSTS",
    /* Hopper TMA */
    "UTMALDG",
};

static const std::unordered_set<std::string> ST_SET = {
    "ST", "STG", "STL", "STS",
    /* Hopper */ 
    "STSM",
};

static const std::unordered_set<std::string> BARRIER_SET = {
    "BAR",    /* classic warp barrier (__syncthreads) */
    "SYNCS",  /* mbarrier (async barrier for TMA, etc.) */
};

static const std::unordered_set<std::string> NO_DST_SET = {
    "BRA", "EXIT", "BAR", "BSSY", "BSYNC", "CALL", "BREAK",
    /* Hopper */
    "FENCE", "LDGDEPBAR", "LDGSTS", "SYNCS", "REDAS", "REDG", "UCGABAR_ARV", "UCGABAR_WAIT", "USETMAXREG",
    "WARPGROUP", "WARPGROUPSET", "UBLKCP", "UBLKPF", "UBLKRED", "UTMACCTL", "UTMACMDFLUSH", "UTMALDG", "UTMAPF", "UTMAREDG", "UTMASTG",
    "ACQBULK", "CGAERRBAR", "ENDCOLLECTIVE", "PREEXIT",
};

/* ------------------------------------------------------------------ */
/*  Opcode classification helpers                                      */
/* ------------------------------------------------------------------ */

static inline std::string get_opcode_short(const std::string& opcode) {
    std::size_t dot_pos = opcode.find('.');
    return opcode.substr(0, dot_pos);
}

static inline bool is_fp(const std::string& opcode_short) {
    return FP_SET.count(opcode_short) > 0;
}

static inline bool is_ld(const std::string& opcode_short) {
    return LD_SET.count(opcode_short) > 0;
}

static inline bool is_st(const std::string& opcode_short) {
    return ST_SET.count(opcode_short) > 0;
}

static inline std::string cf_type(const std::string& opcode_short) {
    if (opcode_short == "JMP")
        return "CF_BR";
    else if (opcode_short == "BRA")
        return "CF_CBR";
    else if (opcode_short == "RET")
        return "CF_RET";
    else
        return "NOT_CF";
}

static inline uint8_t num_dst_reg(const std::string& opcode_short) {
    if (is_st(opcode_short) || NO_DST_SET.count(opcode_short))
        return 0;
    else
        return 1;
}

/* ------------------------------------------------------------------ */
/*  TMA structs & helpers                                              */
/* ------------------------------------------------------------------ */

struct TmaDescInfo {
    uint64_t global_addr;
    uint32_t tile_bytes;
    uint32_t element_size;  /* bytes per element (e.g. 4 for float32) */
};

/* Map CU_TENSOR_MAP_DATA_TYPE enum (from descriptor config word) to byte size */
static inline uint32_t tma_dtype_to_bytes(uint32_t dtype_enum) {
    switch (dtype_enum) {
        case 0:  return 1;  /* UINT8 */
        case 1:  return 2;  /* UINT16 */
        case 2:  return 4;  /* UINT32 */
        case 3:  return 4;  /* INT32 */
        case 4:  return 8;  /* UINT64 */
        case 5:  return 8;  /* INT64 */
        case 6:  return 2;  /* FLOAT16 */
        case 7:  return 4;  /* FLOAT32 */
        case 8:  return 8;  /* FLOAT64 */
        case 9:  return 2;  /* BFLOAT16 */
        case 10: return 4;  /* FLOAT32_FTZ */
        case 11: return 4;  /* TFLOAT32 */
        case 12: return 4;  /* TFLOAT32_FTZ */
        default: return 4;  /* assume 4 as fallback */
    }
}

/* Per-launch kernel info: block dimensions needed for barrier thread count */
struct KernelLaunchInfo {
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t total_threads_per_block() const {
        return block_dim_x * block_dim_y * block_dim_z;
    }
};

/* Count the number of commas in a function signature to estimate the number
 * of kernel parameters.  E.g. "kernel(float*, int, CUtensorMap)" -> 3 params
 * (2 commas + 1).  Returns 0 if no parenthesis is found. */
static inline int count_kernel_params(const std::string& func_name) {
    size_t lp = func_name.find('(');
    size_t rp = func_name.rfind(')');
    if (lp == std::string::npos || rp == std::string::npos || rp <= lp + 1)
        return 0;
    std::string args = func_name.substr(lp + 1, rp - lp - 1);
    if (args.empty() || args == "void") return 0;
    int count = 1;
    int depth = 0;
    for (char c : args) {
        if (c == '<' || c == '(') depth++;
        else if (c == '>' || c == ')') depth--;
        else if (c == ',' && depth == 0) count++;
    }
    return count;
}

/* Parse TMA load size from SASS opcode modifier string.
 * Looks for ".LxxxB" pattern (e.g. ".L128B" -> 128, ".L64B" -> 64).
 * If no size modifier is found (e.g. "UTMALDG.1D"), returns 128 -- the
 * standard TMA tile transfer size.  The actual size is determined by the
 * CUtensorMap descriptor at runtime and cannot be read statically. */
static inline uint32_t parse_tma_load_size(const std::string& opcode) {
    size_t pos = opcode.find(".L");
    while (pos != std::string::npos) {
        size_t b_pos = opcode.find('B', pos + 2);
        if (b_pos != std::string::npos) {
            std::string num_str = opcode.substr(pos + 2, b_pos - pos - 2);
            if (!num_str.empty() && std::all_of(num_str.begin(), num_str.end(), ::isdigit)) {
                int size = std::stoi(num_str);
                if (size > 0) return (uint32_t)size;
            }
        }
        pos = opcode.find(".L", pos + 2);
    }
    return 128; // TMA tile default (actual size is in the tensor descriptor)
}

/* ------------------------------------------------------------------ */
/*  General utilities                                                  */
/* ------------------------------------------------------------------ */

static inline bool file_exists(const std::string& file_path) {
    std::ifstream f(file_path);
    return f.good();
}

static inline bool create_a_directory(const std::string& dir_path, bool print) {
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

static inline std::string rm_bracket(std::string kernel_name) {
    kernel_name.erase(std::remove(kernel_name.begin(), kernel_name.end(), ' '), kernel_name.end());
    size_t pos_bracket = kernel_name.find('(');
    return kernel_name.substr(0, pos_bracket);
}
