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
#include <string>
#include <unordered_set>
#include <vector>

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
    "LD", "LDC", "LDG", "LDL", "LDS", "LDSM"
};

static const std::unordered_set<std::string> ST_SET = {
    "ST", "STG", "STL", "STS"
};
