/* Flush channel kernel – compiled to fatbin and embedded via bin2c.
 * This kernel is loaded at runtime with nvbit_load_tool_module() so that
 * it can be launched via nvbit_launch_kernel() to drain the channel buffer
 * before the receiving thread shuts down. */

#include "utils/channel.hpp"

extern "C" __global__ void flush_channel(ChannelDev* ch_dev) {
    ch_dev->flush();
}
