#! /usr/bin/python3.6
# -*- coding: utf-8 -*-

import time
import numpy as np
import subprocess
import struct
import sys
import pynq.lib.dma
import mcubecluster
from pynq import Overlay
from pynq import MMIO
from pynq import allocate
from hls import *
import random
import math


# FPGA bit files: base and experimental version
BIT_FILE = "../overlays/resnet18_1.bit"

print("downloading {}".format(BIT_FILE))
ovl= Overlay(BIT_FILE)

print("ovl.ip_dict.keys() = ",ovl.ip_dict.keys())
# AXI-Lite register access
regs = ovl.ip_dict["axi32regs_0/S_AXI"]
base_addr = regs["phys_addr"]
addr_range = regs["addr_range"]
mmio_axi32 = MMIO(base_addr, addr_range)
mcubecluster.aurora_reset(mmio_axi32)

# DDR4 MMIO
ddr4 = ovl.ddr4_0.mmio
ddr4_array = ddr4.array    # numpy array of uint32
print("size = {} words".format(ddr4_array.size))

# AXI-Lite hls access

regs = ovl.ip_dict["resnet18_1_0"]
base_addr_snd = regs["phys_addr"]
addr_range_snd = regs["addr_range"]
print("snd:base_addr = 0x{:X}".format(base_addr_snd))
print("snd:addr_range = {}".format(addr_range_snd))
mmio_hls = MMIO(base_addr_snd, addr_range_snd)


n = mmio_axi32.read(OFFSET_ID_VERSION)
print("ID_VERSION = 0x{:X}".format(n))
bid = mmio_axi32.read(OFFSET_STATUS)
print("Board ID = 0x{:X}".format(bid))
mmio_axi32.write(OFFSET_LED,0x80|bid)

time.sleep(1)

mmio_axi32.write(OFFSET_EXT_ADDR,0xfff8)
mmio_axi32.write(OFFSET_EXT_DATA,0x2)
mcubecluster.settbl(mmio_axi32)

# set ddr data
ddr = np.loadtxt('../params/ddr1.txt', dtype='uint32')
ddr4_array[0: len(ddr)] = ddr
print('DDRSIZE: ' + str(len(ddr)*2))



print("================Board ID = 0x{:X}=============".format(bid))
mcubecluster.moninf(mmio_axi32)
mcubecluster.monpktc(0,mmio_axi32)
print("============================================\n")

while 1:
    mmio_hls.write(HLS_CNTRL, 1)
    while 1:
        status = mmio_hls.read(HLS_CNTRL)
        if((status&0x2) == 0x2)	: break
    mmio_axi32.write(OFFSET_EXT_ADDR,0xff80)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print("Execution Time (usec) = ",n/100)
