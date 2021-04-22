#! /usr/bin/python3.6
# -*- coding: utf-8 -*-
# PYNQ DMA and Aurora test

import time
import numpy as np
import subprocess
import struct
import sys
import pynq.lib.dma
from pynq import Overlay
from pynq import MMIO
from pynq import allocate
from mplusiohls import *

def settbl(mmio_axi32):
	print("set table/registers")
	adr = 0
	while (adr<0x0900) :
		mmio_axi32.write(OFFSET_EXT_ADDR,adr)
		mmio_axi32.write(OFFSET_EXT_DATA,5)
		adr = adr+1
	with open(sys.argv[1]) as f:
		while 1:
			line = f.readline()
			if not line: 
				break
			sline = line.split()
			adr = int(sline[0],16)
			dat = int(sline[1],16)
			print(hex(adr),hex(dat) )
			mmio_axi32.write(OFFSET_EXT_ADDR,adr)
			mmio_axi32.write(OFFSET_EXT_DATA,dat)

def call_iic_clock():
   subprocess.run(["sh", "../bin/fic-pll-set.sh"])

def aurora_reset(mmio_axi32):
   call_iic_clock()
   mmio_axi32.write(OFFSET_AURORA_RESET, 1)
   mmio_axi32.write(OFFSET_AURORA_PMAINIT, 1)
   mmio_axi32.write(OFFSET_AURORA_XRST, 1)
   time.sleep(0.5)
   mmio_axi32.write(OFFSET_AURORA_PMAINIT, 0)
   time.sleep(0.1)
   mmio_axi32.write(OFFSET_AURORA_RESET, 0)
   time.sleep(0.1)
   mmio_axi32.write(OFFSET_AURORA_XRST, 0)
   time.sleep(0.1)
   mmio_axi32.write(OFFSET_AURORA_XRST, 1)

def moninf(mmio_axi32):
    mmio_axi32.write(OFFSET_EXT_ADDR,0xfff9)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print("link up = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,0xfffc)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print("  Channel up = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,0xfff8)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print("  Slotnum = 0x{:X}".format(n>>1))
    mmio_axi32.write(OFFSET_EXT_ADDR,0xff80)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print("  Timer(usec) = ",n/100)

def monpktc(sw, mmio_axi32):
    print("sw:",sw)
    caddr = 0xff00 + (sw<<4)
#    print("caddr:",hex(caddr))
 #   caddr = 0xff10
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" in0 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+1)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" in1 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+2)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" in2 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+3)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" in3 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+4)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" in4 = 0x{:X}".format(n))
#
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+8)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" out0 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+9)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" out1 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+10)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" out2 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+11)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" out3 = 0x{:X}".format(n), end="")
    mmio_axi32.write(OFFSET_EXT_ADDR,caddr+12)
    n = mmio_axi32.read(OFFSET_EXT_DATA)
    print(" out4 = 0x{:X}".format(n))
