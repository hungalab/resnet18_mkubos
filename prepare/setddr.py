#! /usr/bin/python3.6
# -*- coding: utf-8 -*-
# mras test

import random
import time
import math
import numpy as np
import struct
from fxpmath import Fxp

WORD_BIT = 16
FRACT_BIT = 13

T = 32

def read_c1weight(fname_w, fname_b, array):
	tmp = []
	for line in open(fname_w).readlines():
		tmp.append(float(line))
	for i in range(7):
		for j in range(7):
			for too in range(64):
				for tii in range(3):
					array.append(tmp[too*3*7*7 + tii*7*7 + i*7 + j])
	for line in open(fname_b).readlines():
		array.append(float(line))

def read_weight(fname_w, fname_b, array, OCH, ICH, K):
	tmp = []
	for line in open(fname_w).readlines():
		tmp.append(float(line))
	for to in range(OCH//T):
		for ti in range(ICH//T):
			for i in range(K):
				for j in range(K):
					for too in range(T):
						for tii in range(T):
							array.append(tmp[(to*T+too)*ICH*K*K + (ti*T+tii)*K*K + i*K + j])
	for line in open(fname_b).readlines():
		array.append(float(line))


def read_fcweight(fname_w, fname_b, array):
	for line in open(fname_w).readlines():
		array.append(float(line))
	for line in open(fname_b).readlines():
		array.append(float(line))

def to_fxp_array(input, output, fractbit):
    if len(output) != 0:
        print("ERROR")
    for i in range(len(input)//2):
        x1 = Fxp(input[2*i],   True, 16, fractbit)
        bdata1 = int(x1.hex(), 16).to_bytes(2, byteorder='big')
        x2 = Fxp(input[2*i+1], True, 16, fractbit)
        bdata2 = int(x2.hex(), 16).to_bytes(2, byteorder='big')
        bdata = bdata2 + bdata1
        idata = int.from_bytes(bdata,'big')
        output.append(idata)



print("setting weight/bias data")
ddr0 = []
ddr1 = []
ddr2 = []
ddr3 = []

read_c1weight("./resnet18_param/conv1.weight.txt",
			  "./resnet18_param/conv1.bias.txt", ddr0)

read_weight(  "./resnet18_param/layer1.0.conv1.weight.txt",
			  "./resnet18_param/layer1.0.conv1.bias.txt", ddr1, 64, 64, 3)

read_weight(  "./resnet18_param/layer1.0.conv2.weight.txt",
			  "./resnet18_param/layer1.0.conv2.bias.txt", ddr1, 64, 64, 3)

read_weight(  "./resnet18_param/layer1.1.conv1.weight.txt",
			  "./resnet18_param/layer1.1.conv1.bias.txt", ddr1, 64, 64, 3)

read_weight(  "./resnet18_param/layer1.1.conv2.weight.txt",
			  "./resnet18_param/layer1.1.conv2.bias.txt", ddr1, 64, 64, 3)

read_weight(  "./resnet18_param/layer2.0.downsample.0.weight.txt",
			  "./resnet18_param/layer2.0.downsample.0.bias.txt", ddr1, 128, 64, 1)

read_weight(  "./resnet18_param/layer2.0.conv1.weight.txt",
			  "./resnet18_param/layer2.0.conv1.bias.txt", ddr1, 128, 64, 3)

read_weight(  "./resnet18_param/layer2.0.conv2.weight.txt",
			  "./resnet18_param/layer2.0.conv2.bias.txt", ddr1, 128, 128, 3)

read_weight(  "./resnet18_param/layer2.1.conv1.weight.txt",
			  "./resnet18_param/layer2.1.conv1.bias.txt", ddr1, 128, 128, 3)

read_weight(  "./resnet18_param/layer2.1.conv2.weight.txt",
			  "./resnet18_param/layer2.1.conv2.bias.txt", ddr1, 128, 128, 3)

read_weight(  "./resnet18_param/layer3.0.downsample.0.weight.txt",
			  "./resnet18_param/layer3.0.downsample.0.bias.txt", ddr2, 256, 128, 1)

read_weight(  "./resnet18_param/layer3.0.conv1.weight.txt",
			  "./resnet18_param/layer3.0.conv1.bias.txt", ddr2, 256, 128, 3)

read_weight(  "./resnet18_param/layer3.0.conv2.weight.txt",
			  "./resnet18_param/layer3.0.conv2.bias.txt", ddr2, 256, 256, 3)

read_weight(  "./resnet18_param/layer3.1.conv1.weight.txt",
			  "./resnet18_param/layer3.1.conv1.bias.txt", ddr2, 256, 256, 3)

read_weight(  "./resnet18_param/layer3.1.conv2.weight.txt",
			  "./resnet18_param/layer3.1.conv2.bias.txt", ddr2, 256, 256, 3)

read_weight(  "./resnet18_param/layer4.0.downsample.0.weight.txt",
			  "./resnet18_param/layer4.0.downsample.0.bias.txt", ddr2, 512, 256, 1)

read_weight(  "./resnet18_param/layer4.0.conv1.weight.txt",
			  "./resnet18_param/layer4.0.conv1.bias.txt", ddr2, 512, 256, 3)

read_weight(  "./resnet18_param/layer4.0.conv2.weight.txt",
			  "./resnet18_param/layer4.0.conv2.bias.txt", ddr2, 512, 512, 3)

read_weight(  "./resnet18_param/layer4.1.conv1.weight.txt",
			  "./resnet18_param/layer4.1.conv1.bias.txt", ddr3, 512, 512, 3)

read_weight(  "./resnet18_param/layer4.1.conv2.weight.txt",
			  "./resnet18_param/layer4.1.conv2.bias.txt", ddr3, 512, 512, 3)

read_fcweight("./resnet18_param/fc.weight.txt", "./resnet18_param/fc.bias.txt", ddr3)

print('DDR SIZE')
print('ddr0:  ' + str(len(ddr0)))
print('ddr1:  ' + str(len(ddr1)))
print('ddr2:  ' + str(len(ddr2)))
print('ddr3:  ' + str(len(ddr3)))
print('total: ' + str( len(ddr0) + len(ddr1) + len(ddr2) + len(ddr3) ))

print('DDR_MAXVAL')
print( 'ddr0: ' + str( max( abs(max(ddr0)), abs(min(ddr0)) ) ) )
print( 'ddr1: ' + str( max( abs(max(ddr1)), abs(min(ddr1)) ) ) )
print( 'ddr2: ' + str( max( abs(max(ddr2)), abs(min(ddr2)) ) ) )
print( 'ddr3: ' + str( max( abs(max(ddr3)), abs(min(ddr3)) ) ) )


fxp_ddr0 = []
fxp_ddr1 = []
fxp_ddr2 = []
fxp_ddr3 = []
#to_fxp_array(ddr0, fxp_ddr0, 13)
print('finish 0')
to_fxp_array(ddr1, fxp_ddr1, 13)
print('finish 1')
to_fxp_array(ddr2, fxp_ddr2, 13)
print('finish 2')
#to_fxp_array(ddr3, fxp_ddr3, 13)
print('finish 3')


print("save ddr")
np.savetxt('./resnet18_param/ddr0.txt', np.array(fxp_ddr0, dtype='uint32'))
np.savetxt('./resnet18_param/ddr1.txt', np.array(fxp_ddr1, dtype='uint32'))
np.savetxt('./resnet18_param/ddr2.txt', np.array(fxp_ddr2, dtype='uint32'))
np.savetxt('./resnet18_param/ddr3.txt', np.array(fxp_ddr3, dtype='uint32'))

