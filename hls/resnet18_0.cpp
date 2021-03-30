#include <stdio.h>
#include <stdlib.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <string.h>

#define C1_K 7
#define P1_K 3

#define C1_ICH 3
#define C1_OCH 64
#define P1_OCH 64

#define C1_ISIZE 224
#define C1_OSIZE 112
#define P1_OSIZE 56

#define C1_S 2
#define P1_S 2

#define C1_P 3
#define P1_P 1

#define DDRSIZE (C1_K*C1_K*C1_OCH*C1_ICH + C1_OCH)


typedef ap_uint<256> uint256;
typedef ap_uint<169> uint169;
typedef ap_uint<4> uint4;
typedef ap_uint<128> uint128;
typedef volatile ap_uint<128> ddr_t;
typedef ap_fixed<16, 6> fm_t;
typedef ap_fixed<16, 3> weight_t;
typedef ap_fixed<32, 6> buf_t;


void conv1(
    fm_t input[C1_ISIZE][C1_ISIZE][C1_ICH],
	buf_t output[C1_OSIZE][C1_OSIZE][C1_OCH],
	ddr_t ddr[]
){

    weight_t bias[C1_OCH];

    for(int i=0; i<C1_OCH/8; i++){
#pragma HLS PIPELINE
		uint128 tmp;
		tmp = ddr[C1_OCH*C1_ICH*C1_K*C1_K/8 + i];
		bias[i*8    ].range(15, 0) = tmp.range( 15,   0);
		bias[i*8 + 1].range(15, 0) = tmp.range( 31,  16);
		bias[i*8 + 2].range(15, 0) = tmp.range( 47,  32);
		bias[i*8 + 3].range(15, 0) = tmp.range( 63,  48);
		bias[i*8 + 4].range(15, 0) = tmp.range( 79,  64);
		bias[i*8 + 5].range(15, 0) = tmp.range( 95,  80);
		bias[i*8 + 6].range(15, 0) = tmp.range(111,  96);
		bias[i*8 + 7].range(15, 0) = tmp.range(127, 112);
    }

	for(int r=0; r<C1_OSIZE; r++){
		for(int c=0; c<C1_OSIZE; c++){
#pragma HLS PIPELINE
			for(int ch=0; ch<C1_OCH; ch++){
				output[r][c][ch] = bias[ch];
			}
		}
	}

	for(int i=0; i<C1_K; i++){
		for(int j=0; j<C1_K; j++){

			weight_t weight[C1_OCH*C1_ICH];
#pragma HLS array_partition variable=weight complete dim=0

			for(int w=0; w<C1_OCH*C1_ICH/8; w++){
#pragma HLS UNROLL
				uint128 tmp;
				tmp = ddr[(i*C1_K+j)*C1_OCH*C1_ICH/8 + w];
				weight[w*8    ].range(15, 0) = tmp.range( 15,   0);
				weight[w*8 + 1].range(15, 0) = tmp.range( 31,  16);
				weight[w*8 + 2].range(15, 0) = tmp.range( 47,  32);
				weight[w*8 + 3].range(15, 0) = tmp.range( 63,  48);
				weight[w*8 + 4].range(15, 0) = tmp.range( 79,  64);
				weight[w*8 + 5].range(15, 0) = tmp.range( 95,  80);
				weight[w*8 + 6].range(15, 0) = tmp.range(111,  96);
				weight[w*8 + 7].range(15, 0) = tmp.range(127, 112);
			}

			for(int row=0; row<C1_OSIZE; row++){
				for(int col=0; col<C1_OSIZE; col++){

#pragma HLS PIPELINE
    				if(C1_S*row+i-C1_P>=0 && C1_S*row+i-C1_P<C1_ISIZE && C1_S*col+j-C1_P>=0 && C1_S*col+j-C1_P<C1_ISIZE){
    					for(int too=0; too<C1_OCH; too++){
#pragma HLS UNROLL
#pragma HLS dependence variable=output inter false
    						for(int tii=0; tii<C1_ICH; tii++){
#pragma HLS UNROLL
    							output[row][col][too] += input[C1_S*row+i-C1_P][C1_S*col+j-C1_P][tii] *
													 weight[too*C1_ICH + tii];
    						}
    					}
    				}
    			}
    		}
    	}
    }

    return;
}

void maxpool(buf_t input[C1_OSIZE][C1_OSIZE][C1_OCH], fm_t output[P1_OSIZE][P1_OSIZE][C1_OCH]){

	maxpool:for(int row=0; row<P1_OSIZE; row++){
		for(int col=0; col<P1_OSIZE; col++){

			buf_t tmp_max[C1_OCH];
#pragma HLS array_partition variable=tmp_max complete

			for(int co=0; co<C1_OCH; co++){
#pragma HLS UNROLL
				tmp_max[co] = 0.0;
			}

			mp_compute:for(int i=0; i<P1_K; i++){
				for(int j=0; j<P1_K; j++){
#pragma HLS PIPELINE
					if(P1_S*row+i-P1_P>=0 && P1_S*row+i-P1_P<C1_OSIZE && P1_S*col+j-P1_P>=0 && P1_S*col+j-P1_P<C1_OSIZE){
						for(int ch=0; ch<C1_OCH; ch++){
#pragma HLS UNROLL
							if(tmp_max[ch] < input[P1_S*row+i-P1_P][P1_S*col+j-P1_P][ch]){
								tmp_max[ch] = input[P1_S*row+i-P1_P][P1_S*col+j-P1_P][ch];
							}
						}
					}
				}
			}

			for(int so=0; so<C1_OCH; so++){
#pragma HLS UNROLL
				output[row][col][so] = tmp_max[so];
			}

		}
	}

	return;
}



void resnet18_0(
		//float input[C1_ISIZE*C1_ISIZE*C1_ICH],
		volatile uint256 input[],
		volatile uint169 sw0out[],
		uint4 startt[1],
	    uint4 stopt[1],
		ddr_t ddr[]
){
//#pragma HLS INTERFACE s_axilite port=input
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis port=startt
#pragma HLS INTERFACE axis port=stopt
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=sw0out
#pragma HLS INTERFACE m_axi depth=2048 port=ddr

	static fm_t inbuf[C1_ISIZE][C1_ISIZE][C1_ICH];
	static buf_t c1out[C1_OSIZE][C1_OSIZE][C1_OCH];
	static fm_t p1out[P1_OSIZE][P1_OSIZE][C1_OCH];

#pragma HLS array_reshape variable=inbuf complete dim=3
#pragma HLS array_reshape variable=c1out complete dim=3
#pragma HLS array_reshape variable=p1out complete dim=3

#pragma HLS RESOURCE variable=inbuf core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=c1out core=RAM_2P_URAM
#pragma HLS RESOURCE variable=p1out core=RAM_2P_BRAM

	startt[0] = 1;

	/*for(int i=0; i<C1_ISIZE; i++){
		for(int j=0; j<C1_ISIZE; j++){
			for(int ch=0; ch<C1_ICH; ch++){
#pragma HLS PIPELINE
				inbuf[i][j][ch] = (fm_t)input[i*C1_ISIZE*C1_ICH + j*C1_ICH + ch];
			}
		}
	}*/

	for(int i=0; i<C1_ISIZE*C1_ISIZE*C1_ICH/8; i++){
#pragma HLS PIPELINE
		uint256 tmp = input[i];
		for(int j=0; j<8; j++){
#pragma HLS UNROLL
			union {int i; float f;} c;
			c.i = tmp.range(32*j + 31, 32*j);
			inbuf[(i*8 + j)/(C1_ISIZE*C1_ICH)][((i*8 + j)/C1_ICH)%C1_ISIZE][(i*8 + j)%C1_ICH] = c.f;
		}
	}
	ddr[10000] = (uint128)inbuf[223][223][2].range(32, 0);

	conv1(inbuf, c1out, ddr);
	maxpool(c1out, p1out);

	for(int x=0; x<P1_OSIZE; x++){
		for(int y=0; y<P1_OSIZE; y++){
			for(int ch=0; ch<C1_OCH/8; ch++){
#pragma HLS PIPELINE
				uint169 tmp;
				tmp.range( 15,   0) = p1out[x][y][ch*8    ].range(15, 0);
				tmp.range( 31,  16) = p1out[x][y][ch*8 + 1].range(15, 0);
				tmp.range( 47,  32) = p1out[x][y][ch*8 + 2].range(15, 0);
				tmp.range( 63,  48) = p1out[x][y][ch*8 + 3].range(15, 0);
				tmp.range( 79,  64) = p1out[x][y][ch*8 + 4].range(15, 0);
				tmp.range( 95,  80) = p1out[x][y][ch*8 + 5].range(15, 0);
				tmp.range(111,  96) = p1out[x][y][ch*8 + 6].range(15, 0);
				tmp.range(127, 112) = p1out[x][y][ch*8 + 7].range(15, 0);
				sw0out[(x*P1_OSIZE + y)*C1_OCH/8 + ch] = tmp;
			}
		}
	}

	stopt[0] = 1;

	return;
}


