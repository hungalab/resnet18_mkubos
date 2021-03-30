#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <string.h>

#define T 32
#define BUFSIZE (56*56*64/T)
#define K_N 3
#define K_D 1
#define TO_MAX (128/T)

#define L1_OSIZE 56
#define L2_OSIZE 28

#define L1_TO ( 64/T)
#define L2_TO (128/T)

#define L10_C1_TI ( 64/T)
#define L10_C2_TI ( 64/T)
#define L11_C1_TI ( 64/T)
#define L11_C2_TI ( 64/T)
#define L20_Cd_TI ( 64/T)
#define L20_C1_TI ( 64/T)
#define L20_C2_TI (128/T)
#define L21_C1_TI (128/T)
#define L21_C2_TI (128/T)

#define L10_C1_S 1
#define L10_C2_S 1
#define L11_C1_S 1
#define L11_C2_S 1
#define L20_Cd_S 2
#define L20_C1_S 2
#define L20_C2_S 1
#define L21_C1_S 1
#define L21_C2_S 1

#define L10_C1_P 1
#define L10_C2_P 1
#define L11_C1_P 1
#define L11_C2_P 1
#define L20_Cd_P 0
#define L20_C1_P 1
#define L20_C2_P 1
#define L21_C1_P 1
#define L21_C2_P 1

#define L10_C1_RELU 1
#define L10_C2_RELU 0
#define L11_C1_RELU 1
#define L11_C2_RELU 0
#define L20_Cd_RELU 0
#define L20_C1_RELU 1
#define L20_C2_RELU 0
#define L21_C1_RELU 1
#define L21_C2_RELU 0

#define L10_C1_OFFSET 0
#define L10_C2_OFFSET (L10_C1_OFFSET + L1_TO*L10_C1_TI*T*T*K_N*K_N + L1_TO*T)
#define L11_C1_OFFSET (L10_C2_OFFSET + L1_TO*L10_C2_TI*T*T*K_N*K_N + L1_TO*T)
#define L11_C2_OFFSET (L11_C1_OFFSET + L1_TO*L11_C1_TI*T*T*K_N*K_N + L1_TO*T)

#define L20_Cd_OFFSET (L11_C2_OFFSET + L1_TO*L11_C2_TI*T*T*K_N*K_N + L1_TO*T)
#define L20_C1_OFFSET (L20_Cd_OFFSET + L2_TO*L20_Cd_TI*T*T*K_D*K_D + L2_TO*T)
#define L20_C2_OFFSET (L20_C1_OFFSET + L2_TO*L20_C1_TI*T*T*K_N*K_N + L2_TO*T)
#define L21_C1_OFFSET (L20_C2_OFFSET + L2_TO*L20_C2_TI*T*T*K_N*K_N + L2_TO*T)
#define L21_C2_OFFSET (L21_C1_OFFSET + L2_TO*L21_C1_TI*T*T*K_N*K_N + L2_TO*T)

#define DDRSIZE       (L21_C2_OFFSET + L2_TO*L21_C2_TI*T*T*K_N*K_N + L2_TO*T)


typedef ap_uint<169> uint169;
typedef ap_uint<4> uint4;
typedef ap_uint<128> uint128;
typedef volatile ap_uint<128> ddr_t;
typedef ap_fixed<16, 6> fm_t;
typedef ap_fixed<16, 3> weight_t;
typedef ap_fixed<32, 6> buf_t;


void load_uint169_fix16(volatile uint169 sw0in[], fm_t inbuf[BUFSIZE][T]){

	for(int l=0; l<BUFSIZE; l++){
		for(int i=0; i<T/8; i++){
#pragma HLS PIPELINE
		   uint169 tmp;
		   tmp = sw0in[l*T/8 + i];
		   for (int j=0; j<8; j++){
			   inbuf[l][i*8 + j].range(15, 0) = tmp.range(j*16+15, j*16);
		   }
		}
	}

	return;
}

void load_bias(weight_t bias[TO_MAX][T], ddr_t ddr[], int TO, int offset){
#pragma HLS INLINE off

	for(int i=0; i<TO; i++){
#pragma HLS PIPELINE
		for(int j=0; j<T/8; j++){
#pragma HLS UNROLL
			uint128 tmp;
			tmp = ddr[(offset + i*T)/8 + j];
			bias[i][j*8    ].range(15, 0) = tmp.range( 15,   0);
			bias[i][j*8 + 1].range(15, 0) = tmp.range( 31,  16);
			bias[i][j*8 + 2].range(15, 0) = tmp.range( 47,  32);
			bias[i][j*8 + 3].range(15, 0) = tmp.range( 63,  48);
			bias[i][j*8 + 4].range(15, 0) = tmp.range( 79,  64);
			bias[i][j*8 + 5].range(15, 0) = tmp.range( 95,  80);
			bias[i][j*8 + 6].range(15, 0) = tmp.range(111,  96);
			bias[i][j*8 + 7].range(15, 0) = tmp.range(127, 112);
		}
	}

	return;
}

void load_weight(weight_t weight[T*T], ddr_t ddr[], int offset){
#pragma HLS INLINE off

	for(int i=0; i<T*T/8; i++){
#pragma HLS UNROLL
		uint128 tmp;
		tmp = ddr[offset/8 + i];
		weight[i*8    ].range(15, 0) = tmp.range( 15,   0);
		weight[i*8 + 1].range(15, 0) = tmp.range( 31,  16);
		weight[i*8 + 2].range(15, 0) = tmp.range( 47,  32);
		weight[i*8 + 3].range(15, 0) = tmp.range( 63,  48);
		weight[i*8 + 4].range(15, 0) = tmp.range( 79,  64);
		weight[i*8 + 5].range(15, 0) = tmp.range( 95,  80);
		weight[i*8 + 6].range(15, 0) = tmp.range(111,  96);
		weight[i*8 + 7].range(15, 0) = tmp.range(127, 112);
	}

	return;
}

void compute(weight_t weight[T*T], fm_t input[BUFSIZE][T], buf_t outbuf[BUFSIZE][T],
		int OSIZE, int TO, int TI, int K,  int S, int P, int to, int ti, int i, int j){
#pragma HLS INLINE off

	Compute:for(int row=0; row<OSIZE; row++){
#pragma HLS loop_tripcount min=14 max=56
		for(int col=0; col<OSIZE; col++){
#pragma HLS loop_tripcount min=14 max=56
#pragma HLS PIPELINE
			if(S*row+i-P>=0 && S*row+i-P<S*OSIZE && S*col+j-P>=0 && S*col+j-P<S*OSIZE){
				fm_t tmp_in[T];
				for(int li=0; li<T; li++){
#pragma HLS UNROLL
					tmp_in[li] = input[(S*row+i-P)*S*OSIZE*TI + (S*col+j-P)*TI + ti][li];
				}
				for(int too=0; too<T; too++){
#pragma HLS UNROLL
#pragma HLS dependence variable=output inter false
					for(int tii=0; tii<T; tii++){
#pragma HLS UNROLL
						outbuf[row*OSIZE*TO + col*TO + to][too] +=
								tmp_in[tii] * weight[too*T + tii];
					}
				}
			}
		}
	}

	return;
}

void conv_layer(fm_t input[BUFSIZE][T], fm_t output[BUFSIZE][T], ddr_t ddr[],
		int OSIZE, int TO, int TI, int K, int S, int P, int RELU, int OFFSET
){

	static buf_t outbuf[BUFSIZE][T];
    weight_t bias[TO_MAX][T];
#pragma HLS array_reshape variable=outbuf complete dim=2
#pragma HLS array_reshape variable=bias   complete dim=2
#pragma HLS RESOURCE variable=outbuf core=RAM_2P_URAM
#pragma HLS RESOURCE variable=bias   core=RAM_1P_BRAM

    load_bias(bias, ddr, TO, OFFSET + TO*TI*T*T*K*K);

	SetOutput:for(int row=0; row<OSIZE; row++){
        for(int col=0; col<OSIZE; col++){
            for(int to=0; to<TO; to++){
#pragma HLS PIPELINE
                for(int too=0; too<T; too++){
#pragma HLS UNROLL
                    outbuf[row*OSIZE*TO + col*TO + to][too] = bias[to][too];
                }
            }
        }
	}

    Convolution:for(int l=0; l<TO*TI*K*K; l++){
#pragma HLS DATAFLOW
#pragma HLS stable variable=input
#pragma HLS stable variable=outbuf
#pragma HLS loop_tripcount min=8 max=144

    	int to = l / (TI*K*K);
    	int ti = (l/(K*K)) % TI;
    	int i = (l/K) % K;
    	int j = l % K;
    	weight_t weight[T*T];
#pragma HLS array_partition variable=weight complete dim=0

    	load_weight(weight, ddr, OFFSET + l*T*T);
    	compute(weight, input, outbuf, OSIZE, TO, TI, K, S, P, to, ti, i, j);

    }

	StoreOutput:for(int row=0; row<OSIZE; row++){
        for(int col=0; col<OSIZE; col++){
            for(int to=0; to<TO; to++){
#pragma HLS PIPELINE
                for(int too=0; too<T; too++){
#pragma HLS UNROLL
                    fm_t tmp, relud;
                    tmp = outbuf[row*OSIZE*TO + col*TO + to][too];
                    if(RELU == 1 && tmp < 0.0){
                        relud = 0.0;
                    }else{
                        relud = tmp;
                    }
                    output[row*OSIZE*TO + col*TO + to][too] = relud;
                }
            }
        }
	}

	return;
}



void add(
	    fm_t input1[BUFSIZE][T], fm_t input2[BUFSIZE][T], fm_t output[BUFSIZE][T],
		int OSIZE, int TO
){

    Add:for(int row=0; row<OSIZE; row++){
    	for(int col=0; col<OSIZE; col++){
    		for(int to=0; to<TO; to++){
#pragma HLS PIPELINE
    			for(int too=0; too<T; too++){
#pragma HLS UNROLL
    				fm_t add_result;
    				fm_t relu_result;
					add_result = input1[row*OSIZE*TO + col*TO + to][too] +
								 input2[row*OSIZE*TO + col*TO + to][too];
					if(add_result < 0.0){
						relu_result = 0.0;
					}else{
						relu_result = add_result;
					}
					output[row*OSIZE*TO + col*TO + to][too] = relu_result;
    			}
    		}
    	}
    }

	return;
}


void resnet18_1(
		volatile uint169 sw0in[],
		volatile uint169 sw0out[],
		ddr_t ddr[],
		ap_uint<4> startt[1],
		ap_uint<4> stopt[1]
){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis port=sw0in
#pragma HLS INTERFACE axis port=sw0out
#pragma HLS INTERFACE axis port=startt
#pragma HLS INTERFACE axis port=stopt
#pragma HLS INTERFACE m_axi port=ddr depth=2048

	static fm_t buf0[BUFSIZE][T];
	static fm_t buf1[BUFSIZE][T];
	static fm_t buf2[BUFSIZE][T];

#pragma HLS array_reshape variable=buf0    complete dim=2
#pragma HLS array_reshape variable=buf1    complete dim=2
#pragma HLS array_reshape variable=buf2    complete dim=2

#pragma HLS RESOURCE variable=buf0    core=RAM_2P_URAM
#pragma HLS RESOURCE variable=buf1    core=RAM_2P_URAM
#pragma HLS RESOURCE variable=buf2    core=RAM_2P_URAM


	load_uint169_fix16(sw0in, buf0);

	startt[0] = (char)buf0[BUFSIZE-1][T-1];

	conv_layer(buf0, buf1, ddr, L1_OSIZE, L1_TO, L10_C1_TI, K_N, L10_C1_S, L10_C1_P, L10_C1_RELU, L10_C1_OFFSET); //L10_CONV_1
    conv_layer(buf1, buf2, ddr, L1_OSIZE, L1_TO, L10_C2_TI, K_N, L10_C2_S, L10_C2_P, L10_C2_RELU, L10_C2_OFFSET); //L10_CONV_2
    add(buf0, buf2, buf1, L1_OSIZE, L1_TO);

	conv_layer(buf1, buf0, ddr, L1_OSIZE, L1_TO, L11_C1_TI, K_N, L11_C1_S, L11_C1_P, L11_C1_RELU, L11_C1_OFFSET); //L11_CONV_1
    conv_layer(buf0, buf2, ddr, L1_OSIZE, L1_TO, L11_C2_TI, K_N, L11_C2_S, L11_C2_P, L11_C2_RELU, L11_C2_OFFSET); //L11_CONV_2
    add(buf1, buf2, buf0, L1_OSIZE, L1_TO);

	conv_layer(buf0, buf1, ddr, L2_OSIZE, L2_TO, L20_Cd_TI, K_D, L20_Cd_S, L20_Cd_P, L20_Cd_RELU, L20_Cd_OFFSET); //L20_CONV_d
    conv_layer(buf0, buf2, ddr, L2_OSIZE, L2_TO, L20_C1_TI, K_N, L20_C1_S, L20_C1_P, L20_C1_RELU, L20_C1_OFFSET); //L20_CONV_1
    conv_layer(buf2, buf0, ddr, L2_OSIZE, L2_TO, L20_C2_TI, K_N, L20_C2_S, L20_C2_P, L20_C2_RELU, L20_C2_OFFSET); //L20_CONV_2
    add(buf1, buf0, buf2, L2_OSIZE, L2_TO);

	conv_layer(buf2, buf0, ddr, L2_OSIZE, L2_TO, L21_C1_TI, K_N, L21_C1_S, L21_C1_P, L21_C1_RELU, L21_C1_OFFSET); //L21_CONV_1
    conv_layer(buf0, buf1, ddr, L2_OSIZE, L2_TO, L21_C2_TI, K_N, L21_C2_S, L21_C2_P, L21_C2_RELU, L21_C2_OFFSET); //L21_CONV_2
    add(buf2, buf1, buf0, L2_OSIZE, L2_TO);

	for(int x=0; x<L2_OSIZE*L2_OSIZE*L2_TO; x++){
		for(int ch=0; ch<T/8; ch++){
#pragma HLS PIPELINE
				uint169 tmp;
				tmp.range( 15,   0) = buf0[x][ch*8    ].range(15, 0);
				tmp.range( 31,  16) = buf0[x][ch*8 + 1].range(15, 0);
				tmp.range( 47,  32) = buf0[x][ch*8 + 2].range(15, 0);
				tmp.range( 63,  48) = buf0[x][ch*8 + 3].range(15, 0);
				tmp.range( 79,  64) = buf0[x][ch*8 + 4].range(15, 0);
				tmp.range( 95,  80) = buf0[x][ch*8 + 5].range(15, 0);
				tmp.range(111,  96) = buf0[x][ch*8 + 6].range(15, 0);
				tmp.range(127, 112) = buf0[x][ch*8 + 7].range(15, 0);
				sw0out[x*T/8 + ch] = tmp;
		}
	}

	stopt[0] = (char)buf0[L2_OSIZE*L2_OSIZE*L2_TO-1][T-1];

	return;
}

