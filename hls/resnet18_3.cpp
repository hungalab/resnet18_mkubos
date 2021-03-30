#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <string.h>

#define T 32
#define BUFSIZE (7*7*512/T)
#define K_N 3
#define K_D 1
#define TO_MAX (512/T)

#define L4_OSIZE 7

#define L4_TO (512/T)

#define L41_C1_TI (512/T)
#define L41_C2_TI (512/T)

#define L41_C1_S 1
#define L41_C2_S 1

#define L41_C1_P 1
#define L41_C2_P 1

#define L41_C1_RELU 1
#define L41_C2_RELU 0

#define L41_C1_OFFSET 0
#define L41_C2_OFFSET (L41_C1_OFFSET + L4_TO*L41_C1_TI*T*T*K_N*K_N + L4_TO*T)

#define FC_W_OFFSET (L41_C2_OFFSET + L4_TO*L41_C2_TI*T*T*K_N*K_N + L4_TO*T)
#define FC_B_OFFSET (FC_W_OFFSET + 512*1000)

#define DDRSIZE     (FC_B_OFFSET + 1000)


typedef ap_uint<169> uint169;
typedef ap_uint<4> uint4;
typedef ap_uint<128> uint128;
typedef volatile ap_uint<128> ddr_t;
typedef ap_fixed<16, 6> fm_t;
typedef ap_fixed<16, 3> weight_t;
typedef ap_fixed<32, 6> buf_t;


/*void load_uint169_fix16(volatile uint169 sw0in[], fm_t inbuf[BUFSIZE][T]){

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
}*/

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
		for(int col=0; col<OSIZE; col++){
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


void average_pool(fm_t input[BUFSIZE][T], fm_t output[512]){

	float buf[512];
#pragma HLS array_partition variable=buf complete

	for(int co=0; co<512; co++){
#pragma HLS UNROLL
		buf[co] = 0.0;
	}

	for(int row=0; row<L4_OSIZE; row++){
		for(int col=0; col<L4_OSIZE; col++){
			for(int to=0; to<512/T; to++){
#pragma HLS PIPELINE II=5
				for(int too=0; too<T; too++){
					float intmp;
					intmp = (float)input[row*L4_OSIZE*512/T + col*512/T + to][too];
					buf[to*T + too] += intmp;
				}
			}
		}
	}

	for(int so=0; so<512; so++){
#pragma HLS PIPELINE
		float tmp;
		tmp = buf[so] / 49.0;
		output[so] = (fm_t)tmp;
	}

	return;
}


void fc(fm_t input[512], buf_t outbuf[1000], ddr_t ddr[]){

	weight_t bias[1000];
#pragma HLS array_reshape variable=bias complete

	for(int i=0; i<1000/8; i++){
#pragma HLS UNROLL
		uint128 tmp;
		tmp = ddr[(FC_B_OFFSET)/8 + i];
		bias[i*8    ].range(15, 0) = tmp.range( 15,   0);
		bias[i*8 + 1].range(15, 0) = tmp.range( 31,  16);
		bias[i*8 + 2].range(15, 0) = tmp.range( 47,  32);
		bias[i*8 + 3].range(15, 0) = tmp.range( 63,  48);
		bias[i*8 + 4].range(15, 0) = tmp.range( 79,  64);
		bias[i*8 + 5].range(15, 0) = tmp.range( 95,  80);
		bias[i*8 + 6].range(15, 0) = tmp.range(111,  96);
		bias[i*8 + 7].range(15, 0) = tmp.range(127, 112);
	}


	for(int och=0; och<1000; och++){

		weight_t weight[512];
#pragma HLS array_partition variable=weight complete

		for(int i=0; i<512/8; i++){
#pragma HLS UNROLL
			uint128 tmp;
			tmp = ddr[(FC_W_OFFSET + 512*och)/8 + i];
			weight[i*8    ].range(15, 0) = tmp.range( 15,   0);
			weight[i*8 + 1].range(15, 0) = tmp.range( 31,  16);
			weight[i*8 + 2].range(15, 0) = tmp.range( 47,  32);
			weight[i*8 + 3].range(15, 0) = tmp.range( 63,  48);
			weight[i*8 + 4].range(15, 0) = tmp.range( 79,  64);
			weight[i*8 + 5].range(15, 0) = tmp.range( 95,  80);
			weight[i*8 + 6].range(15, 0) = tmp.range(111,  96);
			weight[i*8 + 7].range(15, 0) = tmp.range(127, 112);
		}

		outbuf[och] = bias[och];
		for(int ich=0; ich<512; ich++){
#pragma HLS UNROLL
			outbuf[och] += input[ich] * weight[ich];
		}

	}

	return;
}

void resnet18_3(
		volatile uint169 sw0in[],
		float output[1000],
		ddr_t ddr[],
		ap_uint<4> startt[1],
		ap_uint<4> stopt[1]
){
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis port=sw0in
#pragma HLS INTERFACE axis port=startt
#pragma HLS INTERFACE axis port=stopt
#pragma HLS INTERFACE m_axi port=ddr depth=2048

	static fm_t buf0[BUFSIZE][T];
	static fm_t buf1[BUFSIZE][T];
	static fm_t buf2[BUFSIZE][T];
	static fm_t fcin[512];
	static buf_t outbuf[1000];

#pragma HLS array_reshape variable=buf0    complete dim=2
#pragma HLS array_reshape variable=buf1    complete dim=2
#pragma HLS array_reshape variable=buf2    complete dim=2
#pragma HLS array_reshape variable=fcin    complete

#pragma HLS RESOURCE variable=buf0    core=RAM_2P_URAM
#pragma HLS RESOURCE variable=buf1    core=RAM_2P_URAM
#pragma HLS RESOURCE variable=buf2    core=RAM_2P_URAM
#pragma HLS RESOURCE variable=outbuf  core=RAM_2P_URAM


	//load_uint169_fix16(sw0in, buf2);

	for(int l=0; l<BUFSIZE; l++){
		for(int i=0; i<T/8; i++){
#pragma HLS PIPELINE
		   uint169 tmp;
		   tmp = sw0in[l*T/8 + i];
		   for (int j=0; j<8; j++){
			   buf2[l][i*8 + j].range(15, 0) = tmp.range(j*16+15, j*16);
		   }
		}
	}

	startt[0] = (char)buf2[BUFSIZE-1][T-1];

	conv_layer(buf2, buf0, ddr, L4_OSIZE, L4_TO, L41_C1_TI, K_N, L41_C1_S, L41_C1_P, L41_C1_RELU, L41_C1_OFFSET); //L41_CONV_1
    conv_layer(buf0, buf1, ddr, L4_OSIZE, L4_TO, L41_C2_TI, K_N, L41_C2_S, L41_C2_P, L41_C2_RELU, L41_C2_OFFSET); //L41_CONV_2
    add(buf2, buf1, buf0, L4_OSIZE, L4_TO);

	average_pool(buf0, fcin);
	fc(fcin, outbuf, ddr);

	for(int o=0; o<1000; o++){
#pragma HLS PIPELINE
		output[o] = (float)outbuf[o];
	}

	stopt[0] = (char)outbuf[999];

	return;
}

