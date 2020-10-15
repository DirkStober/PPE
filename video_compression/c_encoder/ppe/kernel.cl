#include "test_setup.h"

/* Basic Kernel*/

kernel void convertRGBtoYCbCr(global float *in_rc, global float *in_gc, global float*in_bc,global float * out_y, global float * out_cb, global float *out_cr) {



	for (int thread_id = get_global_id(0); thread_id < SIZE_FRAME; thread_id += get_global_size(0)) {
		float R = in_rc[thread_id];
		float G = in_gc[thread_id];
		float B = in_bc[thread_id];
		float Y = 0 + ((float)0.299*R) + ((float)0.587*G) + ((float)0.113*B);
		float Cb = 128 - ((float)0.168736*R) - ((float)0.331264*G) + ((float)0.5*B);
		float Cr = 128 + ((float)0.5*R) - ((float)0.418688*G) - ((float)0.081312*B);
		out_y[thread_id] = Y;
		out_cb[thread_id] = Cb;
		out_cr[thread_id] = Cr;
	}
}
