#include "test_setup.h"

/* Basic Kernel*/

kernel void convertRGBtoYCbCr(global float *in_rc, global float *in_gc, global float*in_bc, global float * out_y, global float * out_cb, global float *out_cr) {


	int tid = get_global_id(0);
	int size = get_global_size(0);
	for (int thread_id = tid; thread_id < 2048 * 2048; thread_id += size) {
		float R = in_rc[thread_id];
		float G = in_gc[thread_id];
		float B = in_bc[thread_id];
		float Y = 0 + ((float)0.299*R) + ((float)0.587*G) + ((float)0.113*B);
		float Cb =  128 - ((float)0.168736*R) - ((float)0.331264*G) + ((float)0.5*B);
		float Cr = 128 + ((float)0.5*R) - ((float)0.418688*G) - ((float)0.081312*B);
		out_y[thread_id] = Y;
		out_cb[thread_id] = Cb;
		out_cr[thread_id] = Cr;
	}
}



kernel void first_sweep(global float * in_cb, global float * in_cr, global float * out_cb, global float * out_cr) {
	
	for (int t = get_global_id(0); t < 2048; t += get_global_size(0)) {
		out_cb[t] = in_cb[t];
		out_cr[t] = in_cr[t];
		out_cb[SIZE_ROW*(SIZE_ROW-1) + t ] = in_cb[SIZE_ROW*(SIZE_ROW - 1) + t];
		out_cr[SIZE_ROW*(SIZE_ROW-1) + t] = in_cr[SIZE_ROW*(SIZE_ROW - 1) + t];
		float a = 0.25;
		float b = 0.5;
		float c = 0.25;
		if ((t != 0) && (t != (SIZE_ROW - 1))) {
			for (int y = 1; y < SIZE_ROW - 1; y++) {
				out_cb[y*SIZE_ROW + t] = a * in_cb[(y - 1)*SIZE_ROW + t] + b * in_cb[y*SIZE_ROW + t] + c * in_cb[(y + 1)*SIZE_ROW + t];
				out_cr[y*SIZE_ROW + t] = a * in_cr[(y - 1)*SIZE_ROW + t] + b * in_cr[y*SIZE_ROW + t] + c * in_cr[(y + 1)*SIZE_ROW + t];
			}
		}
		else {
			for (int y = 1; y < SIZE_ROW - 1; y++) {
				out_cb[y*SIZE_ROW + t] = in_cb[y*SIZE_ROW + t];
				out_cr[y*SIZE_ROW + t] = in_cr[y*SIZE_ROW + t];
			}
		}
	}
}


kernel void second_sweep(global float * in_cb, global float * in_cr, global float * out_cb, global float * out_cr) {
	
	for (int t = get_global_id(0); t < 2048; t += get_global_size(0)) {
		float prev_cb = in_cb[t*SIZE_ROW];
		float prev_cr = in_cr[t*SIZE_ROW];
		out_cb[t*SIZE_ROW] = prev_cb;
		out_cr[t*SIZE_ROW] = prev_cr;
		out_cb[SIZE_ROW*t+ (SIZE_ROW - 1)] = in_cb[SIZE_ROW*t + (SIZE_ROW - 1)];
		out_cr[SIZE_ROW*t + (SIZE_ROW - 1)] = in_cr[SIZE_ROW*t + (SIZE_ROW - 1)];
		float a = 0.25;
		float b = 0.5;
		float c = 0.25;
	
		if (t != 0 && (t != (SIZE_ROW - 1))){
			for (int x = 1; x < SIZE_ROW - 1; x++) {
				prev_cb = a * prev_cb + b * in_cb[t*SIZE_ROW + x] + c * in_cb[t*SIZE_ROW + x + 1];
				out_cb[t*SIZE_ROW + x] = prev_cb;
				prev_cr = a * prev_cr + b * in_cr[t*SIZE_ROW + x] + c * in_cr[t*SIZE_ROW + x + 1];
				out_cr[t*SIZE_ROW + x] = prev_cr;
			}
		}
		else {
			for (int x = 1; x < SIZE_ROW - 1; x++) {
				out_cb[t*SIZE_ROW + x] = in_cb[t*SIZE_ROW + x];
				out_cr[t*SIZE_ROW + x] = in_cr[t*SIZE_ROW + x];
			}
		}
	}
}

