#include "test_setup.h"

/* Basic Kernel*/
/*
kernel void convertRGBtoYCbCr(global float *in_rc, global float *in_gc, global float*in_bc,global float * out_y, global float * out_cb, global float *out_cr) {
	int thread_id = get_global_id(0);

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
*/
/*Using local memory*/

kernel void convertRGBtoYCbCr
(
	global float *in_rc, global float *in_gc, global float*in_bc, global float * out_y, 
	global float * out_cb, global float *out_cr, __local float * sharedMem , int shared_size
) 
{
	int tid = get_global_id(0);
	

	int bid = get_local_id(0);

	for (int thread_id = tid; thread_id < SIZE_FRAME; thread_id += get_global_size(0)) {
		float Y, Cb, Cr;
		sharedMem[bid] = in_rc[thread_id];
		sharedMem[bid + shared_size] = in_gc[thread_id];
		sharedMem[bid + shared_size * 2] = in_bc[thread_id];

		Y = 0 + ((float)0.299*sharedMem[bid]) + ((float)0.587*sharedMem[bid + shared_size]) + ((float)0.113*sharedMem[bid + shared_size * 2]);
		Cb = 128 - ((float)0.168736*sharedMem[bid]) - ((float)0.331264*sharedMem[bid + shared_size]) + ((float)0.5*sharedMem[bid + shared_size * 2]);
		Cr = 128 + ((float)0.5*sharedMem[bid]) - ((float)0.418688*sharedMem[bid + shared_size]) - ((float)0.081312*sharedMem[bid + shared_size * 2]);
		out_y[thread_id] = Y;
		out_cb[thread_id] = Cb;
		out_cr[thread_id] = Cr;
	}
}

/*debug kernel*/
/*
kernel void convertRGBtoYCbCr
(
	global float *in_rc, global float *in_gc, global float*in_bc, global float * out_y,
	global float * out_cb, global float *out_cr, __local float * sharedMem, int shared_size, global int * block_size
)
{
	int tid = get_global_id(0);


	*block_size = get_local_size(0);
	int bid = get_local_id(0);

	for (int thread_id = tid; thread_id < SIZE_FRAME; thread_id += get_global_size(0)) {
		float Y, Cb, Cr;
		sharedMem[bid] = in_rc[thread_id];
		sharedMem[bid + shared_size] = in_gc[thread_id];
		sharedMem[bid + shared_size * 2] = in_bc[thread_id];

		Y = 0 + ((float)0.299*sharedMem[bid]) + ((float)0.587*sharedMem[bid + shared_size]) + ((float)0.113*sharedMem[bid + shared_size * 2]);
		Cb = 128 - ((float)0.168736*sharedMem[bid]) - ((float)0.331264*sharedMem[bid + shared_size]) + ((float)0.5*sharedMem[bid + shared_size * 2]);
		Cr = 128 + ((float)0.5*sharedMem[bid]) - ((float)0.418688*sharedMem[bid + shared_size]) - ((float)0.081312*sharedMem[bid + shared_size * 2]);
		out_y[thread_id] = Y;
		out_cb[thread_id] = Cb;
		out_cr[thread_id] = Cr;
	}
}*/