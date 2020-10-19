#include "test_setup.h"

/* Basic Kernel*/

kernel void convertRGBtoYCbCr(global float *in_rc, global float *in_gc, global float*in_bc, global float * out_y, global float * out_cb, global float *out_cr) {



	for (int thread_id = get_global_id(0); thread_id < SIZE_FRAME; thread_id += get_global_size(0)) {
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
	
	for (int t = get_global_id(0); t < SIZE_ROW; t += get_global_size(0)) {
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
	
	for (int t = get_global_id(0); t < SIZE_ROW; t += get_global_size(0)) {
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

/*

#define wgs1 14
#define wgs2 14
#define wgsl 16


//__attribute__(reqd_work_group_size(16, 16, 1))
kernel void convert_lowPass(global float *in_rc, global float *in_gc, global float*in_bc, global float * out_y, global float * out_cb, global float *out_cr)
{
	__local float cb_local[wgsl*wgsl];
	__local float cr_local[wgsl*wgsl];

	__local float cb_first_sweep_local[wgsl*wgsl];
	__local float cr_first_sweep_local[wgsl*wgsl];


	int group_id = get_group_id(0);
	int thread_id[2];
	
	thread_id[0] = get_local_id(0);
	thread_id[1] = get_local_id(1);

	int x0 = wgs1 * get_group_id(0) + 1;
	int y0 = wgs2 * get_group_id(1) + 1;





	int global_index = x0+ thread_id[0]-1 + (y0+ thread_id[1]-1)*SIZE_ROW;

	int local_index = thread_id[0] + (wgsl) * thread_id[1];

	float R = in_rc[global_index];
	float G = in_gc[global_index];
	float B = in_bc[global_index];

	float Y = 0 + ((float)0.299*R) + ((float)0.587*G) + ((float)0.113*B);
	float Cb = 128 - ((float)0.168736*R) - ((float)0.331264*G) + ((float)0.5*B);
	float Cr = 128 + ((float)0.5*R) - ((float)0.418688*G) - ((float)0.081312*B);

	out_y[global_index] = Y;
	cb_local[local_index] = Cb;
	cr_local[local_index] = Cr;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	float a = 0.25;
	float b = 0.5;
	float c = 0.25;
	if ((thread_id[1] != 0) && (thread_id[1] != (wgsl - 1))) {
			cb_first_sweep_local[local_index] = a * cb_local[local_index - 16] + b * cb_local[local_index] + c * cb_local[local_index + 16];
			cr_first_sweep_local[local_index] = a * cr_local[local_index - 16] + b * cr_local[local_index] + c * cr_local[local_index + 16];
	}
	else {
		cb_first_sweep_local[local_index] = cb_local[local_index];
		cr_first_sweep_local[local_index] = cr_local[local_index];
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	if ((thread_id[0] != 0) && (thread_id[0] != (wgsl - 1))) {
			out_cb[global_index] = a * cb_first_sweep_local[local_index - 1] + b * cb_first_sweep_local[local_index] + c * cb_first_sweep_local[local_index + 1];
			out_cr[global_index] = a * cr_first_sweep_local[local_index - 1] + b * cr_first_sweep_local[local_index] + c * cr_first_sweep_local[local_index + 1];
	}
	else {
		out_cb[global_index] = cb_first_sweep_local[local_index];
		out_cr[global_index] = cr_first_sweep_local[local_index];
	}


}*/
