


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


/*
kernel void convertRGBtoYCbCr(global float *in, global float *out) {
	int WIDTH = get_global_size(0);
	int HEIGHT = get_global_size(1);
	// Don't do anything if we are on the edge.
	if (get_global_id(0) == 0 || get_global_id(1) == 0)
		return;
	if (get_global_id(0) == (WIDTH - 1) || get_global_id(1) == (HEIGHT - 1))
		return;
	int y = get_global_id(1);
	int x = get_global_id(0);
	// Load the data
	float a = in[WIDTH*(y - 1) + (x)];
	float b = in[WIDTH*(y)+(x - 1)];
	float c = in[WIDTH*(y + 1) + (x)];
	float d = in[WIDTH*(y)+(x + 1)];
	float e = in[WIDTH*y + x];
	// Do the computation and write back the results
	out[WIDTH*y + x] = (0.1*a + 0.2*b + 0.2*c + 0.1*d + 0.4*e);
}*/