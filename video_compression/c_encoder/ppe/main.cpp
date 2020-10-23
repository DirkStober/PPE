#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <string>
#include <tiff.h>
#include <tiffio.h>
#include "custom_types.h"
#include "xml_aux.h"
#include "dct8x8_block.h"
#include <vector>
#include <mutex>
#include "gettimeofday.h"
#include "config.h"
#include <CL/cl.h>
#include "opencl_utils.h"
#include <omp.h>
#include <immintrin.h>



cl_device_id opencl_device;
cl_context opencl_context;
cl_command_queue opencl_queue;

using namespace std;


void loadImage(int number, string path, Image** photo) {
    string filename;
    TIFFRGBAImage img;
    char emsg[1024];
    
	filename = path + to_string(number) + ".tiff";
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if(tif==NULL) fprintf(stderr,"Failed opening image: %s\n", filename);
    if (!(TIFFRGBAImageBegin(&img, tif, 0, emsg))) TIFFError(filename.c_str(), emsg);
     
    uint32 w, h;
    size_t npixels;
    uint32* raster;
     
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    npixels = w * h;
    raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
     
    TIFFReadRGBAImage(tif, w, h, raster, 0);
     
    if(*photo==NULL) 
	*photo = new Image((int)w, (int)h, FULLSIZE);
     
    //Matlab and LibTIFF store the image diferently.
    //Necessary to mirror the image horizonatly to be consistent
    for (int j=0; j<(int)w; j++) {
        for (int i=0; i<(int)h; i++) {  
            // The inversion is ON PURPOSE
            (*photo)->rc->data[(h-1-i)*w+j] = (float)TIFFGetR(raster[i*w+j]);
            (*photo)->gc->data[(h-1-i)*w+j] = (float)TIFFGetG(raster[i*w+j]);
            (*photo)->bc->data[(h-1-i)*w+j] = (float)TIFFGetB(raster[i*w+j]);
        }
    }
     
    _TIFFfree(raster);
    TIFFRGBAImageEnd(&img);
    TIFFClose(tif);
     
}


void convertRGBtoYCbCr(Image* in, Image* out){
    int width = in->width;
    int height = in->height;

		for(int y=0; y<width; y++) {
			for (int x = 0; x<height; x++) {

			float R = in->rc->data[x*width+y];
			float G = in->gc->data[x*width+y];
			float B = in->bc->data[x*width+y];
			float Y = 0+((float)0.299*R)+((float)0.587*G)+((float)0.113*B);
			float Cb = 128-((float)0.168736*R)-((float)0.331264*G)+((float)0.5*B);
            float Cr = 128+((float)0.5*R)-((float)0.418688*G)-((float)0.081312*B);
			out->rc->data[x*width+y] = Y;
			out->gc->data[x*width+y] = Cb;
			out->bc->data[x*width+y] = Cr;
        }
    }
     
    //return out;
}

void openCL_convertRGBtoYCbCr(Image* in, Image * out, cl_kernel * kernel, cl_mem * device_ptrs) {
	int error;
	cl_mem in_r = device_ptrs[0];
	cl_mem in_g = device_ptrs[1];
	cl_mem in_b = device_ptrs[2];
	cl_mem out_y = device_ptrs[3];
	cl_mem out_cb = device_ptrs[4];
	cl_mem out_cr = device_ptrs[5];
	error = clEnqueueWriteBuffer(opencl_queue, in_r, CL_FALSE, 0, SIZE_FRAME * sizeof(float), in->rc->data, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(opencl_queue, in_g, CL_FALSE, 0, SIZE_FRAME * sizeof(float), in->gc->data, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(opencl_queue, in_b, CL_FALSE, 0, SIZE_FRAME * sizeof(float), in->bc->data, 0, NULL, NULL);
	clSetKernelArg(*kernel, 0, sizeof(in_r), &in_r);
	clSetKernelArg(*kernel, 1, sizeof(in_g), &in_g);
	clSetKernelArg(*kernel, 2, sizeof(in_b), &in_b);
	clSetKernelArg(*kernel, 3, sizeof(out_y), &out_y);
	clSetKernelArg(*kernel, 4, sizeof(out_cb), &out_cb);
	clSetKernelArg(*kernel, 5, sizeof(out_cr), &out_cr);
	size_t global_dimensions[] = { 1024 ,0,0 };
	error = clEnqueueNDRangeKernel(opencl_queue, *kernel, 1, NULL, global_dimensions, NULL, 0, NULL, NULL);


	//read the data
	error = clEnqueueReadBuffer(opencl_queue, out_y, CL_FALSE, 0, SIZE_FRAME * sizeof(float), out->rc->data, 0, NULL, NULL);
	error = clEnqueueReadBuffer(opencl_queue, out_cb, CL_FALSE, 0, SIZE_FRAME * sizeof(float), out->gc->data, 0, NULL, NULL);
	error = clEnqueueReadBuffer(opencl_queue, out_cr, CL_FALSE, 0, SIZE_FRAME * sizeof(float), out->bc->data, 0, NULL, NULL);
	error = clFinish(opencl_queue);

}

void openCL_convert_lowPass(Image* in, Frame * out, cl_kernel * kernel, cl_mem * device_ptrs, cl_event * frame_event) {
	int error;
	cl_mem in_r = device_ptrs[0];
	cl_mem in_g = device_ptrs[1];
	cl_mem in_b = device_ptrs[2];
	cl_mem out_y = device_ptrs[3];
	cl_mem out_cb = device_ptrs[4];
	cl_mem out_cr = device_ptrs[5];
	error = clEnqueueWriteBuffer(opencl_queue, in_r, CL_FALSE, 0, SIZE_FRAME * sizeof(float), in->rc->data, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(opencl_queue, in_g, CL_FALSE, 0, SIZE_FRAME * sizeof(float), in->gc->data, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(opencl_queue, in_b, CL_FALSE, 0, SIZE_FRAME * sizeof(float), in->bc->data, 0, NULL, NULL);
	size_t global_dimensions[] = { 128 ,0,0 };
	error = clEnqueueNDRangeKernel(opencl_queue, kernel[0], 1, NULL, global_dimensions, NULL, 0, NULL, NULL);

	error = clEnqueueReadBuffer(opencl_queue, out_y, CL_FALSE, 0, SIZE_FRAME * sizeof(float), out->Y->data, 0, NULL, NULL);

	error = clEnqueueNDRangeKernel(opencl_queue, kernel[1], 1, NULL, global_dimensions, NULL, 0, NULL, NULL);
	error = clEnqueueNDRangeKernel(opencl_queue, kernel[2], 1, NULL, global_dimensions, NULL, 0, NULL, NULL);


	//read the data
	error = clEnqueueReadBuffer(opencl_queue, out_cb, CL_FALSE, 0, SIZE_FRAME * sizeof(float), out->Cb->data, 0, NULL, NULL);
	error = clEnqueueReadBuffer(opencl_queue, out_cr, CL_FALSE, 0, SIZE_FRAME * sizeof(float), out->Cr->data, 0, NULL, frame_event);
	//error = clFinish(opencl_queue);

}



Channel* lowPass(Channel* in, Channel* out){
    // Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
    // E.g., blur
    // weights for neighboring pixels
    float a=0.25;
    float b=0.5;
    float c=0.25;

	int width = in->width; 
	int height = in->height;
     
    //out = in; TODO Is this necessary?
	for(int i=0; i<width*height; i++) out->data[i] =in->data[i];
     
     
    // In X
    for (int y=1; y<(width-1); y++) {
        for (int x=1; x<(height-1); x++) {
            out->data[x*width+y] = a*in->data[(x-1)*width+y]+b*in->data[x*width+y]+c*in->data[(x+1)*width+y];
        }
    }
    // In Y
    for (int y=1; y<(width-1); y++) {
        for (int x=1; x<(height-1); x++) {
            out->data[x*width+y] = a*out->data[x*width+(y-1)]+b*out->data[x*width+y]+c*out->data[x*width+(y+1)];
        }
    }
     
    return out;
}


std::vector<mVector>* motionVectorSearch(Frame* source, Frame* match, int width, int height) {
    std::vector<mVector> *motion_vectors = new std::vector<mVector>(); // empty list of ints

    float Y_weight = 0.5;
    float Cr_weight = 0.25;
    float Cb_weight = 0.25;
     
    //Window size is how much on each side of the block we search
    int window_size = 16;
    int block_size = 16;
     
    //How far from the edge we can go since we don't special case the edges
    int inset = (int) max((float)window_size, (float)block_size);
    int iter=0;
     
    for (int my=inset; my<height-(inset+window_size)+1; my+=block_size) {
      for (int mx=inset; mx<width-(inset+window_size)+1; mx+=block_size) {
         
            float best_match_sad = 1e10;
            int best_match_location[2] = {0, 0};
             
            for(int sy=my-window_size; sy<my+window_size; sy++) {
                for(int sx=mx-window_size; sx<mx+window_size; sx++) {        
                    float current_match_sad = 0;
                    // Do the SAD
                    for (int y=0; y<block_size; y++) {
                        for (int x=0; x<block_size; x++) {                
                            int match_x = mx+x;
                            int match_y = my+y;
                            int search_x = sx+x;
                            int search_y = sy+y;
                            float diff_Y = abs(match->Y->data[match_x*width+match_y] - source->Y->data[search_x*width+search_y]);
                            float diff_Cb = abs(match->Cb->data[match_x*width+match_y] - source->Cb->data[search_x*width+search_y]);
                            float diff_Cr = abs(match->Cr->data[match_x*width+match_y] - source->Cr->data[search_x*width+search_y]);
                             
                            float diff_total = Y_weight*diff_Y + Cb_weight*diff_Cb + Cr_weight*diff_Cr;
                            current_match_sad = current_match_sad + diff_total;

                        }
                    } //end SAD
                     
                    if (current_match_sad <= best_match_sad){
                        best_match_sad = current_match_sad;
                        best_match_location[0] = sx-mx;
                        best_match_location[1] = sy-my;
                    }        
                }
            }
             
            mVector v;
            v.a=best_match_location[0];
            v.b=best_match_location[1];
            motion_vectors->push_back(v);
 
        }
    }
     
    return motion_vectors;
}


Frame* computeDelta(Frame* i_frame_ycbcr, Frame* p_frame_ycbcr, std::vector<mVector>* motion_vectors){
    Frame *delta = new Frame(p_frame_ycbcr);
 
    int width = i_frame_ycbcr->width;
    int height = i_frame_ycbcr->height;
    int window_size = 16;
    int block_size = 16;
    // How far from the edge we can go since we don't special case the edges
    int inset = (int) max((float) window_size, (float)block_size);
     
    int current_block = 0;
    for(int my=inset; my<width-(inset+window_size)+1; my+=block_size) {
        for(int mx=inset; mx<height-(inset+window_size)+1; mx+=block_size) {
            int vector[2];
            vector[0]=(int)motion_vectors->at(current_block).a;
            vector[1]=(int)motion_vectors->at(current_block).b;
             
            // copy the block
                for(int y=0; y<block_size; y++) {
                    for(int x=0; x<block_size; x++) {
 
                    int src_x = mx+vector[0]+x;
                    int src_y = my+vector[1]+y;
                    int dst_x = mx+x;
                    int dst_y = my+y;
                    delta->Y->data[dst_x*width+dst_y] = delta->Y->data[dst_x*width+dst_y] - i_frame_ycbcr->Y->data[src_x*width+src_y];
                    delta->Cb->data[dst_x*width+dst_y] = delta->Cb->data[dst_x*width+dst_y] - i_frame_ycbcr->Cb->data[src_x*width+src_y];
                    delta->Cr->data[dst_x*width+dst_y] = delta->Cr->data[dst_x*width+dst_y] - i_frame_ycbcr->Cr->data[src_x*width+src_y];
                }
            }
 
            current_block = current_block + 1;
        }
    }
    return delta;
}
 

Channel* downSample(Channel* in){
	int width = in->width;
	int height = in->height;
	int w2=width/2;
	int h2=height/2;

	Channel* out = new Channel((width/2),(height/2));

	for (int x2 = 0, x = 0; x2<h2; x2++) {
		for(int y2=0,y=0; y2<w2; y2++) {
            out->data[x2*w2+y2]= in->data[x*width+y];
			y += 2;
        }
		x += 2;
    }
 
    return out;
}


void dct8x8(Channel* in, Channel* out){
	int width = in->width; 
	int height = in -> height;

    // 8x8 block dct on each block
    for(int i=0; i<width*height; i++) {
        //in->data[i] -= 128;
        out->data[i] = 0; //zeros
    }

	for (int x = 0; x<height; x += 8) {
		for(int y=0; y<width; y+=8) {
            dct8x8_block(&(in->data[x*width+y]),&(out->data[x*width+y]), width);
        }
    }
}


void round_block(float* in, float* out, int stride){
    float quantMatrix[8][8] ={
        {16, 11, 10, 16,  24,  40,  51,  61},
        {12, 12, 14, 19,  26,  58,  60,  55},
        {14, 13, 16, 24,  40,  57,  69,  56},
        {14, 17, 22, 29,  51,  87,  80,  62},
        {18, 22, 37, 56,  68, 109, 103,  77},
        {24, 35, 55, 64,  81, 104, 113,  92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99},
    };

	for (int x = 0; x<8; x++) {
		for(int y=0; y<8; y++) {
            quantMatrix[x][y] = ceil(quantMatrix[x][y]/QUALITY);
            out[x*stride+y] = (float)round(in[x*stride+y]/quantMatrix[x][y]);
        }
    }
}


void quant8x8(Channel* in, Channel* out) {
	int width = in->width;
	int height = in->height;

    for(int i=0; i<width*height; i++) {
        out->data[i]=0; //zeros
    }

	for (int x = 0; x<height; x += 8) {
		for (int y=0; y<width; y+=8) {    
            round_block(&(in->data[x*width+y]), &(out->data[x*width+y]), width);
        }
    }
}


void dcDiff(Channel* in, Channel* out) {
	int width = in->width;
	int height = in->height;

    int number_of_dc = width*height/64;
	double* dc_values_transposed = new double[number_of_dc];
    double* dc_values = new double[number_of_dc];
 
    int iter = 0;
    for(int j=0; j<width; j+=8){
        for(int i=0; i<height; i+=8) {
            dc_values_transposed[iter] = in->data[i*width+j];
            dc_values[iter] = in->data[i*width+j];
            iter++;
        }
    }
 
    int new_w = (int) max((float)(width/8), 1);
    int new_h = (int) max((float)(height/8), 1);
     
    out->data[0] = (float)dc_values[0];
  
    double prev = 0.;
    iter = 0;
    for (int j=0; j<new_w; j++) {
        for (int i=0; i<new_h; i++) {
            out->data[iter]= (float)(dc_values[i*new_w+j] - prev);
            prev = dc_values[i*new_w+j];
            iter++;
        }
    }
	delete dc_values_transposed;
	delete dc_values;

}

void cpyBlock(float* in, float* out, int blocksize, int stride) {
    for (int j=0; j<blocksize; j++) {
        for (int i=0; i<blocksize; i++) {
            out[i*blocksize+j] = in[i*stride+j];
        }
    }
}


void zigZagOrder(Channel* in, Channel* ordered) {
	int width = in->width;
	int height = in->height;
	int zigZagIndex[64] = { 0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,
		48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,
		44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63 };


	int tid = omp_get_thread_num();
	int size = omp_get_num_threads();
	
	int blockNumber = tid* width /8;
	float _block[MPEG_CONSTANT];


	for (int x = tid*8; x<height; x += size*8) {
		blockNumber = x/8 * width / 8;
		for (int y = 0; y<width; y += 8) {
			cpyBlock(&(in->data[x*width + y]), _block, 8, width); //block = in(x:x+7,y:y+7);
																  //Put the coefficients in zig-zag order
			float zigZagOrdered[MPEG_CONSTANT] = { 0 };
			for (int index = 0; index < MPEG_CONSTANT; index++) {
				zigZagOrdered[index] = _block[zigZagIndex[index]];
			}
			for (int i = 0; i<MPEG_CONSTANT; i++)
				ordered->data[blockNumber*MPEG_CONSTANT + i] = zigZagOrdered[i];
			blockNumber++;
		}
	}
}


void encode8x8(Channel* ordered, SMatrix* encoded){
	int width = encoded->height;
	int height = encoded->width;
    int num_blocks = height;

	int tid = omp_get_thread_num();
	int size = omp_get_num_threads();

	for(int i=tid; i<num_blocks; i+=size) {
		std::string block_encode[MPEG_CONSTANT];
		for (int j=0; j<MPEG_CONSTANT; j++) {
            block_encode[j]="\0"; //necessary to initialize every string position to empty string
        }

        int num_coeff = MPEG_CONSTANT; //width
        int encoded_index = 0;
        int in_zero_run = 0;
        int zero_count = 0;
 
        // Skip DC coefficient
        for(int c=1; c<num_coeff; c++){
            float coeff = ordered->data[i*width + c];
            if (coeff == 0){
                if (in_zero_run == 0){
                    zero_count = 0;
                    in_zero_run = 1;
                }
                zero_count = zero_count + 1;
            }
            else {
                if (in_zero_run == 1){
                    in_zero_run = 0;
					block_encode[encoded_index] = "Z" + std::to_string(zero_count);
                    encoded_index = encoded_index+1;
                }
				block_encode[encoded_index] = std::to_string((int)coeff);
                encoded_index = encoded_index+1;
            }
        }
 
        // If we were in a zero run at the end attach it as well.    
        if (in_zero_run == 1) {
            if (zero_count > 1) {
				block_encode[encoded_index] = "Z" + std::to_string(zero_count);
            } else {
				block_encode[encoded_index] = "0";
            }
        }
 
 
        for(int it=0; it < MPEG_CONSTANT; it++) {
			if (block_encode[it].length() > 0) 
				encoded->data[i*width+it] = new std::string(block_encode[it]);
            else
                it = MPEG_CONSTANT;
        }
    }
}

void setupCL(cl_kernel * kernel, cl_program * program) {
	// ======== Setup OpenCL
	cl_platform_id platforms[10];
	cl_device_id found_devices[10];
	cl_uint     num_platforms;
	cl_uint number_of_devices_found;
	char platform_name[1000];
	clGetPlatformIDs(10, platforms, &num_platforms);
	for (int i = 0; i < num_platforms; i++) {
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1000, platform_name, NULL);
		//clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 10, found_devices, &number_of_devices_found);
		std::cout << "platform_name: " << platform_name << std::endl;
	}

	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 10, found_devices, &number_of_devices_found);
	opencl_device = found_devices[0];
	char device_name[1000];
	clGetDeviceInfo(opencl_device, CL_DEVICE_NAME, 1000, device_name, NULL);
	cout << "Device name : " << device_name << "\n";

	opencl_context = clCreateContext(NULL, 1, &opencl_device, NULL, NULL, NULL);
	opencl_queue = clCreateCommandQueue(opencl_context, opencl_device, CL_QUEUE_PROFILING_ENABLE, NULL);


	FILE *file;
	file = fopen("kernel.cl", "r");
	struct stat stat_info;
	int error = stat("kernel.cl", &stat_info);
	if (error) {
		printf("load_source_file: stat failed on file %s: %d\n", "kernel.cl", error);
		exit(1);
	}
	char * program_text;
	program_text = (char*)malloc(stat_info.st_size + 1);
	memset(program_text, 0, stat_info.st_size + 1);

	size_t result = fread(program_text, stat_info.st_size, 1, file);

	*program = clCreateProgramWithSource(opencl_context, 1, (const char**)&program_text, NULL, &error);

	error = clBuildProgram(*program, 1, &opencl_device, NULL, NULL, NULL);
	checkError(error, "clBuildProgram");
	kernel[0] = clCreateKernel(*program, "convertRGBtoYCbCr", &error);
	kernel[1] = clCreateKernel(*program, "first_sweep", &error);
	kernel[2] = clCreateKernel(*program, "second_sweep", &error);
	checkError(error, "clCreateKernel");
	free(program_text);

	/*Get some information about work group size*/
}



void blocked_ds_dct_round(Channel * data_in , Channel * data_out )
{


	
	int tid = omp_get_thread_num();
	int size = omp_get_num_threads();
	for (int y = 8 * tid; y < (SIZE_ROW / 2); y += 8* size)
	{
		for (int x = 0; x < (SIZE_ROW / 2); x += 8)
		{
			for (int yy = y; yy < (y + 8); yy++)
			{
				for (int xx = x; xx < (x + 8); xx++)
				{
					data_out->data[yy*(SIZE_ROW / 2) + xx] = data_in->data[yy * 2 * SIZE_ROW + xx * 2];
				}
			}
			dct8x8_block(&data_out->data[y*(SIZE_ROW / 2) + x], &data_out->data[y*(SIZE_ROW / 2) + x], (SIZE_ROW / 2));
			round_block(&data_out->data[y*(SIZE_ROW / 2) + x], &data_out->data[y*(SIZE_ROW / 2) + x], (SIZE_ROW / 2));
		}
	}
}

void blocked_dct_round(Channel * data_in, Channel * data_out)
{
	int tid = omp_get_thread_num();
	int size = omp_get_num_threads();
	for (int y = 8*tid; y < SIZE_ROW; y += 8*size)
	{
		for (int x = 0; x < SIZE_ROW; x += 8)
		{
			dct8x8_block(&data_in->data[y*(SIZE_ROW) + x], &data_out->data[y*(SIZE_ROW) + x], (SIZE_ROW));
			round_block(&data_out->data[y*(SIZE_ROW) + x], &data_out->data[y*(SIZE_ROW) + x], (SIZE_ROW));
		}
	}

}



void motionVectorSearch(Frame* source, Frame* match,Frame * delta, int width, int height, std::vector<mVector> *motion_vectors) {
	//std::vector<mVector> *motion_vectors = new std::vector<mVector>(); // empty list of ints

		int d_width = source->width;
	int d_height = source->height;

	float Y_weight = 2;
	float Cr_weight = 1;
	float Cb_weight = 1;

	uint16_t yw_vec[16] = {
		Y_weight, Y_weight, Y_weight, Y_weight,
		Y_weight, Y_weight, Y_weight, Y_weight,
		Y_weight, Y_weight, Y_weight, Y_weight,
		Y_weight, Y_weight, Y_weight, Y_weight,
	};
	uint16_t crw_vec[16] = {
		Cr_weight, Cr_weight, Cr_weight, Cr_weight,
		Cr_weight, Cr_weight, Cr_weight, Cr_weight,
		Cr_weight, Cr_weight, Cr_weight, Cr_weight,
		Cr_weight, Cr_weight, Cr_weight, Cr_weight,
	};
	uint16_t cbw_vec[16] = {
		Cb_weight, Cb_weight, Cb_weight, Cb_weight,
		Cb_weight, Cb_weight, Cb_weight, Cb_weight,
		Cb_weight, Cb_weight, Cb_weight, Cb_weight,
		Cb_weight, Cb_weight, Cb_weight, Cb_weight,
	};

	__m256i
		mm_yweight = _mm256_load_si256((const __m256i *)yw_vec),
		mm_crweight = _mm256_load_si256((const __m256i *)crw_vec),
		mm_cbweight = _mm256_load_si256((const __m256i *)cbw_vec);

	//Window size is how much on each side of the block we search
	int window_size = 8;
	int block_size = 8;

	//How far from the edge we can go since we don't special case the edges
	int inset = (int)max((float)window_size, (float)block_size);
	int iter = 0;
	inset = 16;
	int tid = omp_get_thread_num();
	int size = omp_get_num_threads();
	for (int my = inset + tid*block_size; my < width - (inset + window_size) + 1; my += block_size*size) {
		for (int mx = inset; mx < height - (inset + window_size) + 1; mx += block_size) {

			uint16_t zeros[16 * 16][16 * 2] = { 0 };

			for (int outer_loop = 0; outer_loop < 16; ++outer_loop)
			{

				int
					match_x = mx + outer_loop,
					match_y = my;
				__m256
					mmatch_y1 = _mm256_load_ps(&(match->Y->data[match_x*width + match_y])),
					mmatch_cb1 = _mm256_load_ps(&(match->Cb->data[match_x*width + match_y])),
					mmatch_cr1 = _mm256_load_ps(&(match->Cr->data[match_x*width + match_y]));

				__m256i
					mmatch32_y1 = _mm256_cvtps_epi32(mmatch_y1),
					mmatch32_cb1 = _mm256_cvtps_epi32(mmatch_cb1),
					mmatch32_cr1 = _mm256_cvtps_epi32(mmatch_cr1);

				__m256i
					mmatch16_y1 = _mm256_packus_epi32(mmatch32_y1, mmatch32_y1),
					mmatch16_cb1 = _mm256_packus_epi32(mmatch32_cb1, mmatch32_cb1),
					mmatch16_cr1 = _mm256_packus_epi32(mmatch32_cr1, mmatch32_cr1);

				const int
					perm_mask = 0b11011000;
				//perm_mask = 0b00100111;

				mmatch16_y1 = _mm256_permute4x64_epi64(mmatch16_y1, perm_mask),
					mmatch16_cb1 = _mm256_permute4x64_epi64(mmatch16_cb1, perm_mask),
					mmatch16_cr1 = _mm256_permute4x64_epi64(mmatch16_cr1, perm_mask);

				__m256i
					mmatch8_y = _mm256_packus_epi16(mmatch16_y1, mmatch16_y1),
					mmatch8_cb = _mm256_packus_epi16(mmatch16_cb1, mmatch16_cb1),
					mmatch8_cr = _mm256_packus_epi16(mmatch16_cr1, mmatch16_cr1);

				mmatch8_y = _mm256_permute4x64_epi64(mmatch8_y, perm_mask),
					mmatch8_cb = _mm256_permute4x64_epi64(mmatch8_cb, perm_mask),
					mmatch8_cr = _mm256_permute4x64_epi64(mmatch8_cr, perm_mask);

				for (int inner_loop = outer_loop - window_size, y_idx = outer_loop;
					inner_loop < outer_loop;
					inner_loop++, y_idx += 8)
				{


					/*
					Do one row of sads
					*/


					/*
					First half
					*/

					int sx = mx - window_size;
					int search_x = sx;
					int search_y = inner_loop;
					__m256
						msearch_y1 = _mm256_load_ps(&(source->Y->data[search_x*width + search_y])),
						msearch_cb1 = _mm256_load_ps(&(source->Cb->data[search_x*width + search_y])),
						msearch_cr1 = _mm256_load_ps(&(source->Cr->data[search_x*width + search_y])),
						msearch_y2 = _mm256_load_ps(&(source->Y->data[search_x*width + search_y + 8])),
						msearch_cb2 = _mm256_load_ps(&(source->Cb->data[search_x*width + search_y + 8])),
						msearch_cr2 = _mm256_load_ps(&(source->Cr->data[search_x*width + search_y + 8]));


					__m256i
						msearch32_y1 = _mm256_cvtps_epi32(msearch_y1),
						msearch32_cb1 = _mm256_cvtps_epi32(msearch_cb1),
						msearch32_cr1 = _mm256_cvtps_epi32(msearch_cr1),
						msearch32_y2 = _mm256_cvtps_epi32(msearch_y2),
						msearch32_cb2 = _mm256_cvtps_epi32(msearch_cb2),
						msearch32_cr2 = _mm256_cvtps_epi32(msearch_cr2);

					__m256i
						msearch16_y1 = _mm256_packus_epi32(msearch32_y1, msearch32_y2),
						msearch16_cb1 = _mm256_packus_epi32(msearch32_cb1, msearch32_cb2),
						msearch16_cr1 = _mm256_packus_epi32(msearch32_cr1, msearch32_cr2);

					msearch16_y1 = _mm256_permute4x64_epi64(msearch16_y1, perm_mask),
						msearch16_cb1 = _mm256_permute4x64_epi64(msearch16_cb1, perm_mask),
						msearch16_cr1 = _mm256_permute4x64_epi64(msearch16_cr1, perm_mask);

					__m256i
						msearch8_y = _mm256_packus_epi16(msearch16_y1, msearch16_y1),
						msearch8_cb = _mm256_packus_epi16(msearch16_cb1, msearch16_cb1),
						msearch8_cr = _mm256_packus_epi16(msearch16_cr1, msearch16_cr1);

					msearch8_y = _mm256_permute4x64_epi64(msearch8_y, perm_mask),
						msearch8_cb = _mm256_permute4x64_epi64(msearch8_cb, perm_mask),
						msearch8_cr = _mm256_permute4x64_epi64(msearch8_cr, perm_mask);

					/*
					Compute all sums for one row[window_size * 2]
					*/

					const int
						offset_mask = 0b101000;

					__m256i
						//sums = _mm256_load_si256((const __m256i*) zeros[y_idx]),
						sum_y = _mm256_mpsadbw_epu8(msearch8_y, mmatch8_y, offset_mask),
						sum_cb = _mm256_mpsadbw_epu8(msearch8_cb, mmatch8_cb, offset_mask),
						sum_cr = _mm256_mpsadbw_epu8(msearch8_cr, mmatch8_cr, offset_mask);

					__m256i
						wsum_y = _mm256_mullo_epi16(sum_y, mm_yweight),
						wsum_cb = _mm256_mullo_epi16(sum_cb, mm_cbweight),
						wsum_cr = _mm256_mullo_epi16(sum_cr, mm_crweight);
					__m256i
						sums = _mm256_add_epi16(
							_mm256_add_epi16(wsum_y, wsum_cb),
							wsum_cr);

					_mm256_store_si256((__m256i*) &zeros[y_idx], sums);

					/*sx = mx - window_size;
					search_x = sx;
					search_y = inner_loop;*/

					/*
					Second half
					*/

					__m256
						msearch_y3 = _mm256_load_ps(&(source->Y->data[search_x*width + search_y + 16])),
						msearch_cb3 = _mm256_load_ps(&(source->Cb->data[search_x*width + search_y + 16])),
						msearch_cr3 = _mm256_load_ps(&(source->Cr->data[search_x*width + search_y + 16])),
						msearch_y4 = _mm256_load_ps(&(source->Y->data[search_x*width + search_y + 24])),
						msearch_cb4 = _mm256_load_ps(&(source->Cb->data[search_x*width + search_y + 24])),
						msearch_cr4 = _mm256_load_ps(&(source->Cr->data[search_x*width + search_y + 24]));


					__m256i
						msearch32_y3 = _mm256_cvtps_epi32(msearch_y3),
						msearch32_cb3 = _mm256_cvtps_epi32(msearch_cb3),
						msearch32_cr3 = _mm256_cvtps_epi32(msearch_cr3),
						msearch32_y4 = _mm256_cvtps_epi32(msearch_y4),
						msearch32_cb4 = _mm256_cvtps_epi32(msearch_cb4),
						msearch32_cr4 = _mm256_cvtps_epi32(msearch_cr4);

					__m256i
						msearch16_y2 = _mm256_packus_epi32(msearch32_y3, msearch32_y4),
						msearch16_cb2 = _mm256_packus_epi32(msearch32_cb3, msearch32_cb4),
						msearch16_cr2 = _mm256_packus_epi32(msearch32_cr3, msearch32_cr4);

					msearch16_y2 = _mm256_permute4x64_epi64(msearch16_y2, perm_mask),
						msearch16_cb2 = _mm256_permute4x64_epi64(msearch16_cb2, perm_mask),
						msearch16_cr2 = _mm256_permute4x64_epi64(msearch16_cr2, perm_mask);

					__m256i
						msearch8_y2 = _mm256_packus_epi16(msearch16_y2, msearch16_y2),
						msearch8_cb2 = _mm256_packus_epi16(msearch16_cb2, msearch16_cb2),
						msearch8_cr2 = _mm256_packus_epi16(msearch16_cr2, msearch16_cr2);

					msearch8_y2 = _mm256_permute4x64_epi64(msearch8_y2, perm_mask),
						msearch8_cb2 = _mm256_permute4x64_epi64(msearch8_cb2, perm_mask),
						msearch8_cr2 = _mm256_permute4x64_epi64(msearch8_cr2, perm_mask);


					sum_y = _mm256_mpsadbw_epu8(msearch8_y2, mmatch8_y, offset_mask),
						sum_cb = _mm256_mpsadbw_epu8(msearch8_cb2, mmatch8_cb, offset_mask),
						sum_cr = _mm256_mpsadbw_epu8(msearch8_cr2, mmatch8_cr, offset_mask);


					wsum_y = _mm256_mullo_epi16(sum_y, mm_yweight),
						wsum_cb = _mm256_mullo_epi16(sum_cb, mm_cbweight),
						wsum_cr = _mm256_mullo_epi16(sum_cr, mm_crweight);
					sums = _mm256_add_epi16(
						_mm256_add_epi16(wsum_y, wsum_cb),
						wsum_cr);

					_mm256_store_si256((__m256i*) &zeros[y_idx + 16], sums);
				}
			}


			///
			int best_match_location[2] = { 0, 0 };
			for (int other_loop = 0,
				best_sum = 0xFFFFFFFF,
				sy = my - window_size;
				other_loop < 16;
				++other_loop,
				++sy)
			{
				//for (int second_loop = 0; second_loop < 8; ++second_loop)

				for (int g_x = 0,
					sx = mx - window_size;
					g_x < 8;
					++g_x, ++sx)
				{
					uint32_t temp_sum = 0;
					for (int curr_sum_idx = 0; curr_sum_idx < 8; ++curr_sum_idx)
					{
						int
							y = curr_sum_idx * 1 + other_loop * 8,
							x1 = 0 + g_x,
							x2 = 8 + g_x,
							x3 = 16 + g_x,
							x4 = 24 + g_x;
						temp_sum += zeros[y][x1] + zeros[y][x2];
					}

					if (temp_sum < best_sum)
					{
						best_sum = temp_sum;
						best_match_location[0] = sx - mx;
						best_match_location[1] = sy - my;
					}
				}
			}
			mVector v;
			v.a = best_match_location[0];
			v.b = best_match_location[1];
			motion_vectors->push_back(v);
			for (int y = 0; y < block_size; y++) {
				for (int x = 0; x < block_size; x++) {

					int src_x = mx + best_match_location[0] + x;
					int src_y = my + best_match_location[1] + y;
					int dst_x = mx + x;
					int dst_y = my + y;
					delta->Y->data[dst_x*d_width + dst_y] = delta->Y->data[dst_x*d_width + dst_y] - source->Y->data[src_x*d_width + src_y];
					delta->Cb->data[dst_x*d_width + dst_y] = delta->Cb->data[dst_x*d_width + dst_y] - source->Cb->data[src_x*d_width + src_y];
					delta->Cr->data[dst_x*d_width + dst_y] = delta->Cr->data[dst_x*d_width + dst_y] - source->Cr->data[src_x*d_width + src_y];
				}
			}

		}
	}


}




int encode() {
    int end_frame = int(N_FRAMES);
    int i_frame_frequency = int(I_FRAME_FREQ);
	struct timeval starttime, endtime;
	double runtime[10] = {0};
    
    // Hardcoded paths
    string image_path =  "..\\..\\inputs\\" + string(image_name) + "\\" + image_name + ".";
	string stream_path = "..\\..\\outputs\\stream_c_" + string(image_name) + ".xml";

    xmlDocPtr stream = NULL;
 
    Image* frame_rgb = NULL;
    Image* previous_frame_rgb = NULL;
    Frame* previous_frame_lowpassed = NULL;
 
    loadImage(0, image_path, &frame_rgb);
 
    int width = frame_rgb->width;
    int height = frame_rgb->height;
    int npixels = width*height;
 
	delete frame_rgb;

	createStatsFile();
    stream = create_xml_stream(width, height, QUALITY, WINDOW_SIZE, BLOCK_SIZE);
    vector<mVector>* motion_vectors = NULL;

	// ======== Initialize


	cl_kernel  kernel[3];
	cl_program program;
	setupCL(kernel,&program);
	int error;

	// Create the data objects
	cl_mem device_ptrs[6];
	for (int i = 0; i < 3; i++) {
		device_ptrs[i] = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE|| CL_MEM_HOST_WRITE_ONLY, SIZE_FRAME * sizeof(float), NULL, &error);
		clSetKernelArg(kernel[0], i, sizeof(device_ptrs[i]), &device_ptrs[i]);
	}
	for (int i = 3; i < 6; i++) {
		device_ptrs[i] = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE|| CL_MEM_HOST_READ_ONLY, SIZE_FRAME * sizeof(float), NULL, &error);
		clSetKernelArg(kernel[0], i, sizeof(device_ptrs[i]), &device_ptrs[i]);
	}
	clSetKernelArg(kernel[1], 0, sizeof(&device_ptrs[4]), &device_ptrs[4]);
	clSetKernelArg(kernel[1], 1, sizeof(device_ptrs[5]), &device_ptrs[5]);
	clSetKernelArg(kernel[1], 2, sizeof(device_ptrs[1]), &device_ptrs[1]);
	clSetKernelArg(kernel[1], 3, sizeof(device_ptrs[2]), &device_ptrs[2]);
	clSetKernelArg(kernel[2], 0, sizeof(device_ptrs[1]), &device_ptrs[1]);
	clSetKernelArg(kernel[2], 1, sizeof(device_ptrs[2]), &device_ptrs[2]);
	clSetKernelArg(kernel[2], 2, sizeof(device_ptrs[4]), &device_ptrs[4]);
	clSetKernelArg(kernel[2], 3, sizeof(device_ptrs[5]), &device_ptrs[5]);

	std::vector< Frame*> loaded_images(end_frame);
	std::vector<std::mutex> loaded_images_mutex(end_frame);

	std::vector<xmlDocPtr> encoded_images(end_frame);
	std::vector<std::mutex> encoded_images_mutex(end_frame);


	int num_threads = 5;


	//omp_set_num_threads(num_threads);

	cl_event *loaded_events = new cl_event[end_frame];

	omp_set_nested(1);
	#pragma omp parallel num_threads(2)
	{
		int tid = omp_get_thread_num();
		if (tid == 1) {
			for (int i = 0; i < end_frame; i++) {
				loaded_images_mutex[i].lock();
			}
		}
		else if (tid == 0) {
			for (int i = 0; i < end_frame; i++) {
				encoded_images_mutex[i].lock();
			}
		}
		#pragma omp barrier
		if(tid == 1)
			{
				for (int i = 0; i < end_frame; i++) {
					Image * load_RGB = new Image(SIZE_ROW, SIZE_ROW, 0);
					loaded_images[i] = new Frame(SIZE_ROW, SIZE_ROW, 0);
					printf("Loading Frame %d...\n",i);
					loadImage(i, image_path, &load_RGB);
					openCL_convert_lowPass(load_RGB, loaded_images[i], kernel, device_ptrs,&loaded_events[i]);
					loaded_images_mutex[i].unlock();
					printf("Frame %d ready...\n", i);
				}
				for (int i = 0; i < end_frame; i++) {
					print("WriteImage..."); 
					encoded_images_mutex[i].lock();
					write_stream(stream_path, encoded_images[i]);
					encoded_images_mutex[i].unlock();

				}
			}
		else {
			Frame * frame_lowpassed;
			Frame * frame_lowpassed_a_final;
			Frame* frame_quant = new Frame(width, height, DOWNSAMPLE);
			Frame* frame_zigzag = new Frame(MPEG_CONSTANT, width*height / MPEG_CONSTANT, ZIGZAG);
			FrameEncode* frame_encode = new FrameEncode(width, height, MPEG_CONSTANT);
			std::vector<mVector> * mvecs[4];
			#pragma omp parallel num_threads(4)
			{
				int tid = omp_get_thread_num();
				for (int frame_number = 0; frame_number < end_frame; frame_number += 1) {

					frame_rgb = NULL;
					gettimeofday(&starttime, NULL);

					if (tid == 0) {
						print("Waiting for image");
						loaded_images_mutex[frame_number].lock();
						loaded_images_mutex[frame_number].unlock();
						clWaitForEvents(1, &loaded_events[frame_number]);
						frame_lowpassed = loaded_images[frame_number];

						if (frame_number % i_frame_frequency != 0) {
							print("Motion vector: ...");
							frame_lowpassed_a_final = new Frame(frame_lowpassed);
						}
						else {
							// We have a I frame
							frame_lowpassed_a_final = new Frame(frame_lowpassed);
							motion_vectors = NULL;
						}
					}
					#pragma omp barrier
					if (frame_number % i_frame_frequency != 0) {
						mvecs[tid] = new std::vector<mVector>();
						motionVectorSearch(previous_frame_lowpassed, frame_lowpassed, frame_lowpassed_a_final, frame_lowpassed->width, frame_lowpassed->height, mvecs[tid]);

					}
					#pragma omp barrier
					if (tid == 0) {
						if (frame_number % i_frame_frequency != 0) {
							print("Collecting motion vectors: ...");
							motion_vectors = new std::vector<mVector>();
							for (int i = 0; i < mvecs[0]->size(); i++) {
								motion_vectors->push_back(mvecs[0]->at(i));
								motion_vectors->push_back(mvecs[1]->at(i));
								motion_vectors->push_back(mvecs[2]->at(i));
								motion_vectors->push_back(mvecs[3]->at(i));
							}
							for (int i = 0; i < 4; i++) {
								free(mvecs[i]);
							}
						}
						delete frame_lowpassed; frame_lowpassed = NULL;
						if (frame_number > 0) delete previous_frame_lowpassed;
						previous_frame_lowpassed = new Frame(frame_lowpassed_a_final);
						print("blocked filtering: ...");
					}
					

					
					blocked_ds_dct_round(frame_lowpassed_a_final->Cr, frame_quant->Cr);

					blocked_ds_dct_round(frame_lowpassed_a_final->Cb, frame_quant->Cb);

					blocked_dct_round(frame_lowpassed_a_final->Y, frame_quant->Y);

					#pragma omp barrier	

					zigZagOrder(frame_quant->Y, frame_zigzag->Y);
					zigZagOrder(frame_quant->Cb, frame_zigzag->Cb);
					zigZagOrder(frame_quant->Cr, frame_zigzag->Cr);

					encode8x8(frame_zigzag->Y, frame_encode->Y);
					encode8x8(frame_zigzag->Cb, frame_encode->Cb);
					encode8x8(frame_zigzag->Cr, frame_encode->Cr);

					#pragma omp barrier	
					if(tid==0){
						dump_zigzag(frame_zigzag, "frame_zigzag", frame_number);



						Frame* frame_dc_diff = new Frame(1, (width / 8)*(height / 8), DCDIFF); //dealocate later

						dcDiff(frame_quant->Y, frame_dc_diff->Y);
						dcDiff(frame_quant->Cb, frame_dc_diff->Cb);
						dcDiff(frame_quant->Cr, frame_dc_diff->Cr);

						stream_frame(encoded_images[frame_number], frame_number, motion_vectors, frame_number - 1, frame_dc_diff, frame_encode);
						encoded_images_mutex[frame_number].unlock();
						//write_stream(stream_path, stream);

						delete frame_dc_diff;

						if (motion_vectors != NULL) {
							free(motion_vectors);
							motion_vectors = NULL;
						}

						writestats(frame_number, frame_number % i_frame_frequency, runtime);
					}
				}
			}
			delete frame_quant;
			delete frame_zigzag;
			delete frame_encode;
		}
	}

	// Cleanup
	for (int i = 0; i < 6; i++) {
		clReleaseMemObject(device_ptrs[i]);
	}
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseKernel(kernel[2]);
	clReleaseProgram(program);

	closeStats();
	/* Uncoment to prevent visual studio output window from closing */
	//system("pause");
    
	return 0;
}
 
 
int main(int args, char** argv){
	struct timeval starttime, endtime;
	gettimeofday(&starttime, NULL);
    encode();
	gettimeofday(&endtime, NULL);
	double runtime = double(endtime.tv_sec)*1000.0f + double(endtime.tv_usec) / 1000.0f - double(starttime.tv_sec)*1000.0f - double(starttime.tv_usec) / 1000.0f; //in ms  
	printf("Total runtime : %lf\n", runtime);
    return 0;
}