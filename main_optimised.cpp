/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

// #include <pthread.h>
#include <stdio.h>
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <iomanip>

using namespace std;

typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;	// DO NOT CHANGE!!
const int K=100000;	// DO NOT CHANGE!!

float logDataVSPrior_optimized(const float* datReal, const float* datImag, const float* priReal, const float* priImag, \
                                const float* ctf, const float* sigRcp, const int num, const float disturb0, float * real);

void *AllignedMalloc(size_t size, int aligned)//分配字节对齐的内存地址
{
    // aligned is a power of 2
    assert((aligned&(aligned - 1)) == 0);
    // 分配内存空间
    void *data = malloc(sizeof(void *)+aligned + size);
    // 地址对齐
    void **temp = (void **)data + 1;
    void **alignedData = (void **)(((size_t)temp + aligned - 1)&-aligned);
    // 保存原始内存地址
    alignedData[-1] = data;
    return alignedData;  // 被转换为一级指针
}

int main ( int argc, char *argv[] )
{
    /*************************
    * Init for MPI
    **************************/
    int MPI_thread_scheme, MPI_p_id, MPI_p_num;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &MPI_thread_scheme);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_p_id);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_p_num);

    float *all_result;
    float *disturb = (float*)AllignedMalloc(sizeof(float) * K, 64);
    float *result = (float*)AllignedMalloc(sizeof(float) * K, 64);
    float *real = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float *datReal = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float *datImag = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float *priReal = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float *priImag = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float *ctf = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float *sigRcp = (float*)AllignedMalloc(sizeof(float) * m / MPI_p_num + sizeof(float) * MPI_p_num, 64);
    float dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    ofstream fout;
    auto startTime = Clock::now();
    auto IOstartTime = Clock::now();
    MPI_Status status;

    /***************************
     * Read data from input.dat
     * *************************/
    float *datReal_0, *datImag_0, *priReal_0, *priImag_0, *ctf_0, *sigRcp_0;    
    ifstream fin;

    fin.open("input.dat");
    if(!fin.is_open()){
        cout << "Error opening file input.dat" << endl;
        exit(1);
    }
    int i=0;
    if(MPI_p_id == 0){
        datReal_0 = (float*)AllignedMalloc(sizeof(float) * m, 64);
        datImag_0 = (float*)AllignedMalloc(sizeof(float) * m, 64);
        priReal_0 = (float*)AllignedMalloc(sizeof(float) * m, 64);
        priImag_0 = (float*)AllignedMalloc(sizeof(float) * m, 64);
        ctf_0 = (float*)AllignedMalloc(sizeof(float) * m, 64);
        sigRcp_0 = (float*)AllignedMalloc(sizeof(float) * m, 64);
        while( !fin.eof() ) {
            fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
            datReal_0[i] = dat0;
            datImag_0[i] = dat1;
            priReal_0[i] = pri0;
            priImag_0[i] = pri1;
            ctf_0[i] = ctf0;
            sigRcp_0[i] = sigRcp0;
            i++;
            if(i == m) break;
        } 
    }
    fin.close();

    fin.open("K.dat");
    if(!fin.is_open()){
        cout << "Error opening file K.dat" << endl;
        exit(1);
    }
    i=0;
    while(!fin.eof()){
        fin >> disturb[i];
        i++;
        if(i == K) break;
    }
    fin.close();
    if(MPI_p_id == 0) all_result = (float*)AllignedMalloc(sizeof(float) * K, 64);

    /***************************
     * main computation is here
     * ************************/
    if(MPI_p_id == 0) startTime = Clock::now();

    //预处理与数据划分
    int mynum = m / MPI_p_num;
    int offset = mynum * MPI_p_id;
    if(MPI_p_id == MPI_p_num - 1) mynum = m - offset;
    int scatter_num[MPI_p_num];
    for(int i=0; i<MPI_p_num; i++){
        scatter_num[i] = mynum;
    }
    scatter_num[MPI_p_num - 1] = m - m / MPI_p_num * (MPI_p_num - 1);
    int scatter_offset[MPI_p_num];
    for(int i=0; i<MPI_p_num; i++){
        scatter_offset[i] = m / MPI_p_num * i;
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Scatterv(datReal_0, scatter_num, scatter_offset, MPI_FLOAT, datReal, mynum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(datImag_0, scatter_num, scatter_offset, MPI_FLOAT, datImag, mynum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(priReal_0, scatter_num, scatter_offset, MPI_FLOAT, priReal, mynum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(priImag_0, scatter_num, scatter_offset, MPI_FLOAT, priImag, mynum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(ctf_0, scatter_num, scatter_offset, MPI_FLOAT, ctf, mynum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(sigRcp_0, scatter_num, scatter_offset, MPI_FLOAT, sigRcp, mynum, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for(unsigned int t = 0; t < K; t++){
        result[t] = logDataVSPrior_optimized(datReal, datImag, priReal, priImag, ctf,\
                                            sigRcp, mynum, disturb[t], real);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(result, all_result, K, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(MPI_p_id == 0){
        FILE *fout_c;
        fout_c = fopen("result.dat", "w");
        setvbuf(fout_c, (char *)malloc(2 * K * sizeof(double)), _IOFBF, 2 * K * sizeof(double));
        if(fout_c == NULL){
            cout << "Error opening file for result" << endl;
            exit(1);
        }
        for(int t=0; t<K; t++)
        {
            fprintf(fout_c, "%d: %5e\n", t+1, all_result[t]);
        }
        fclose(fout_c);      

        auto endTime = Clock::now(); 
        auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        cout << "Computing time=" << compTime.count() << endl;
    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}

float logDataVSPrior_optimized(const float* datReal,  const float* datImag,  const float* priReal,  const float* priImag, \
                               const float* ctf,  const float* sigRcp, const int num, const float disturb0,  float* real)
{
    float result = 0.0;
    __m512 realVec, imagVec, datRealVec, datImagVec, ctfVec , priRealVec, priImagVec, sigRcpVec, resultVec, sumVec, disturbVec, finalVec;

    sumVec = _mm512_set1_ps(0.0);
    disturbVec = _mm512_set1_ps(disturb0);
    for (int i = 0; i < num; i+=16){
        //norm对实部计算
        datRealVec = _mm512_load_ps(datReal + i);
        ctfVec = _mm512_load_ps(ctf + i);
        ctfVec = _mm512_mul_ps(ctfVec, disturbVec);
        priRealVec = _mm512_load_ps(priReal + i);
        realVec = _mm512_fnmadd_ps(ctfVec, priRealVec, datRealVec);
        //norm对虚部计算
        datImagVec = _mm512_load_ps(datImag + i);
        priImagVec = _mm512_load_ps(priImag + i);        
        imagVec = _mm512_fnmadd_ps(ctfVec, priImagVec, datImagVec);
        sigRcpVec = _mm512_load_ps(sigRcp + i);
        //平方和
        resultVec = _mm512_fmadd_ps(realVec, realVec, _mm512_mul_ps(imagVec, imagVec));
        //乘系数并写存
        // sumVec = _mm512_fmadd_ps(resultVec, sigRcpVec, sumVec);
        sumVec = _mm512_mul_ps(resultVec, sigRcpVec);
        _mm512_store_ps(real + i, sumVec);
    }

    //16无法整除部分处理
    datRealVec = _mm512_load_ps(datReal + num - 16);
    ctfVec = _mm512_load_ps(ctf + num - 16);
    ctfVec = _mm512_mul_ps(ctfVec, disturbVec);
    priRealVec = _mm512_load_ps(priReal + num - 16);
    realVec = _mm512_fnmadd_ps(ctfVec, priRealVec, datRealVec);
    //norm对虚部计算
    datImagVec = _mm512_load_ps(datImag + num - 16);
    priImagVec = _mm512_load_ps(priImag + num - 16);        
    imagVec = _mm512_fnmadd_ps(ctfVec, priImagVec, datImagVec);
    sigRcpVec = _mm512_load_ps(sigRcp + num - 16);
    //平方和
    resultVec = _mm512_fmadd_ps(realVec, realVec, _mm512_mul_ps(imagVec, imagVec));
    //乘系数并写存
    sumVec = _mm512_mul_ps(resultVec, sigRcpVec);
    _mm512_store_ps(real + num - 16, sumVec);

    //累加最终结果
    for(int i=0; i<num; i++){
        result += real[i];
    }
    // result = _mm512_reduce_add_ps(sumVec);
    return result;
}
