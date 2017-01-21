from   util            import *
import numpy           as np
import pycuda.driver   as cuda
import pycuda.autoinit
from   pycuda.compiler import SourceModule

THREAD = 1

def CalcFitness(ppl, weight, AllowableAmount, eliteSize):
    population  = ppl.population[eliteSize:]
    gene        = [g.ind for g in population]
    gene        = np.asarray(gene, dtype=np.int)
    #print gene
    #print weight
    #print AllowableAmount
    gene_gpu    = cuda.mem_alloc(gene.nbytes)
    w           = np.asarray(weight, dtype=np.float32)
    w_gpu       = cuda.mem_alloc(w.nbytes)
    amount      = np.asarray(AllowableAmount, dtype=np.float32)
    amount_gpu  = cuda.mem_alloc(amount.nbytes)
    fit         = np.zeros(len(gene), dtype=np.float32)
    fit_gpu     = cuda.mem_alloc(fit.nbytes)
    cuda.memcpy_htod(gene_gpu,    gene)
    cuda.memcpy_htod(w_gpu,       w)
    cuda.memcpy_htod(amount_gpu,  amount)
    cuda.memcpy_htod(fit_gpu,     fit)
    mod = SourceModule("""
    #include <stdio.h>
    #include <math.h>

    #define THREADSIZE ({0})
    #define GENESIZE   ({1})
    #define SUM_WEIGHT ({2})
    """.format(THREAD, len(gene[0]), sum(w)) +
    """

    __global__ void Calc
    (
    int *gene, 
    float *w,
    float *amount,
    float *fit
    )
    {
        __shared__ float gweight[1];
        __shared__ int g[GENESIZE];

        int i=0;
        int j=0;
        //printf("genesize=%d\\n", GENESIZE);
        //printf("sum_weight=%f\\n", SUM_WEIGHT);
        //printf("block%d:thread%d:gene=%d\\n", blockIdx.x, threadIdx.x, gene[blockIdx.x*GENESIZE*2+2]);
        g[0] = gene[i+(GENESIZE*blockIdx.x*2)];
        g[1] = gene[i+2+(GENESIZE*blockIdx.x*2)];
        for (i=2;i<GENESIZE;i++) {
            g[i] = gene[i+i+(GENESIZE*blockIdx.x*2)];
        }
        __syncthreads();
        //printf("block%d:thread%d:g=%d\\n", blockIdx.x, threadIdx.x, g[1]);
        int sample = 0;
        
        for (i=0;i<GENESIZE;i++) {
            gweight[0] = 0;
            sample = g[i];
            if (sample!=-1) {
                //printf("block%d:thread%d:sample=%d\\n", blockIdx.x, threadIdx.x, sample);
                for (j=0;j<GENESIZE;j++) {
                    if (sample==g[j]) {
                        //printf("block%d:thread%d:w=%f\\n", blockIdx.x, threadIdx.x, w[i]);
                        gweight[0] += w[j];
                        g[j] = -1;
                    }
                }
                //printf("block%d:thread%d:gweight=%f\\n", blockIdx.x, threadIdx.x, gweight[0]);
                float a    = amount[0];
                //printf("block%d:thread%d:a=%f\\n", blockIdx.x, threadIdx.x, a);
                float loss = (a-gweight[0])*(a-gweight[0]);
                //printf("[before]block%d:thread%d:loss=%f\\n", blockIdx.x, threadIdx.x, loss);
                if (loss<0 || loss>0) {
                    loss += SUM_WEIGHT;
                }
                //printf("[after]block%d:thread%d:loss=%f\\n", blockIdx.x, threadIdx.x, loss);
                fit[blockIdx.x]+=fabsf(loss);
                //printf("block%d:thread%d:fit=%f\\n", blockIdx.x, threadIdx.x, fit[blockIdx.x]);
            }
        }
    }
    """)
    sbytes = fit.nbytes+512
    func = mod.get_function("Calc")
    func(gene_gpu, w_gpu, amount_gpu, fit_gpu, grid=(len(population),1), block=(THREAD,1,1), shared=sbytes)
    result = np.zeros(len(gene), dtype=np.float32);
    cuda.memcpy_dtoh(result, fit_gpu)
    #print result
    for i in range(len(population)):
        fit = 0.0
        cur = i
        population[i].SetFitness(result[i])
        #print result[i]
