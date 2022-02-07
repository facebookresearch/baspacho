# Benchmarking

The tests below have been executed on a ThinkStationP720 equipped with
- RAM: 128Gb
- CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
- GPU: Quadro RTX 5000

## Test types

Random sparse matrices are generating from structured random graphs (one parameter per vertex,
there is a non-empty block for each edge connecting two vertices - similarly to the block-sparse
matrix coming from a factor graph). The graph types we consider are:
- FLAT: any two vertices are connected with probability 'fill'
- FLAT+SCHUR: like flat, plus a set of 'schursize' type parameters that can be eliminated independently
  is added with 'schurfill' probability of having a random connection to any existing vertices.
- GRID: a 2-dimensional grid is generated, vertices are connected up to distance 'conn' with
  probability 'fill'
- MERI: a certain number of "tracks" is generated, they are a sequence of len 'size' of parameters, each
  having a probability 'fill' of being connected to the "neighboring" parameters up to distance 'band'
  inside the tracks. They are joined into a non-trivial topology as follows: 'n' tracks connect two poles,
  and each pole has an additional set of 'hairs' track departing from it.
For each topology + settings 5 problems are generated at random and tested.

## Factor (OpenBLAS/Cuda 11.5)
Command: `cmake --build build -v -- -j16 && build/bench -B 1_CHOLMOD -O factor`
```
Problem type: 10_FLAT_size=1000_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.436s,             0.429s,             0.419s,             0.432s,             0.449s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.368s (-15.56%),   0.378s (-11.82%),   0.364s (-13.27%),   0.355s (-17.75%),   0.379s (-15.63%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    63.7ms (-85.39%),   53.3ms (-87.58%),   52.3ms (-87.53%),   52.1ms (-87.93%),   54.3ms (-87.91%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    50.0ms (-88.53%),   48.8ms (-88.64%),   46.7ms (-88.85%),   41.3ms (-90.44%),   47.7ms (-89.40%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    40.0ms (-90.82%),   40.1ms (-90.66%),   39.9ms (-90.48%),   39.3ms (-90.91%),   42.0ms (-90.65%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    42.2ms (-90.32%),   43.1ms (-89.95%),   44.5ms (-89.39%),   43.4ms (-89.94%),   47.3ms (-89.47%)    

Problem type: 11_FLAT_size=4000_fill=0.01_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    13.320s,            13.072s,            13.133s,            13.098s,            13.076s             
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    12.496s (-6.19%),   12.319s (-5.76%),   12.433s (-5.34%),   12.298s (-6.11%),   12.615s (-3.52%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    1.352s (-89.85%),   1.351s (-89.66%),   1.377s (-89.51%),   1.370s (-89.54%),   1.388s (-89.38%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    1.455s (-89.08%),   1.432s (-89.05%),   1.475s (-88.77%),   1.442s (-88.99%),   1.504s (-88.50%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    1.599s (-87.99%),   1.575s (-87.95%),   1.617s (-87.69%),   1.580s (-87.94%),   1.656s (-87.33%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    1.680s (-87.39%),   1.651s (-87.37%),   1.695s (-87.10%),   1.656s (-87.36%),   1.754s (-86.59%)    

Problem type: 12_FLAT_size=2000_fill=0.03_bsize=2-5
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    3.416s,             3.457s,             3.370s,             3.619s,             3.379s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    3.086s (-9.65%),    3.154s (-8.78%),    3.098s (-8.08%),    3.210s (-11.29%),   3.147s (-6.89%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.395s (-88.44%),   0.421s (-87.82%),   0.406s (-87.97%),   0.411s (-88.65%),   0.419s (-87.60%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.365s (-89.33%),   0.369s (-89.33%),   0.370s (-89.01%),   0.383s (-89.41%),   0.377s (-88.84%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    0.387s (-88.67%),   0.394s (-88.60%),   0.394s (-88.30%),   0.410s (-88.66%),   0.406s (-88.00%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    0.421s (-87.66%),   0.432s (-87.52%),   0.433s (-87.14%),   0.448s (-87.63%),   0.448s (-86.75%)    

Problem type: 20_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    1.913s,             1.908s,             1.909s,             1.907s,             1.920s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.549s (-71.30%),   0.547s (-71.33%),   0.550s (-71.21%),   0.545s (-71.40%),   0.550s (-71.37%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.117s (-93.87%),   0.117s (-93.86%),   0.117s (-93.86%),   0.117s (-93.85%),   0.115s (-94.00%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    99.5ms (-94.80%),   97.9ms (-94.87%),   98.3ms (-94.85%),   98.8ms (-94.82%),   0.100s (-94.77%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    98.9ms (-94.83%),   99.2ms (-94.80%),   98.7ms (-94.83%),   99.3ms (-94.79%),   98.6ms (-94.86%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    0.103s (-94.63%),   0.102s (-94.63%),   0.102s (-94.65%),   0.102s (-94.64%),   0.102s (-94.69%)    

Problem type: 21_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.2
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.426s,             0.427s,             0.420s,             0.429s,             0.419s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.360s (-15.57%),   0.359s (-15.91%),   0.352s (-16.27%),   0.367s (-14.56%),   0.354s (-15.62%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    52.5ms (-87.68%),   53.6ms (-87.46%),   52.7ms (-87.47%),   56.3ms (-86.89%),   53.5ms (-87.24%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    48.2ms (-88.70%),   48.7ms (-88.60%),   44.1ms (-89.52%),   49.6ms (-88.44%),   46.3ms (-88.96%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    39.1ms (-90.83%),   39.6ms (-90.72%),   38.8ms (-90.77%),   41.1ms (-90.42%),   39.1ms (-90.66%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    42.0ms (-90.14%),   43.5ms (-89.81%),   41.6ms (-90.10%),   44.5ms (-89.62%),   41.0ms (-90.21%)    

Problem type: 30_GRID_size=100x100_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.390s,             0.397s,             0.393s,             0.393s,             0.393s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.369s (-5.55%),    0.383s (-3.60%),    0.376s (-4.50%),    0.366s (-6.94%),    0.367s (-6.60%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    91.2ms (-76.64%),   91.3ms (-77.00%),   91.3ms (-76.79%),   92.1ms (-76.56%),   91.2ms (-76.81%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    46.1ms (-88.18%),   50.5ms (-87.29%),   52.2ms (-86.72%),   46.1ms (-88.27%),   52.0ms (-86.79%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    41.0ms (-89.50%),   38.0ms (-90.42%),   40.3ms (-89.75%),   40.4ms (-89.72%),   38.1ms (-90.31%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    38.0ms (-90.26%),   37.5ms (-90.56%),   38.2ms (-90.28%),   37.8ms (-90.38%),   37.1ms (-90.58%)    

Problem type: 31_GRID_size=150x150_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    1.960s,             1.987s,             1.991s,             2.002s,             2.040s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    1.106s (-43.57%),   1.118s (-43.75%),   1.089s (-45.30%),   1.094s (-45.37%),   1.097s (-46.24%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.242s (-87.67%),   0.234s (-88.23%),   0.243s (-87.82%),   0.229s (-88.58%),   0.224s (-89.01%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.128s (-93.46%),   0.130s (-93.46%),   0.129s (-93.54%),   0.130s (-93.52%),   0.130s (-93.64%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    0.115s (-94.14%),   0.115s (-94.22%),   0.115s (-94.23%),   0.115s (-94.25%),   0.115s (-94.37%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    0.115s (-94.13%),   0.114s (-94.24%),   0.115s (-94.22%),   0.115s (-94.27%),   0.115s (-94.38%)    

Problem type: 32_GRID_size=200x200_fill=0.25_conn=2_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    10.625s,            8.614s,             10.769s,            9.346s,             10.025s             
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    2.821s (-73.45%),   2.679s (-68.90%),   2.791s (-74.09%),   2.743s (-70.65%),   2.681s (-73.25%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.476s (-95.52%),   0.465s (-94.60%),   0.475s (-95.59%),   0.484s (-94.83%),   0.471s (-95.30%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.299s (-97.18%),   0.293s (-96.60%),   0.304s (-97.18%),   0.295s (-96.85%),   0.289s (-97.11%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    0.277s (-97.39%),   0.272s (-96.84%),   0.282s (-97.39%),   0.273s (-97.08%),   0.267s (-97.34%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    0.286s (-97.31%),   0.278s (-96.77%),   0.290s (-97.30%),   0.281s (-96.99%),   0.274s (-97.27%)    

Problem type: 33_GRID_size=200x200_fill=0.05_conn=3_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.964s,             1.094s,             1.058s,             0.960s,             1.088s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.476s (-50.58%),   0.470s (-57.00%),   0.537s (-49.23%),   0.469s (-51.14%),   0.548s (-49.69%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.148s (-84.68%),   0.138s (-87.39%),   0.158s (-85.09%),   0.142s (-85.22%),   0.157s (-85.56%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    63.2ms (-93.44%),   63.3ms (-94.21%),   70.8ms (-93.31%),   62.7ms (-93.47%),   69.3ms (-93.63%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    52.3ms (-94.57%),   53.4ms (-95.12%),   61.1ms (-94.22%),   54.1ms (-94.36%),   60.9ms (-94.41%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    47.7ms (-95.05%),   47.7ms (-95.64%),   55.7ms (-94.74%),   48.9ms (-94.90%),   55.5ms (-94.90%)    

Problem type: 40_MERI_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.654s,             0.639s,             0.644s,             0.660s,             0.634s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.537s (-17.91%),   0.538s (-15.77%),   0.525s (-18.36%),   0.543s (-17.85%),   0.517s (-18.43%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.189s (-71.16%),   0.179s (-71.91%),   0.193s (-70.08%),   0.187s (-71.63%),   0.194s (-69.32%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    80.5ms (-87.69%),   78.6ms (-87.69%),   78.6ms (-87.79%),   81.3ms (-87.68%),   78.6ms (-87.60%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    64.8ms (-90.10%),   61.7ms (-90.33%),   61.7ms (-90.41%),   64.0ms (-90.31%),   62.0ms (-90.22%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    53.6ms (-91.81%),   51.9ms (-91.88%),   53.6ms (-91.66%),   55.6ms (-91.58%),   53.0ms (-91.64%)    

Problem type: 41_MERI_size=1500_n=7_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    1.009s,             1.014s,             1.001s,             1.015s,             1.004s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.821s (-18.65%),   0.840s (-17.13%),   0.813s (-18.77%),   0.837s (-17.57%),   0.823s (-17.97%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.274s (-72.81%),   0.289s (-71.45%),   0.284s (-71.68%),   0.289s (-71.50%),   0.290s (-71.06%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.128s (-87.32%),   0.130s (-87.16%),   0.131s (-86.88%),   0.130s (-87.20%),   0.131s (-86.90%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    96.3ms (-90.46%),   95.5ms (-90.58%),   95.6ms (-90.46%),   98.4ms (-90.30%),   95.4ms (-90.50%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    82.1ms (-91.87%),   82.1ms (-91.90%),   80.8ms (-91.93%),   85.3ms (-91.59%),   81.9ms (-91.84%)
```

## Factor (Intel-MKL)
Command: `cmake --build build -v -- -j16 && build/bench -B 1_CHOLMOD -O factor -S ^2`
```
Problem type: 10_FLAT_size=1000_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.200s,             67.0ms,             63.1ms,             62.0ms,             63.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    66.1ms (-67.00%),   50.2ms (-24.97%),   66.4ms (+5.24%),    49.3ms (-20.54%),   53.3ms (-16.10%)    

Problem type: 11_FLAT_size=4000_fill=0.01_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    1.311s,             1.288s,             1.276s,             1.243s,             1.311s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    1.575s (+20.19%),   1.588s (+23.26%),   1.560s (+22.21%),   1.243s (-0.04%),    1.600s (+22.04%)    

Problem type: 12_FLAT_size=2000_fill=0.03_bsize=2-5
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.411s,             0.414s,             0.392s,             0.410s,             0.395s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.471s (+14.75%),   0.475s (+14.82%),   0.462s (+17.96%),   0.501s (+22.44%),   0.391s (-1.11%)     

Problem type: 20_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    1.709s,             1.816s,             1.744s,             1.729s,             1.669s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.209s (-87.75%),   0.207s (-88.58%),   0.232s (-86.72%),   0.205s (-88.12%),   0.204s (-87.80%)    

Problem type: 21_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.2
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    82.6ms,             76.9ms,             74.9ms,             76.0ms,             77.8ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    46.4ms (-43.85%),   46.8ms (-39.22%),   49.4ms (-33.99%),   50.2ms (-33.91%),   47.8ms (-38.52%)    

Problem type: 30_GRID_size=100x100_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.237s,             0.235s,             0.244s,             0.235s,             0.238s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.195s (-17.43%),   0.204s (-13.13%),   0.185s (-24.11%),   0.203s (-13.29%),   0.203s (-14.74%)    

Problem type: 31_GRID_size=150x150_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.724s,             0.736s,             0.735s,             0.722s,             0.786s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.551s (-23.85%),   0.551s (-25.18%),   0.565s (-23.11%),   0.558s (-22.70%),   0.585s (-25.58%)    

Problem type: 32_GRID_size=200x200_fill=0.25_conn=2_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    1.798s,             1.746s,             2.060s,             1.839s,             1.908s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    1.247s (-30.65%),   1.238s (-29.06%),   1.185s (-42.46%),   1.260s (-31.49%),   1.296s (-32.05%)    

Problem type: 33_GRID_size=200x200_fill=0.05_conn=3_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.373s,             0.449s,             0.381s,             0.366s,             0.380s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.180s (-51.81%),   0.165s (-63.25%),   0.177s (-53.49%),   0.156s (-57.25%),   0.179s (-52.80%)    

Problem type: 40_MERI_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.335s,             0.364s,             0.404s,             0.332s,             0.340s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.148s (-55.66%),   0.145s (-60.21%),   0.160s (-60.48%),   0.149s (-55.20%),   0.147s (-56.63%)    

Problem type: 41_MERI_size=1500_n=7_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: factor
- 1_CHOLMOD (basis for comparison):
    0.604s,             0.573s,             0.549s,             0.531s,             0.531s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.236s (-60.98%),   0.221s (-61.45%),   0.214s (-60.95%),   0.215s (-59.61%),   0.213s (-59.81%)
```

## Solve (OpenBLAS/Cuda 11.5, nRHS = 1, 2, 10)
Command: `cmake --build build -v -- -j16 && build/bench -B 1_CHOLMOD -O solve-1,solve-2,solve-10`
```
Problem type: 10_FLAT_size=1000_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    8.3ms,              8.0ms,              7.4ms,              7.3ms,              7.4ms               
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    18.4ms (+122.51%),  19.8ms (+147.03%),  16.3ms (+120.15%),  15.8ms (+115.65%),  15.4ms (+107.82%)   
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    2.6ms (-68.77%),    2.1ms (-73.59%),    2.3ms (-69.22%),    2.1ms (-71.02%),    2.3ms (-68.80%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    4.4ms (-47.34%),    4.1ms (-49.31%),    3.9ms (-47.36%),    4.0ms (-45.27%),    4.2ms (-42.92%)     
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    2.3ms (-72.11%),    2.1ms (-73.15%),    2.1ms (-72.20%),    2.0ms (-73.39%),    2.3ms (-69.26%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    1.3ms (-84.72%),    1.2ms (-85.19%),    1.1ms (-84.74%),    1.1ms (-84.72%),    1.3ms (-82.95%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    23.9ms,             25.2ms,             23.7ms,             24.1ms,             25.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    25.7ms (+7.33%),    24.3ms (-3.46%),    24.1ms (+1.43%),    24.0ms (-0.58%),    24.7ms (-2.43%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    15.4ms (-35.63%),   11.7ms (-53.39%),   13.7ms (-42.20%),   11.9ms (-50.58%),   12.8ms (-49.56%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    14.5ms (-39.37%),   14.0ms (-44.24%),   13.6ms (-42.73%),   13.2ms (-45.11%),   15.5ms (-38.61%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    8.6ms (-63.90%),    8.3ms (-67.12%),    7.9ms (-66.74%),    7.8ms (-67.87%),    9.0ms (-64.63%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    6.0ms (-74.87%),    5.7ms (-77.35%),    5.5ms (-76.78%),    5.4ms (-77.62%),    6.1ms (-75.94%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    18.7ms,             19.9ms,             18.8ms,             18.5ms,             18.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    19.7ms (+5.34%),    17.8ms (-10.59%),   17.3ms (-7.92%),    17.4ms (-5.92%),    17.2ms (-7.41%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    8.1ms (-56.67%),    6.5ms (-67.39%),    5.8ms (-69.01%),    5.7ms (-69.04%),    6.9ms (-62.95%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    14.4ms (-23.00%),   14.2ms (-28.62%),   13.2ms (-30.01%),   12.6ms (-31.82%),   15.3ms (-17.84%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    8.5ms (-54.39%),    8.2ms (-58.93%),    7.6ms (-59.60%),    7.3ms (-60.18%),    8.8ms (-52.66%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    5.8ms (-69.06%),    5.6ms (-72.14%),    5.2ms (-72.12%),    5.1ms (-72.13%),    6.0ms (-67.57%)     

Problem type: 11_FLAT_size=4000_fill=0.01_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    94.1ms,             78.0ms,             88.7ms,             87.4ms,             78.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.201s (+113.82%),  0.200s (+155.97%),  0.204s (+130.15%),  0.204s (+133.57%),  0.200s (+155.76%)   
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    11.4ms (-87.94%),   11.3ms (-85.52%),   11.9ms (-86.63%),   11.5ms (-86.89%),   11.1ms (-85.80%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    15.4ms (-83.63%),   16.3ms (-79.06%),   16.2ms (-81.75%),   16.0ms (-81.72%),   15.4ms (-80.29%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    9.2ms (-90.21%),    9.9ms (-87.30%),    9.1ms (-89.69%),    9.1ms (-89.65%),    9.2ms (-88.24%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    6.0ms (-93.62%),    5.9ms (-92.42%),    5.8ms (-93.41%),    5.8ms (-93.32%),    6.0ms (-92.32%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.306s,             0.274s,             0.308s,             0.301s,             0.279s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.295s (-3.69%),    0.294s (+7.35%),    0.298s (-3.28%),    0.296s (-1.46%),    0.292s (+4.48%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    80.1ms (-73.84%),   80.3ms (-70.68%),   78.1ms (-74.65%),   78.1ms (-74.03%),   76.7ms (-72.54%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.112s (-63.40%),   0.115s (-57.97%),   0.116s (-62.25%),   0.115s (-61.67%),   0.113s (-59.62%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    77.9ms (-74.56%),   82.0ms (-70.03%),   78.0ms (-74.68%),   76.5ms (-74.56%),   77.1ms (-72.41%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    57.8ms (-81.13%),   56.6ms (-79.33%),   56.9ms (-81.54%),   56.1ms (-81.34%),   59.1ms (-78.85%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    0.252s,             0.224s,             0.257s,             0.249s,             0.228s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.222s (-11.91%),   0.222s (-0.56%),    0.225s (-12.29%),   0.225s (-9.59%),    0.223s (-2.16%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    53.5ms (-78.81%),   53.1ms (-76.27%),   52.2ms (-79.69%),   52.3ms (-78.99%),   51.2ms (-77.50%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.109s (-56.86%),   0.112s (-50.05%),   0.114s (-55.69%),   0.112s (-54.92%),   0.110s (-51.68%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    76.7ms (-69.62%),   79.0ms (-64.66%),   74.2ms (-71.12%),   73.6ms (-70.47%),   76.2ms (-66.52%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    54.0ms (-78.61%),   53.3ms (-76.18%),   53.1ms (-79.33%),   52.5ms (-78.93%),   54.0ms (-76.27%)    

Problem type: 12_FLAT_size=2000_fill=0.03_bsize=2-5
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    37.9ms,             37.4ms,             37.9ms,             37.7ms,             38.8ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    79.0ms (+108.63%),  80.6ms (+115.66%),  80.6ms (+112.46%),  82.9ms (+119.87%),  78.5ms (+102.27%)   
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    7.6ms (-80.05%),    7.9ms (-78.83%),    7.3ms (-80.86%),    7.4ms (-80.24%),    7.6ms (-80.47%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    9.3ms (-75.52%),    9.3ms (-75.24%),    9.5ms (-74.94%),    9.4ms (-75.19%),    9.4ms (-75.76%)     
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    5.2ms (-86.38%),    5.1ms (-86.24%),    5.2ms (-86.30%),    5.2ms (-86.09%),    5.4ms (-85.97%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    3.1ms (-91.93%),    3.2ms (-91.57%),    3.1ms (-91.85%),    3.1ms (-91.68%),    3.2ms (-91.81%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.115s,             0.107s,             0.118s,             0.120s,             0.121s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.117s (+1.93%),    0.121s (+13.25%),   0.120s (+2.14%),    0.122s (+2.27%),    0.124s (+2.48%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    46.4ms (-59.59%),   49.2ms (-53.99%),   46.8ms (-60.23%),   49.2ms (-58.85%),   48.6ms (-59.98%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    49.9ms (-56.59%),   50.1ms (-53.19%),   52.6ms (-55.26%),   51.3ms (-57.09%),   55.4ms (-54.37%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    32.0ms (-72.14%),   31.8ms (-70.26%),   33.3ms (-71.71%),   34.1ms (-71.46%),   35.4ms (-70.81%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    23.6ms (-79.43%),   23.9ms (-77.68%),   24.2ms (-79.44%),   24.4ms (-79.59%),   26.1ms (-78.52%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    95.0ms,             84.2ms,             94.0ms,             97.6ms,             98.8ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    88.5ms (-6.89%),    91.6ms (+8.82%),    89.4ms (-4.83%),    92.1ms (-5.55%),    88.0ms (-10.98%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    25.0ms (-73.67%),   25.3ms (-69.90%),   25.1ms (-73.32%),   25.6ms (-73.75%),   25.5ms (-74.18%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    46.9ms (-50.68%),   47.1ms (-44.07%),   50.0ms (-46.78%),   47.9ms (-50.87%),   51.0ms (-48.45%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    30.0ms (-68.46%),   30.2ms (-64.19%),   31.4ms (-66.59%),   33.0ms (-66.16%),   33.1ms (-66.52%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    22.1ms (-76.79%),   23.4ms (-72.17%),   22.7ms (-75.84%),   22.7ms (-76.76%),   24.7ms (-74.97%)    

Problem type: 20_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    54.2ms,             56.0ms,             54.1ms,             54.5ms,             54.1ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    50.8ms (-6.34%),    49.3ms (-12.03%),   52.0ms (-3.91%),    49.6ms (-9.03%),    51.0ms (-5.71%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    4.8ms (-91.06%),    4.8ms (-91.40%),    5.2ms (-90.35%),    4.7ms (-91.41%),    5.6ms (-89.68%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    6.1ms (-88.84%),    5.8ms (-89.63%),    5.8ms (-89.25%),    6.3ms (-88.38%),    6.1ms (-88.80%)     
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    4.5ms (-91.79%),    4.6ms (-91.82%),    4.6ms (-91.59%),    4.5ms (-91.82%),    4.5ms (-91.65%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    4.6ms (-91.43%),    4.5ms (-91.91%),    4.9ms (-90.96%),    4.6ms (-91.51%),    4.5ms (-91.67%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.252s,             0.261s,             0.251s,             0.255s,             0.247s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.114s (-54.67%),   0.109s (-58.46%),   0.115s (-54.00%),   0.112s (-56.00%),   0.112s (-54.58%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    33.1ms (-86.84%),   31.8ms (-87.83%),   37.2ms (-85.17%),   32.2ms (-87.37%),   33.5ms (-86.42%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    52.7ms (-79.09%),   47.7ms (-81.75%),   48.2ms (-80.77%),   50.8ms (-80.08%),   49.0ms (-80.14%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    93.1ms (-63.04%),   88.7ms (-66.04%),   88.5ms (-64.70%),   88.6ms (-65.24%),   88.9ms (-63.97%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    0.156s (-38.02%),   0.154s (-41.19%),   0.163s (-35.06%),   0.153s (-39.94%),   0.153s (-38.03%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    0.119s,             0.120s,             0.116s,             0.116s,             0.120s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    58.4ms (-50.76%),   60.3ms (-49.74%),   60.1ms (-48.25%),   55.6ms (-52.17%),   59.2ms (-50.77%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    9.5ms (-92.00%),    10.1ms (-91.62%),   9.8ms (-91.58%),    9.1ms (-92.18%),    9.1ms (-92.41%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    13.4ms (-88.69%),   13.2ms (-89.03%),   13.1ms (-88.75%),   14.0ms (-87.92%),   13.4ms (-88.86%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    13.0ms (-89.06%),   12.2ms (-89.84%),   12.3ms (-89.41%),   12.5ms (-89.28%),   12.4ms (-89.68%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    17.5ms (-85.20%),   16.7ms (-86.04%),   18.0ms (-84.51%),   17.0ms (-85.43%),   16.9ms (-85.94%)    

Problem type: 21_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.2
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    8.7ms,              8.5ms,              8.8ms,              8.7ms,              8.6ms               
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    16.2ms (+85.52%),   16.0ms (+88.12%),   16.3ms (+85.32%),   16.0ms (+83.63%),   16.5ms (+91.82%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    2.2ms (-74.75%),    2.1ms (-74.96%),    2.9ms (-67.00%),    2.4ms (-72.23%),    2.5ms (-71.06%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    3.9ms (-54.87%),    4.0ms (-52.74%),    4.3ms (-50.68%),    3.9ms (-55.16%),    4.0ms (-53.38%)     
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    2.0ms (-76.79%),    2.1ms (-75.20%),    2.3ms (-73.99%),    2.2ms (-74.56%),    2.4ms (-72.10%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    1.3ms (-85.35%),    1.2ms (-85.64%),    1.3ms (-84.95%),    1.2ms (-85.80%),    1.2ms (-86.04%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    27.5ms,             27.4ms,             28.1ms,             29.5ms,             29.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    25.9ms (-5.67%),    25.6ms (-6.29%),    25.6ms (-8.88%),    25.6ms (-13.20%),   26.0ms (-11.90%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    12.2ms (-55.79%),   12.3ms (-54.90%),   14.4ms (-48.72%),   13.7ms (-53.51%),   14.7ms (-49.98%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    14.0ms (-49.11%),   13.7ms (-49.86%),   15.3ms (-45.45%),   15.0ms (-49.26%),   14.6ms (-50.59%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    8.1ms (-70.41%),    8.4ms (-69.37%),    9.1ms (-67.68%),    8.8ms (-70.28%),    8.6ms (-70.86%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    5.8ms (-78.76%),    6.1ms (-77.54%),    6.4ms (-77.11%),    6.1ms (-79.45%),    6.0ms (-79.57%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    19.6ms,             19.2ms,             20.8ms,             20.2ms,             20.4ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    18.3ms (-6.69%),    18.3ms (-4.85%),    18.0ms (-13.33%),   17.9ms (-11.43%),   18.2ms (-10.42%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    6.1ms (-68.85%),    6.3ms (-67.42%),    7.6ms (-63.33%),    6.6ms (-67.49%),    7.3ms (-64.27%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    13.0ms (-33.65%),   13.2ms (-31.36%),   14.6ms (-29.43%),   14.3ms (-29.31%),   14.2ms (-30.22%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    7.8ms (-60.44%),    7.9ms (-58.96%),    8.6ms (-58.54%),    8.5ms (-57.71%),    8.4ms (-58.88%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    5.5ms (-72.09%),    5.7ms (-70.27%),    5.9ms (-71.78%),    5.7ms (-71.80%),    5.7ms (-71.94%)     

Problem type: 30_GRID_size=100x100_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    21.9ms,             22.2ms,             22.2ms,             21.8ms,             21.8ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    36.9ms (+68.57%),   36.5ms (+64.78%),   36.6ms (+65.30%),   37.4ms (+71.45%),   35.3ms (+61.59%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    31.3ms (+43.00%),   32.7ms (+47.60%),   29.1ms (+31.24%),   31.5ms (+44.34%),   31.2ms (+42.96%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    17.3ms (-20.96%),   17.3ms (-21.82%),   17.3ms (-21.91%),   17.3ms (-20.49%),   17.3ms (-20.49%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    8.8ms (-59.88%),    8.7ms (-60.95%),    8.7ms (-60.95%),    8.7ms (-60.25%),    8.7ms (-59.94%)     
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    4.6ms (-79.21%),    4.5ms (-79.53%),    4.5ms (-79.71%),    4.5ms (-79.35%),    4.5ms (-79.34%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    56.9ms,             58.5ms,             58.1ms,             56.4ms,             56.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    87.4ms (+53.68%),   87.1ms (+48.96%),   85.4ms (+46.82%),   87.1ms (+54.37%),   85.8ms (+51.86%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.118s (+108.19%),  0.120s (+105.89%),  0.119s (+104.31%),  0.119s (+111.38%),  0.119s (+110.68%)   
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    47.5ms (-16.57%),   46.9ms (-19.81%),   46.7ms (-19.64%),   46.6ms (-17.31%),   46.6ms (-17.44%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    26.8ms (-52.89%),   26.4ms (-54.82%),   26.2ms (-54.88%),   26.1ms (-53.69%),   26.1ms (-53.81%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    20.7ms (-63.65%),   20.7ms (-64.54%),   20.5ms (-64.66%),   20.7ms (-63.27%),   20.5ms (-63.76%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    30.4ms,             31.0ms,             30.8ms,             31.2ms,             31.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    43.6ms (+43.51%),   42.7ms (+37.61%),   43.2ms (+40.31%),   43.8ms (+40.48%),   43.7ms (+41.17%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    47.6ms (+56.43%),   45.1ms (+45.61%),   45.7ms (+48.30%),   45.0ms (+44.34%),   46.4ms (+49.72%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    30.3ms (-0.18%),    30.5ms (-1.62%),    30.1ms (-2.37%),    30.0ms (-3.87%),    31.0ms (+0.03%)     
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    16.1ms (-46.90%),   16.0ms (-48.43%),   16.1ms (-47.75%),   16.3ms (-47.58%),   16.3ms (-47.20%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    9.8ms (-67.90%),    9.8ms (-68.43%),    9.8ms (-68.23%),    9.9ms (-68.24%),    9.9ms (-67.89%)     

Problem type: 31_GRID_size=150x150_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    56.1ms,             0.192s,             57.1ms,             56.6ms,             56.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    85.4ms (+52.18%),   85.5ms (-55.40%),   82.1ms (+43.78%),   83.0ms (+46.64%),   83.6ms (+47.81%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    51.4ms (-8.45%),    52.2ms (-72.77%),   49.9ms (-12.62%),   50.1ms (-11.48%),   51.2ms (-9.36%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    31.7ms (-43.45%),   31.6ms (-83.50%),   34.7ms (-39.31%),   33.3ms (-41.12%),   33.2ms (-41.29%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    16.2ms (-71.15%),   16.9ms (-91.16%),   17.0ms (-70.24%),   16.9ms (-70.11%),   16.7ms (-70.42%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    9.0ms (-83.99%),    9.0ms (-95.32%),    9.0ms (-84.24%),    9.1ms (-83.93%),    9.2ms (-83.72%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.155s,             0.139s,             0.155s,             0.151s,             0.152s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.197s (+27.08%),   0.191s (+37.67%),   0.190s (+22.55%),   0.194s (+28.55%),   0.202s (+33.16%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.226s (+45.83%),   0.221s (+58.93%),   0.219s (+41.60%),   0.221s (+46.60%),   0.223s (+47.11%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    95.1ms (-38.62%),   98.3ms (-29.20%),   99.1ms (-35.99%),   99.0ms (-34.46%),   97.9ms (-35.37%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    61.5ms (-60.26%),   64.5ms (-53.54%),   64.9ms (-58.06%),   63.8ms (-57.73%),   61.6ms (-59.35%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    64.9ms (-58.08%),   64.9ms (-53.29%),   65.0ms (-58.00%),   65.1ms (-56.91%),   65.0ms (-57.13%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    81.7ms,             71.0ms,             81.5ms,             80.2ms,             81.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.104s (+27.83%),   0.102s (+43.24%),   0.102s (+24.98%),   0.102s (+27.09%),   0.101s (+24.58%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    82.7ms (+1.19%),    82.2ms (+15.75%),   83.7ms (+2.66%),    87.3ms (+8.89%),    82.5ms (+1.91%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    59.2ms (-27.53%),   58.9ms (-17.07%),   61.8ms (-24.15%),   61.8ms (-22.98%),   61.4ms (-24.12%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    33.3ms (-59.22%),   33.6ms (-52.68%),   35.0ms (-57.11%),   35.0ms (-56.30%),   34.6ms (-57.22%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    22.7ms (-72.20%),   22.6ms (-68.16%),   23.0ms (-71.76%),   22.5ms (-71.89%),   22.5ms (-72.23%)    

Problem type: 32_GRID_size=200x200_fill=0.25_conn=2_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    0.111s,             93.7ms,             97.1ms,             98.1ms,             94.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.161s (+45.01%),   0.162s (+72.28%),   0.166s (+70.44%),   0.179s (+82.85%),   0.156s (+65.12%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.116s (+4.24%),    0.123s (+31.20%),   0.121s (+25.08%),   0.120s (+22.63%),   0.112s (+18.48%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    54.1ms (-51.41%),   56.2ms (-40.02%),   56.0ms (-42.33%),   55.4ms (-43.50%),   56.9ms (-39.66%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    29.7ms (-73.26%),   30.5ms (-67.51%),   28.7ms (-70.45%),   28.2ms (-71.23%),   27.9ms (-70.41%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    16.1ms (-85.53%),   16.4ms (-82.47%),   16.2ms (-83.33%),   17.1ms (-82.52%),   16.9ms (-82.11%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.341s,             0.299s,             0.319s,             0.322s,             0.310s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.351s (+2.92%),    0.344s (+15.16%),   0.438s (+37.37%),   0.374s (+16.32%),   0.346s (+11.40%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.527s (+54.58%),   0.558s (+86.67%),   0.560s (+75.59%),   0.542s (+68.49%),   0.517s (+66.72%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.201s (-40.89%),   0.212s (-29.10%),   0.209s (-34.65%),   0.206s (-36.13%),   0.208s (-32.81%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    0.142s (-58.34%),   0.144s (-52.01%),   0.137s (-57.07%),   0.136s (-57.80%),   0.131s (-57.88%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    0.127s (-62.76%),   0.129s (-56.78%),   0.132s (-58.59%),   0.135s (-58.10%),   0.126s (-59.43%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    0.200s,             0.160s,             0.166s,             0.157s,             0.151s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.199s (-0.46%),    0.199s (+24.27%),   0.220s (+32.69%),   0.224s (+42.52%),   0.198s (+31.36%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.180s (-10.05%),   0.190s (+18.81%),   0.190s (+14.61%),   0.187s (+18.78%),   0.178s (+18.11%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    0.106s (-46.79%),   0.109s (-31.90%),   0.108s (-34.53%),   0.108s (-31.41%),   0.110s (-27.01%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    65.7ms (-67.11%),   66.2ms (-58.57%),   62.8ms (-62.09%),   62.6ms (-60.21%),   60.5ms (-59.91%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    44.8ms (-77.55%),   45.0ms (-71.86%),   45.0ms (-72.82%),   47.1ms (-70.09%),   44.0ms (-70.87%)    

Problem type: 33_GRID_size=200x200_fill=0.05_conn=3_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    27.0ms,             27.6ms,             28.1ms,             27.2ms,             30.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    36.9ms (+36.45%),   37.1ms (+34.42%),   40.3ms (+43.34%),   37.9ms (+39.32%),   39.7ms (+29.55%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    25.8ms (-4.61%),    28.3ms (+2.45%),    27.5ms (-1.95%),    28.0ms (+2.97%),    29.7ms (-3.23%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    20.5ms (-24.17%),   21.9ms (-20.76%),   20.7ms (-26.30%),   21.1ms (-22.48%),   20.9ms (-31.90%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    10.5ms (-61.13%),   11.1ms (-59.86%),   10.7ms (-62.07%),   10.3ms (-62.17%),   10.6ms (-65.41%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    5.7ms (-78.98%),    5.7ms (-79.29%),    5.9ms (-79.05%),    5.6ms (-79.47%),    6.0ms (-80.43%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    85.6ms,             87.7ms,             88.0ms,             92.0ms,             91.2ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    91.0ms (+6.39%),    92.5ms (+5.51%),    96.8ms (+10.00%),   92.4ms (+0.46%),    97.2ms (+6.58%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.115s (+34.38%),   0.129s (+47.22%),   0.115s (+30.59%),   0.119s (+29.14%),   0.120s (+31.92%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    53.8ms (-37.08%),   58.5ms (-33.26%),   55.7ms (-36.72%),   55.6ms (-39.57%),   56.2ms (-38.33%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    33.7ms (-60.63%),   35.5ms (-59.55%),   34.7ms (-60.56%),   33.1ms (-64.06%),   34.3ms (-62.35%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    27.5ms (-67.87%),   28.0ms (-68.04%),   29.7ms (-66.26%),   27.0ms (-70.64%),   30.3ms (-66.81%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    42.5ms,             42.8ms,             44.5ms,             42.2ms,             46.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    46.2ms (+8.65%),    46.5ms (+8.59%),    50.6ms (+13.80%),   47.3ms (+11.97%),   49.2ms (+6.96%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    46.9ms (+10.36%),   50.5ms (+17.95%),   51.4ms (+15.56%),   47.4ms (+12.26%),   50.0ms (+8.67%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    38.4ms (-9.69%),    40.0ms (-6.60%),    39.7ms (-10.81%),   38.1ms (-9.78%),    38.8ms (-15.58%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    21.4ms (-49.70%),   22.0ms (-48.66%),   22.7ms (-49.05%),   20.9ms (-50.45%),   21.8ms (-52.72%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    13.8ms (-67.48%),   13.8ms (-67.65%),   15.6ms (-64.93%),   13.5ms (-67.94%),   15.6ms (-66.04%)    

Problem type: 40_MERI_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    27.4ms,             24.6ms,             24.9ms,             25.5ms,             25.1ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    37.7ms (+37.66%),   37.1ms (+51.01%),   36.3ms (+45.93%),   37.3ms (+46.05%),   35.8ms (+42.54%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    19.3ms (-29.49%),   17.4ms (-29.19%),   17.3ms (-30.52%),   17.0ms (-33.29%),   15.4ms (-38.86%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    30.9ms (+13.06%),   30.2ms (+23.11%),   30.3ms (+21.73%),   32.0ms (+25.39%),   31.8ms (+26.44%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    16.5ms (-39.53%),   16.5ms (-33.00%),   15.6ms (-37.15%),   16.7ms (-34.40%),   15.5ms (-38.24%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    8.7ms (-68.15%),    8.3ms (-66.31%),    8.3ms (-66.76%),    8.4ms (-67.07%),    8.2ms (-67.56%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    77.1ms,             72.9ms,             72.1ms,             74.5ms,             73.9ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    91.7ms (+18.97%),   90.6ms (+24.26%),   88.6ms (+22.95%),   90.8ms (+21.97%),   87.5ms (+18.42%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    82.2ms (+6.67%),    78.1ms (+7.06%),    78.3ms (+8.70%),    82.5ms (+10.73%),   79.3ms (+7.36%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    59.0ms (-23.44%),   58.0ms (-20.47%),   58.0ms (-19.55%),   61.8ms (-17.05%),   60.1ms (-18.67%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    34.4ms (-55.39%),   34.4ms (-52.77%),   32.2ms (-55.28%),   33.5ms (-55.05%),   32.3ms (-56.32%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    21.5ms (-72.06%),   21.0ms (-71.19%),   20.9ms (-70.95%),   22.0ms (-70.48%),   20.8ms (-71.84%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    42.4ms,             39.2ms,             39.3ms,             39.8ms,             39.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    43.5ms (+2.42%),    42.7ms (+9.13%),    41.8ms (+6.23%),    43.6ms (+9.77%),    41.1ms (+3.74%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    44.2ms (+4.09%),    40.4ms (+3.06%),    39.2ms (-0.30%),    42.4ms (+6.62%),    43.2ms (+9.16%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    55.8ms (+31.49%),   61.7ms (+57.44%),   54.9ms (+39.54%),   59.1ms (+48.69%),   58.1ms (+46.74%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    32.6ms (-23.28%),   32.3ms (-17.58%),   30.5ms (-22.52%),   32.4ms (-18.55%),   30.6ms (-22.67%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    19.7ms (-53.48%),   18.6ms (-52.44%),   18.7ms (-52.56%),   19.2ms (-51.59%),   18.5ms (-53.32%)    

Problem type: 41_MERI_size=1500_n=7_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    39.4ms,             39.2ms,             39.4ms,             42.6ms,             43.4ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    55.2ms (+40.11%),   55.8ms (+42.18%),   55.7ms (+41.44%),   55.9ms (+31.14%),   56.8ms (+31.07%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    24.8ms (-37.12%),   24.5ms (-37.44%),   24.2ms (-38.47%),   25.5ms (-40.13%),   24.3ms (-44.05%)    
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    46.8ms (+18.82%),   47.1ms (+20.03%),   47.0ms (+19.46%),   46.5ms (+9.07%),    54.3ms (+25.15%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    25.1ms (-36.23%),   25.1ms (-36.05%),   24.4ms (-38.12%),   23.9ms (-43.99%),   23.8ms (-45.02%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    12.7ms (-67.75%),   13.1ms (-66.72%),   12.7ms (-67.80%),   12.9ms (-69.64%),   12.5ms (-71.27%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.117s,             0.112s,             0.116s,             0.119s,             0.123s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.135s (+15.30%),   0.136s (+21.47%),   0.135s (+16.72%),   0.133s (+11.59%),   0.138s (+12.51%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.116s (-1.24%),    0.114s (+2.40%),    0.117s (+0.98%),    0.121s (+1.35%),    0.120s (-2.14%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    89.5ms (-23.60%),   88.4ms (-20.80%),   88.8ms (-23.19%),   89.8ms (-24.74%),   89.7ms (-26.94%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    53.5ms (-54.32%),   52.7ms (-52.78%),   50.7ms (-56.11%),   53.2ms (-55.43%),   51.6ms (-57.99%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    32.9ms (-71.89%),   33.1ms (-70.32%),   32.5ms (-71.90%),   33.7ms (-71.74%),   32.9ms (-73.20%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    62.7ms,             60.9ms,             62.2ms,             66.0ms,             65.7ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    63.4ms (+1.04%),    63.9ms (+4.98%),    63.7ms (+2.31%),    66.0ms (+0.05%),    65.7ms (-0.04%)     
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    63.2ms (+0.68%),    61.8ms (+1.39%),    60.6ms (-2.59%),    62.4ms (-5.39%),    63.1ms (-3.92%)     
- 4_BaSpaCho_CUDA_batchsize=4 (vs. 1_CHOLMOD):
    86.3ms (+37.52%),   85.0ms (+39.57%),   86.5ms (+39.05%),   86.6ms (+31.33%),   85.0ms (+29.43%)    
- 5_BaSpaCho_CUDA_batchsize=8 (vs. 1_CHOLMOD):
    49.9ms (-20.53%),   49.5ms (-18.67%),   47.9ms (-22.98%),   47.6ms (-27.83%),   48.1ms (-26.81%)    
- 6_BaSpaCho_CUDA_batchsize=16 (vs. 1_CHOLMOD):
    29.0ms (-53.71%),   29.1ms (-52.24%),   28.8ms (-53.65%),   29.6ms (-55.12%),   28.7ms (-56.33%)
```

## Solve (Intel-MKL, nRHS = 1, 2, 10)
Command: `cmake --build build -v -- -j16 && build/bench -B 1_CHOLMOD -O solve-1,solve-2,solve-10 -S ^2`
```
Problem type: 10_FLAT_size=1000_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    3.4ms,              3.7ms,              4.0ms,              4.1ms,              4.1ms               
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    3.6ms (+5.08%),     3.3ms (-9.34%),     3.8ms (-5.03%),     3.5ms (-14.14%),    3.3ms (-18.98%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    8.3ms,              8.6ms,              7.4ms,              8.2ms,              7.4ms               
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    7.1ms (-14.87%),    6.2ms (-28.01%),    6.9ms (-7.34%),     6.5ms (-20.34%),    6.4ms (-13.00%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    12.8ms,             13.2ms,             11.9ms,             12.0ms,             12.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    7.9ms (-38.49%),    6.6ms (-50.11%),    6.9ms (-42.00%),    6.5ms (-45.85%),    6.4ms (-48.71%)     

Problem type: 11_FLAT_size=4000_fill=0.01_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    32.8ms,             33.0ms,             33.1ms,             31.6ms,             32.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    28.0ms (-14.60%),   30.5ms (-7.67%),    32.5ms (-1.60%),    29.3ms (-7.21%),    29.3ms (-9.93%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    51.6ms,             50.0ms,             50.9ms,             51.1ms,             49.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    48.5ms (-6.00%),    55.8ms (+11.61%),   53.4ms (+5.05%),    48.4ms (-5.30%),    51.5ms (+4.07%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    0.164s,             0.161s,             0.163s,             0.257s,             0.162s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    46.0ms (-71.91%),   50.9ms (-68.39%),   49.3ms (-69.77%),   47.0ms (-81.73%),   48.3ms (-70.19%)    

Problem type: 12_FLAT_size=2000_fill=0.03_bsize=2-5
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    16.5ms,             14.5ms,             14.9ms,             14.9ms,             15.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    13.7ms (-17.04%),   13.8ms (-5.15%),    13.3ms (-11.23%),   14.6ms (-1.60%),    14.4ms (-3.86%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    27.8ms,             22.4ms,             24.2ms,             23.9ms,             25.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    26.9ms (-3.10%),    26.2ms (+16.70%),   27.1ms (+11.91%),   29.1ms (+21.71%),   29.1ms (+13.61%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    64.7ms,             64.2ms,             62.8ms,             65.7ms,             66.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    21.0ms (-67.53%),   21.8ms (-66.09%),   22.1ms (-64.83%),   23.8ms (-63.80%),   29.9ms (-54.64%)    

Problem type: 20_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    86.3ms,             64.7ms,             64.5ms,             85.3ms,             64.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    34.8ms (-59.74%),   36.6ms (-43.50%),   36.4ms (-43.57%),   36.9ms (-56.69%),   37.6ms (-41.70%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.348s,             0.347s,             0.358s,             0.356s,             0.351s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.101s (-71.01%),   0.103s (-70.34%),   96.8ms (-73.01%),   0.102s (-71.41%),   0.101s (-71.36%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    0.175s,             0.174s,             0.177s,             0.177s,             0.175s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    43.5ms (-75.17%),   44.9ms (-74.28%),   45.4ms (-74.34%),   46.1ms (-73.95%),   45.2ms (-74.20%)    

Problem type: 21_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.2
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    6.0ms,              5.9ms,              6.4ms,              6.3ms,              5.9ms               
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    4.1ms (-31.97%),    4.1ms (-30.14%),    4.3ms (-32.43%),    4.1ms (-34.76%),    4.0ms (-31.13%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    17.9ms,             18.0ms,             18.0ms,             17.8ms,             18.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    8.8ms (-50.99%),    8.2ms (-54.53%),    8.2ms (-54.32%),    8.3ms (-53.15%),    8.2ms (-54.56%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    19.9ms,             19.7ms,             21.4ms,             20.2ms,             20.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    7.6ms (-61.78%),    7.5ms (-61.76%),    7.6ms (-64.56%),    7.5ms (-62.89%),    7.1ms (-64.47%)     

Problem type: 30_GRID_size=100x100_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    17.0ms,             18.9ms,             20.6ms,             16.8ms,             21.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    23.9ms (+40.75%),   26.8ms (+42.15%),   26.4ms (+28.41%),   25.3ms (+50.05%),   25.1ms (+17.83%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    56.8ms,             56.6ms,             70.0ms,             56.7ms,             56.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    60.8ms (+6.94%),    61.7ms (+8.89%),    64.2ms (-8.21%),    60.7ms (+6.92%),    60.9ms (+7.54%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    27.0ms,             26.8ms,             34.3ms,             26.5ms,             26.6ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    30.4ms (+12.69%),   33.7ms (+25.79%),   36.0ms (+5.03%),    33.0ms (+24.25%),   33.5ms (+26.04%)    

Problem type: 31_GRID_size=150x150_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    41.2ms,             40.7ms,             41.0ms,             40.5ms,             40.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    62.0ms (+50.23%),   61.0ms (+50.03%),   65.8ms (+60.55%),   61.0ms (+50.55%),   58.6ms (+44.47%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.136s,             0.131s,             0.131s,             0.132s,             0.130s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.132s (-2.78%),    0.131s (+0.34%),    0.138s (+5.44%),    0.130s (-1.11%),    0.131s (+0.20%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    85.6ms,             62.5ms,             64.3ms,             62.2ms,             62.2ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    71.3ms (-16.72%),   71.4ms (+14.26%),   72.0ms (+11.89%),   70.8ms (+13.90%),   69.2ms (+11.36%)    

Problem type: 32_GRID_size=200x200_fill=0.25_conn=2_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    70.5ms,             72.3ms,             70.5ms,             71.7ms,             70.5ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    99.0ms (+40.33%),   0.101s (+39.23%),   0.102s (+44.12%),   0.104s (+44.96%),   0.109s (+54.28%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.229s,             0.232s,             0.229s,             0.293s,             0.238s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.210s (-8.56%),    0.223s (-4.15%),    0.211s (-7.86%),    0.224s (-23.63%),   0.231s (-3.25%)     
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    0.126s,             0.124s,             0.127s,             0.135s,             0.132s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.123s (-2.60%),    0.115s (-6.73%),    0.122s (-4.25%),    0.125s (-7.97%),    0.127s (-4.04%)     

Problem type: 33_GRID_size=200x200_fill=0.05_conn=3_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    24.7ms,             26.6ms,             25.5ms,             24.9ms,             25.1ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    25.7ms (+3.95%),    27.8ms (+4.61%),    24.4ms (-4.25%),    23.0ms (-7.78%),    26.3ms (+4.61%)     
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    90.3ms,             0.103s,             88.8ms,             88.7ms,             94.1ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    61.2ms (-32.19%),   63.2ms (-38.50%),   60.3ms (-32.11%),   58.6ms (-33.97%),   65.7ms (-30.21%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    52.4ms,             54.9ms,             52.4ms,             51.6ms,             52.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    31.6ms (-39.70%),   34.1ms (-37.83%),   30.8ms (-41.11%),   29.9ms (-42.08%),   34.2ms (-34.25%)    

Problem type: 40_MERI_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    22.2ms,             21.7ms,             21.6ms,             21.9ms,             22.2ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    19.6ms (-11.82%),   20.1ms (-7.57%),    19.7ms (-8.92%),    19.0ms (-13.34%),   19.5ms (-12.41%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    69.6ms,             65.9ms,             63.3ms,             63.4ms,             62.2ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    51.9ms (-25.47%),   52.7ms (-20.03%),   50.8ms (-19.82%),   51.6ms (-18.70%),   49.7ms (-20.13%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    31.3ms,             30.8ms,             31.1ms,             31.2ms,             30.7ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    25.0ms (-19.94%),   25.3ms (-18.04%),   26.3ms (-15.27%),   25.1ms (-19.47%),   24.6ms (-19.64%)    

Problem type: 41_MERI_size=1500_n=7_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: solve-1
- 1_CHOLMOD (basis for comparison):
    34.4ms,             34.2ms,             34.5ms,             36.1ms,             36.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    29.9ms (-12.88%),   31.1ms (-8.81%),    30.9ms (-10.33%),   31.1ms (-13.73%),   30.1ms (-17.11%)    
Operation: solve-10
- 1_CHOLMOD (basis for comparison):
    0.114s,             98.5ms,             0.103s,             0.102s,             0.103s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    78.4ms (-31.45%),   81.5ms (-17.30%),   80.1ms (-22.49%),   80.8ms (-20.57%),   80.7ms (-21.56%)    
Operation: solve-2
- 1_CHOLMOD (basis for comparison):
    48.7ms,             48.2ms,             47.8ms,             50.0ms,             49.9ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    39.0ms (-19.81%),   38.5ms (-20.06%),   37.9ms (-20.82%),   39.1ms (-21.71%),   38.6ms (-22.65%)
```

## Analysis (OpenBLAS/Cuda 11.5)
Command: `cmake --build build -v -- -j16 && build/bench -B 1_CHOLMOD -O analysis -S ^[23]`
```
Problem type: 10_FLAT_size=1000_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    29.8ms,             25.6ms,             29.4ms,             26.7ms,             27.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    21.6ms (-27.45%),   20.0ms (-21.90%),   21.2ms (-27.63%),   20.1ms (-24.68%),   20.5ms (-24.13%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    22.8ms (-23.69%),   26.3ms (+2.66%),    23.4ms (-20.31%),   22.6ms (-15.31%),   21.2ms (-21.62%)    

Problem type: 11_FLAT_size=4000_fill=0.01_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    72.0ms,             64.4ms,             69.9ms,             70.9ms,             65.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.256s (+256.07%),  0.265s (+311.09%),  0.284s (+306.39%),  0.264s (+271.75%),  0.275s (+321.19%)   
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.278s (+286.11%),  0.267s (+314.92%),  0.291s (+316.09%),  0.281s (+295.57%),  0.300s (+360.02%)   

Problem type: 12_FLAT_size=2000_fill=0.03_bsize=2-5
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    57.0ms,             55.8ms,             51.0ms,             52.8ms,             50.2ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    62.6ms (+9.70%),    66.9ms (+20.07%),   61.8ms (+21.32%),   62.8ms (+18.94%),   62.0ms (+23.55%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    63.3ms (+10.95%),   65.5ms (+17.41%),   63.4ms (+24.47%),   64.4ms (+21.87%),   63.6ms (+26.80%)    

Problem type: 20_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    9.740s,             9.918s,             9.770s,             10.006s,            9.862s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    3.464s (-64.43%),   3.483s (-64.88%),   3.504s (-64.14%),   3.451s (-65.51%),   3.413s (-65.39%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    3.495s (-64.12%),   3.414s (-65.58%),   3.506s (-64.12%),   3.463s (-65.39%),   3.408s (-65.44%)    

Problem type: 21_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.2
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    27.6ms,             27.0ms,             27.5ms,             27.2ms,             27.2ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    22.2ms (-19.58%),   23.1ms (-14.52%),   21.8ms (-20.79%),   22.0ms (-19.02%),   21.0ms (-22.89%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    24.2ms (-12.42%),   22.9ms (-15.31%),   24.0ms (-12.82%),   23.9ms (-12.14%),   23.9ms (-12.45%)    

Problem type: 30_GRID_size=100x100_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    55.1ms,             56.0ms,             55.6ms,             55.9ms,             56.0ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    66.1ms (+19.94%),   65.8ms (+17.52%),   66.2ms (+18.99%),   67.3ms (+20.43%),   66.4ms (+18.46%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    55.9ms (+1.47%),    57.2ms (+2.18%),    57.9ms (+4.10%),    56.2ms (+0.55%),    58.8ms (+5.00%)     

Problem type: 31_GRID_size=150x150_fill=1.0_conn=2_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    0.142s,             0.140s,             0.141s,             0.143s,             0.140s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.190s (+33.63%),   0.188s (+34.14%),   0.186s (+32.43%),   0.188s (+31.41%),   0.183s (+30.50%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.183s (+28.66%),   0.183s (+30.03%),   0.185s (+31.29%),   0.184s (+28.82%),   0.183s (+30.61%)    

Problem type: 32_GRID_size=200x200_fill=0.25_conn=2_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    0.215s,             0.225s,             0.217s,             0.214s,             0.232s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.376s (+75.15%),   0.422s (+87.69%),   0.420s (+93.29%),   0.413s (+92.69%),   0.404s (+74.53%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.414s (+92.64%),   0.396s (+76.09%),   0.415s (+91.14%),   0.393s (+83.41%),   0.387s (+66.83%)    

Problem type: 33_GRID_size=200x200_fill=0.05_conn=3_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    65.6ms,             62.8ms,             63.4ms,             63.0ms,             63.3ms              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    79.1ms (+20.66%),   75.9ms (+20.81%),   80.8ms (+27.54%),   76.1ms (+20.82%),   79.4ms (+25.47%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    78.4ms (+19.64%),   77.3ms (+23.06%),   82.9ms (+30.84%),   77.7ms (+23.38%),   84.0ms (+32.77%)    

Problem type: 40_MERI_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    0.272s,             0.276s,             0.277s,             0.279s,             0.283s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    98.7ms (-63.76%),   98.7ms (-64.27%),   97.0ms (-64.91%),   0.100s (-64.02%),   97.5ms (-65.51%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.101s (-62.87%),   97.7ms (-64.64%),   97.8ms (-64.65%),   0.114s (-59.28%),   0.100s (-64.56%)    

Problem type: 41_MERI_size=1500_n=7_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3
.....(5/5, done!)                            
Operation: analysis
- 1_CHOLMOD (basis for comparison):
    0.462s,             0.460s,             0.463s,             0.475s,             0.465s              
- 2_BaSpaCho_BLAS_numthreads=16 (vs. 1_CHOLMOD):
    0.153s (-66.78%),   0.154s (-66.49%),   0.155s (-66.48%),   0.155s (-67.43%),   0.166s (-64.40%)    
- 3_BaSpaCho_CUDA (vs. 1_CHOLMOD):
    0.158s (-65.88%),   0.155s (-66.30%),   0.155s (-66.43%),   0.156s (-67.19%),   0.154s (-66.96%)
```