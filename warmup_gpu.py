import torch
import torch.nn as nn
import time

def run(iters=3000,  device = "cuda:0"):
    
    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)
    x = torch.rand(size=(1024*4, 1024*4)).to(device)
    w1 = torch.rand(size=(1024*4, 1024*4)).to(device)
    w2 = torch.rand(size=(1024*4, 1024*4)).to(device)
    
    for i in range(iters):
        torch.cuda.nvtx.range_push('iter{}'.format(i))

        with torch.cuda.stream(s1):
            out1 = x.matmul(w1)
    
        with torch.cuda.stream(s2):
            out2 = x.matmul(w2)
            
        torch.cuda.nvtx.range_pop()
        

if __name__=='__main__':
    print("[INFO] starting .........")
    since = time.time()
    # warmup
    run( device = "cuda:0")
    torch.cuda.cudart().cudaProfilerStart()
    run( device = "cuda:1")
    torch.cuda.cudart().cudaProfilerStop()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    )) 