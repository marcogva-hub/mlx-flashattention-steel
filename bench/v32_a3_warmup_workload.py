import mlx.core as mx
import time
start = time.time()
n = 0
while time.time() - start < 30:
    a = mx.random.normal((4096, 4096), dtype=mx.float16)
    b = mx.random.normal((4096, 4096), dtype=mx.float16)
    c = a @ b
    mx.synchronize()
    n += 1
print(f"warmup done in {time.time()-start:.1f}s, {n} matmul iters")
