import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

def silu(ct):
    # SiLU(x) = x * sigmoid(x)
    return ct * ct.sigmoid()

def main():
    crypten.init()
    device = "cuda"
    runs = 10

    silu_time, silu_bytes, silu_rounds = {}, {}, {}

    for size in [(8, 11008), (8, 14336)]:
        x = crypten.cryptensor(torch.randn(size, device=device), device=device)

        crypten.reset_communication_stats()
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(runs):
            silu(x)

        torch.cuda.synchronize()
        silu_time[size[1]] = time.time() - start
        stats = crypten.get_communication_stats()
        silu_bytes[size[1]] = stats["bytes"]
        silu_rounds[size[1]] = stats["rounds"]

    if crypten.comm.get().get_rank() == 0:
        for h in [11008, 14336]:
            print(f"(8, {h}) "
                  f"time: {silu_time[h]/runs:.4f}s, "
                  f"bytes: {silu_bytes[h]/1048576/runs:.0f} MB, "
                  f"rounds: {silu_rounds[h]/runs:.0f}")

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()
