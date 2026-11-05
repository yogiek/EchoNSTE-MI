"""Check that the container has everything the baseline needs."""
import importlib, sys

print("Python      :", sys.version.split()[0])
ok = True
for mod in ["torch", "torchvision", "cv2", "numpy", "pandas", "sklearn", "tabulate"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:12s}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        ok = False
        print(f"{mod:12s}: MISSING  ({e.__class__.__name__})")

try:
    import torch
    print("CUDA        :", torch.cuda.is_available(), "| CUDA version:", torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 1024**3:.0f} GB")
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        print("  GPU test  :", float((x @ x).mean()) is not None)
    else:
        ok = False
except Exception as e:
    ok = False
    print("torch/CUDA check failed:", e)

try:
    import shutil
    total, used, free = shutil.disk_usage("/dev/shm")
    print(f"/dev/shm    : {total / 1024**3:.1f} GB "
          f"({'OK' if total > 2 * 1024**3 else 'too small -> use --workers 0 or recreate container with --ipc=host'})")
except Exception:
    pass

print("\nRESULT:", "ready" if ok else "fix the items marked MISSING / CUDA False")
