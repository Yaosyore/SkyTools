# -*- coding: utf-8 -*-
"""
光遇资源识别工具
用法:
    python detect.py <图片路径> [图片路径2 ...]
    python detect.py "C:/path/to/image.png"

输出六类资源数量（白蜡/爱心/季蜡/季心/红蜡/代币），数值为0的资源不显示。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recognizer import ResourceRecognizer, SLOT_NAMES

def model_dir():
    if getattr(sys, 'frozen', False):        # exe运行时：onnx与exe同目录
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rr = ResourceRecognizer(model_dir())
    ok = True
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(f"[不存在] {p}")
            ok = False
            continue
        try:
            counts, w, h = rr.predict(p)
            label = os.path.basename(p)
            parts = [f"{SLOT_NAMES[i]} {counts[i]}" for i in range(6) if counts[i] > 0]
            print(f"{label} ({w}x{h}): " + (", ".join(parts) if parts else "全部为 0"))
        except Exception as e:
            print(f"[失败] {p}: {e}")
            ok = False
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
