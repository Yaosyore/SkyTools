# -*- mode: python ; coding: utf-8 -*-
import os
PKG = r"C:\Users\24223\AppData\Local\Doubao\User Data\sandbox_runtime\bases\772984f6fd3a8ae0ae7d2422cd7c2bfa\python\Lib\site-packages\rapidocr_onnxruntime"
TK = r"C:\Users\24223\AppData\Local\Doubao\User Data\sandbox_runtime\bases\772984f6fd3a8ae0ae7d2422cd7c2bfa\python\Lib\site-packages\tkinterdnd2"
SRC = r"C:\Users\24223\Desktop\SkyTools\_Source\ExperimentalTools\ET1\ET1-4\ET1-4.2.1"

a = Analysis(
    [os.path.join(SRC, "ET4_gui.py")],
    pathex=[SRC],
    binaries=[],
    datas=[
        (os.path.join(PKG, "config.yaml"), "rapidocr_onnxruntime"),
        (os.path.join(PKG, "ch_ppocr_v2_cls", "config.yaml"), "rapidocr_onnxruntime/ch_ppocr_v2_cls"),
        (os.path.join(PKG, "ch_ppocr_v3_det", "config.yaml"), "rapidocr_onnxruntime/ch_ppocr_v3_det"),
        (os.path.join(PKG, "ch_ppocr_v3_rec", "config.yaml"), "rapidocr_onnxruntime/ch_ppocr_v3_rec"),
        (os.path.join(PKG, "models"), "rapidocr_onnxruntime/models"),
        (os.path.join(TK, "tkdnd"), "tkinterdnd2/tkdnd"),
        (os.path.join(SRC, "icon_cnn5.onnx"), "models"),
        (os.path.join(SRC, "digit_cnn.onnx"), "models"),
        (os.path.join(SRC, "icon_cnn5_meta.json"), "models"),
    ],
    hiddenimports=[
        "rapidocr_onnxruntime.ch_ppocr_v2_cls",
        "rapidocr_onnxruntime.ch_ppocr_v3_det",
        "rapidocr_onnxruntime.ch_ppocr_v3_rec",
    ],
    excludes=["torch", "torchvision", "scipy", "matplotlib", "pandas", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="ET1-4.2.1",
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
