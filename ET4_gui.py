# -*- coding: utf-8 -*-
"""光遇 · 资源识别工具（GUI，参考 SkyDataTool 风格）
拖入一张或多张图片 → 自动加入列表并逐张识别六类资源数量。
识别依赖同目录 onnx 模型（icon_cnn5.onnx / digit_cnn.onnx / icon_cnn5_meta.json）。
"""
import os, sys, queue, threading, csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from recognizer import ResourceRecognizer, SLOT_NAMES

FONT = ("Microsoft YaHei UI", 10)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}

def model_dir():
    if getattr(sys, "frozen", False):
        # 自包含版：模型内嵌于 _MEIPASS/models；否则回退 exe 同目录
        if hasattr(sys, "_MEIPASS"):
            p = os.path.join(sys._MEIPASS, "models")
            if os.path.isdir(p):
                return p
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

class App:
    def __init__(self, root):
        self.root = root
        self.rr = ResourceRecognizer(model_dir())
        self.items = []            # 每项: path/basename/counts/res/err/idx
        self.queue = queue.Queue() # 后台线程 → 主线程
        self.sem = threading.Semaphore(3)  # 最多同时识别3张
        self.total = 0
        self.done = 0

        root.title("光遇 · 资源识别工具")
        root.configure(bg="#f0f0f0")
        root.geometry("880x540")
        root.minsize(640, 400)
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", self.on_drop)
        self._build_ui()
        root.after(80, self._drain_queue)

    def _build_ui(self):
        # 顶部提示
        top = tk.Frame(self.root, bg="#f0f0f0")
        top.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(top, text="将图片拖入窗口即可自动识别", font=FONT, bg="#f0f0f0",
                 anchor="w").pack(side="left")
        self._lblStats = tk.Label(top, text="已识别 0 / 0", font=FONT, bg="#f0f0f0")
        self._lblStats.pack(side="right")

        # 中间列表
        mid = tk.Frame(self.root, bg="#f0f0f0")
        mid.pack(fill="both", expand=True, padx=12, pady=6)
        self._list = tk.Listbox(mid, font=FONT, activestyle="none",
                                selectbackground="#cde3ff", relief="solid", bd=1)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self._list.yview)
        self._list.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._list.pack(side="left", fill="both", expand=True)

        # 底部按钮 + 状态栏
        bot = tk.Frame(self.root, bg="#f0f0f0")
        bot.pack(fill="x", padx=12, pady=(4, 12))
        tk.Button(bot, text="清空列表", font=FONT, width=12, command=self._clear).pack(side="left")
        tk.Button(bot, text="导出 CSV", font=FONT, width=12, command=self._export).pack(side="left", padx=(8, 0))
        tk.Button(bot, text="重新识别全部", font=FONT, width=14, command=self._redo_all).pack(side="right")
        self._status = tk.Label(self.root, text="就绪，拖入图片开始", font=FONT,
                                bg="#dcdcdc", anchor="w", padx=12)
        self._status.pack(fill="x", side="bottom")

    # ---- 拖放 ----
    def on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        for f in files:
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in IMG_EXT:
                p = os.path.abspath(f)
                if not any(it["path"] == p for it in self.items):
                    self._add(p)

    def _add(self, path):
        item = {"path": path, "basename": os.path.basename(path),
                "size": "", "counts": None, "res": None, "err": None, "idx": len(self.items)}
        self.items.append(item)
        self.total += 1
        self._list.insert("end", f"{item['basename']}  …识别中")
        threading.Thread(target=self._recognize, args=(item,), daemon=True).start()
        self._update_stats()

    def _recognize(self, item):
        with self.sem:
            try:
                counts, w, h = self.rr.predict(item["path"])
                res = "  ".join(f"{SLOT_NAMES[i]} {counts[i]}" for i in range(6) if counts[i] > 0)
                res = res if res else "全部为 0"
                self.queue.put((item, counts, f"{w}x{h}", res, None))
            except Exception as e:
                self.queue.put((item, None, "", "", str(e)))

    def _drain_queue(self):
        try:
            while True:
                item, counts, size, res, err = self.queue.get_nowait()
                self.done += 1
                item["counts"] = counts; item["size"] = size
                item["res"] = res; item["err"] = err
                idx = self.items.index(item)
                text = self._fmt(item)
                self._list.delete(idx); self._list.insert(idx, text)
                self._update_stats()
                if err:
                    self._status.config(text=f"[失败] {item['basename']}: {err}")
                else:
                    self._status.config(text=f"已识别 {item['basename']}: {res}")
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    @staticmethod
    def _fmt(it):
        line = it["basename"]
        if it["size"]:
            line += f"  ({it['size']})"
        if it["err"]:
            line += "  [识别失败]"
        elif it["res"]:
            line += f"  →  {it['res']}"
        else:
            line += "  …识别中"
        return line

    def _update_stats(self):
        self._lblStats.config(text=f"已识别 {self.done} / {self.total}")

    def _clear(self):
        self.items.clear(); self._list.delete(0, "end")
        self.total = 0; self.done = 0
        self._update_stats(); self._status.config(text="已清空，拖入图片开始")

    def _redo_all(self):
        if not self.items:
            return
        # 重新识别：保留顺序，重置状态
        self.total = 0; self.done = 0
        for i, it in enumerate(self.items):
            it["counts"] = None; it["res"] = None; it["err"] = None
            self._list.delete(i); self._list.insert(i, self._fmt(it))
            threading.Thread(target=self._recognize, args=(it,), daemon=True).start()
        self._update_stats(); self._status.config(text="重新识别中…")

    def _export(self):
        done = [it for it in self.items if it["counts"] is not None]
        if not done:
            messagebox.showinfo("导出 CSV", "暂无识别结果可导出。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV", "*.csv")], title="导出识别结果")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["文件", "宽度", "高度"] + SLOT_NAMES)
                for it in done:
                    row = [it["basename"], *it["size"].split("x")] + \
                          ([str(c) if c else "" for c in it["counts"]] if it["counts"] else [""]*6)
                    w.writerow(row)
            self._status.config(text=f"已导出 {len(done)} 条到 {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

def main():
    root = TkinterDnD.Tk()
    root.tk.call("tk", "scaling", 1.2)
    try:
        root.iconbitmap(default=os.path.join(model_dir(), "ET4.ico"))
    except Exception:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
