#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py
- 重みファイルと入力画像を sys.argv で受け取り推論を実行（Top-5 表示）
- ネスト制限に配慮（ガード節・関数分割・最小 try/except・辞書ディスパッチ）
"""
import sys, os, pathlib, importlib, inspect
from typing import Tuple, List, Dict, Any
import torch
import torch.nn.functional as F
from PIL import Image

# ----------------------------- 便利関数 -----------------------------
def _e(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}"); sys.exit(code)

def _info(msg: str) -> None:
    print(f"[INFO]  {msg}")

def _is_img(p: pathlib.Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# ----------------------------- 構成品 -----------------------------
def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_config():
    try:
        return importlib.import_module("config")
    except Exception as e:
        _e(f"config.py を import できません: {e}")

def _guess_num_classes(cf) -> int:
    # 最優先: 明示的なクラス名
    for key in ("class_names", "classes"):
        v = getattr(cf, key, None)
        if isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
            return len(v)
    # 次善: よくある属性名
    for key in ("classesSize", "num_classes", "n_classes", "class_num", "num_class"):
        v = getattr(cf, key, None)
        if isinstance(v, int) and v > 0: return v
    return 1000

def _collect_build_kwargs(cf, sig: inspect.Signature) -> Dict[str, Any]:
    """build_model のシグネチャに合わせて kwargs を動的構築"""
    params = sig.parameters
    wants = set(params.keys())
    guessed_n = _guess_num_classes(cf)
    candidates = {
        "classesSize": guessed_n,
        "num_classes": guessed_n,
        "n_classes": guessed_n,
        "class_num": guessed_n,
        "num_class": guessed_n,
        # 入力解像度やチャネル数が要る場合に備えて
        "cellSize": getattr(cf, "cellSize", None),
        "image_size": getattr(cf, "cellSize", None),
        "in_ch": getattr(cf, "in_ch", None),
        "in_channels": getattr(cf, "in_ch", None),
        "device": pick_device(),
    }
    # kwargs をシグネチャでフィルタ
    kw = {k: v for k, v in candidates.items() if k in wants and v is not None}
    return kw

def build_model_from_config():
    cf = _load_config()
    if hasattr(cf, "build_model"):
        try:
            sig = inspect.signature(cf.build_model)
            kw = _collect_build_kwargs(cf, sig)
            return cf.build_model(**kw) if kw else cf.build_model()
        except TypeError as e:
            _info(f"build_model に引数を付与して再試行: {e}")
            kw = _collect_build_kwargs(cf, inspect.Signature(parameters=[]))
            return cf.build_model(**kw)
        except Exception as e:
            _e(f"build_model の呼び出しに失敗: {e}")
    # 代替: torchvision resnet18
    from torchvision.models import resnet18
    import torch.nn as nn
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, _guess_num_classes(cf))
    return m

def load_weights(model: torch.nn.Module, weight_path: str, device: torch.device) -> None:
    try:
        sd = torch.load(weight_path, map_location=device)
    except Exception as e:
        _e(f"重みファイルを読み込めません: {e}")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing: _info(f"missing keys: {len(missing)}")
    if unexpected: _info(f"unexpected keys: {len(unexpected)}")

# ----------------------------- 変換と画像 -----------------------------
def _get_transform():
    cf = _load_config()
    cand = [getattr(cf, "data_transforms", None),
            getattr(cf, "data_transform", None),
            getattr(cf, "val_transform", None)]
    for c in cand:
        if isinstance(c, dict) and "val" in c: return c["val"]
        if callable(c): return c
    from torchvision import transforms as T
    size = int(getattr(cf, "cellSize", 224))
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                      T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def load_image_tensor(img_path: str, device: torch.device) -> torch.Tensor:
    p = pathlib.Path(img_path)
    if not p.exists(): _e(f"画像が見つかりません: {img_path}")
    if p.is_dir():
        imgs = sorted([q for q in p.iterdir() if _is_img(q)])
        if not imgs: _e(f"画像が見つかりません（ディレクトリ空）: {img_path}")
        _info(f"ディレクトリ指定のため先頭ファイルを使用: {imgs[0].name}")
        p = imgs[0]
    try:
        im = Image.open(p).convert("RGB")
    except Exception as e:
        _e(f"PIL で開けません: {e}")
    return _get_transform()(im).unsqueeze(0).to(device)

# ----------------------------- 推論と表示 -----------------------------
def softmax_topk(logits: torch.Tensor, k: int = 5):
    probs = F.softmax(logits, dim=1)
    top_p, top_i = probs.topk(min(k, probs.size(1)), dim=1)
    return top_i.squeeze(0).tolist(), top_p.squeeze(0).tolist()

def maybe_load_classnames() -> List[str]:
    try:
        cf = _load_config()
        for key in ("class_names", "classes"):
            v = getattr(cf, key, None)
            if isinstance(v, (list, tuple)) and all(isinstance(s, str) for s in v):
                return list(v)
    except Exception:
        pass
    return []

def pretty_show(weight_path: str, model: torch.nn.Module, logits: torch.Tensor) -> None:
    cls = maybe_load_classnames()
    idx, prob = softmax_topk(logits, 5)
    _info(f"weights: {weight_path}")
    _info(f"model : {model.__class__.__name__}")
    _info(f"output: {tuple(logits.shape)}  (N,C)")
    for r, (i, p) in enumerate(zip(idx, prob), 1):
        name = cls[i] if i < len(cls) else f"class_{i}"
        print(f"TOP{r}: {i:>4}  {name:<20}  prob={p:.4f}")

# ----------------------------- main -----------------------------
def main():
    if len(sys.argv) < 3: _e("使い方: python predict.py <weight.pth> <image_or_dir>")
    w, img = sys.argv[1], sys.argv[2]
    if not os.path.exists(w): _e(f"重みファイルが存在しません: {w}")
    device = pick_device()

    model = build_model_from_config().to(device)
    model.eval()
    load_weights(model, w, device)

    x = load_image_tensor(img, device)
    with torch.no_grad():
        y = model(x)
        if y.ndim != 2: _e(f"想定外の出力形状: {tuple(y.shape)} (N,C が想定)")
    pretty_show(w, model, y)

if __name__ == "__main__":
    main()
