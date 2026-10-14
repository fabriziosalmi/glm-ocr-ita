"""Corpus = my synthetic pages + pages rendered (200 dpi, as pdf-ocr-ng does) from pdf-ocr-ng/data/*.pdf.
GT: synthetic -> exact txt; digital PDF pages -> text layer; scans -> none (drift vs bf16 only)."""
import fitz, pathlib, shutil, json
SRC = pathlib.Path.home()/"Documents/git/pdf-ocr-ng/data"
OUT = pathlib.Path("corpus"); OUT.mkdir(exist_ok=True)
meta = []
for p in sorted(pathlib.Path("eval").glob("*.png")):
    shutil.copy(p, OUT/f"syn_{p.name}"); gt = p.with_suffix(".txt")
    shutil.copy(gt, OUT/f"syn_{p.stem}.txt"); meta.append({"id": f"syn_{p.stem}", "gt": "exact"})
for pdf in sorted(SRC.glob("[0-9][0-9]_*.pdf")):
    doc = fitz.open(pdf); n = len(doc); pages = sorted({0, n-1} if pdf.name[:2] != "09" else {0, 4})
    for i in pages:
        pg = doc[i]; txt = pg.get_text().strip(); id_ = f"{pdf.stem[:2]}_p{i+1}"
        pg.get_pixmap(dpi=200).save(OUT/f"{id_}.png")
        if len(txt) > 200: (OUT/f"{id_}.txt").write_text(txt, encoding="utf-8"); g = "textlayer"
        else: g = "none"
        meta.append({"id": id_, "gt": g})
json.dump(meta, open(OUT/"meta.json", "w"), indent=1)
print(len(meta), "pages;", sum(m["gt"] != "none" for m in meta), "with GT")
