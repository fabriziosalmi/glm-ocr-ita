"""Does a page set keep the contract the scorer relies on? Run before a set enters any split (docs/DATASET_PROMPT.md)."""
import collections
import struct
import unicodedata

from . import corpus, metrics

SIZE = (1240, 1754)   # A4 at 150 dpi


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24]) if head[:8] == b"\x89PNG\r\n\x1a\n" else None


def check(name, size=SIZE):
    """Returns (counts, problems). problems: kind -> list of page ids, minus the kinds datasets.toml accepts for this set.
    An empty dict means the set is usable as it is."""
    d = corpus.corpus_dir(name)
    meta = corpus.load_meta(name)
    bad = collections.defaultdict(list)
    ids = [m["id"] for m in meta]
    bad.update({"duplicate id in meta.json": dup} if (dup := [i for i, n in collections.Counter(ids).items() if n > 1]) else {})
    known = set(ids)
    for p in sorted(d.glob("*.png")) + sorted(d.glob("*.txt")):
        if p.stem not in known:
            bad["file not listed in meta.json"].append(p.name)
    missing = short = entities = 0
    for m in meta:
        i = m["id"]
        png, txt = d / f"{i}.png", d / f"{i}.txt"
        if not png.exists() or not txt.exists():
            bad["page or ground truth missing"].append(i)
            continue
        got = png_size(png)
        if got != size and got != size[::-1]:
            bad[f"image is not {size[0]}x{size[1]}"].append(i)
        raw = txt.read_text(encoding="utf-8")
        text = corpus.read_reference(txt)
        if text != unicodedata.normalize("NFC", text):
            bad["ground truth is not NFC"].append(i)
        if corpus.escaped_breaks(raw):
            bad["ground truth holds a literal backslash-n instead of line breaks"].append(i)
        elif "\\n" in raw:
            bad["ground truth mixes real line breaks and a literal backslash-n"].append(i)
        if not m["entities"]:
            bad["no entities"].append(i)
        ref = metrics.squash(text)
        gone = [e for e in m["entities"] if metrics.squash(e["text"]) not in ref]
        if any(e.get("type") == "hand" for e in gone):
            bad["handwritten entity not in the ground truth"].append(i)
        if any(e.get("type") != "hand" for e in gone):
            bad["entity not in the ground truth"].append(i)
        missing += len(gone)
        entities += len(m["entities"])
        short += sum(len(metrics.squash(e["text"])) < 5 for e in m["entities"])
    langs = {m.get("lang") for m in meta}
    for level in sorted({m.get("level") for m in meta}):
        if len(langs) > 1 and len({m.get("lang") for m in meta if m.get("level") == level}) == 1:
            bad["a level holds one language only (language and level cannot be told apart)"].append(f"L{level}")
    count = lambda k: dict(sorted(collections.Counter(str(m.get(k)) for m in meta).items()))
    counts = {"dir": str(d), "pages": len(meta), "entities": entities, "entities_missing_from_gt": missing,
              "entities_under_5_chars": short, "rotated": sum(bool(m.get("rot90")) for m in meta),
              "lang": count("lang"), "level": count("level"), "templates": len({m.get("tpl") for m in meta})}
    accepted = next((s.get("accept", []) for k, s in corpus.registry().items() if k == name), [])
    return counts, {k: v for k, v in bad.items() if k not in accepted}
