"""work/opb_venv/bin/python generators/make_opb.py [--per-suite 8] : a reduced subset of Omni Parse Bench (Datalab; dataset
datalab-to/omni_parse_bench on Hugging Face, scorer omni-parse-bench 0.2.0, Apache-2.0; tests CC BY 4.0, pages under the license of
their source, listed in the dataset's licenses/; docs/EXTERNAL.md) -> external/opb_sub/: {id}.pdf (the page as the benchmark has
it), {id}.png (rendered for the model: long side 1754 px, as every page set here), {id}.gold.json (its tests), meta.json.

Why. Real PDFs from many producers with tests written apart from our model: for the PDF text layer (glmocr/pdf.py), whose checks
had only been tried on a synthetic set and on the Gazzetta, and for v1.1 against the original model on English and European pages.
Which pages. PDF pages only (the text layer needs one), from ten suites of real documents: no arXiv math, no handwriting, no
engineering drawings. Of multiling_finepdfs, European languages only. --per-suite pages from each, drawn with a fixed seed;
pages with no text test (present, order, repeat) are left out, since only those are scored here (our reader gives no layout
blocks, and tables are not what is asked). Fixed before any model or the text layer read a page.
Scored by experiments/opb_score.py with the benchmark's own scorer, text tests only."""
import argparse, json, pathlib, random
import huggingface_hub as hub
import pandas as pd
import pypdfium2 as pdfium

ROOT = pathlib.Path(__file__).resolve().parent.parent; OUT = ROOT / "external/opb_sub"; REPO = "datalab-to/omni_parse_bench"
SUITES = ["realdoc_pdfs", "parsebench_docs", "multiling_finepdfs", "multilingual", "forms_filled", "forms_documentcloud",
          "olmo_multi_column", "olmo_old_scans", "olmo_long_tiny_text", "tables_derived"]
EUROPEAN = {"eng", "ita", "fra", "deu", "spa", "por", "nld", "pol", "ces", "slk", "slv", "hrv", "ron", "hun", "swe", "dan", "nor",
            "nob", "fin", "est", "lav", "lit", "cat", "glg", "eus", "ell", "bul", "ukr", "rus", "srp", "gle", "isl", "mlt", "ltz"}
TEXT_TESTS = {"present", "order", "repeat"}
fetch = lambda path: hub.hf_hub_download(REPO, path, repo_type="dataset")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per-suite", type=int, default=8); ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args(); rng = random.Random(a.seed); OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(fetch("manifest.parquet")); meta = []
    for suite in SUITES:
        rows = m[(m.suite == suite) & m.doc_path.str.endswith(".pdf")].sort_values("page_id").to_dict("records"); rng.shuffle(rows); taken = 0
        for r in rows:
            if taken == a.per_suite:
                break
            tests = json.load(open(fetch(r["gt_path"])))["tests"]
            text = [t for t in tests if t["test_type"] in TEXT_TESTS]
            langs = sorted({x for t in text for x in t["tags"] if len(x) == 3 and x.islower()})
            if not text or (suite == "multiling_finepdfs" and not set(langs) <= EUROPEAN):
                continue
            i = r["page_id"]; pdf_path = OUT / f"{i}.pdf"; pdf_path.write_bytes(open(fetch(r["doc_path"]), "rb").read())
            (OUT / f"{i}.gold.json").write_text(json.dumps({"tests": tests}, ensure_ascii=False))
            page = pdfium.PdfDocument(str(pdf_path))[0]
            img = page.render(scale=1754 / max(page.get_size())).to_pil().convert("RGB"); img.save(OUT / f"{i}.png")
            (OUT / f"{i}.txt").write_text("", encoding="utf-8")   # no running text as reference: the tests are the reference
            meta.append({"id": i, "pdf": f"{i}.pdf", "gt": "opb_tests", "level": 1, "tpl": suite, "lang": langs[0] if len(langs) == 1 else "mixed",
                         "langs": langs, "rot90": False, "size": list(img.size), "document": r["document"], "suite": suite,
                         "text_tests": len(text), "derivations": sorted({t["derivation"] for t in text}), "entities": []})
            taken += 1
        print(suite, taken)
    json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(meta), "pages,", sum(x["text_tests"] for x in meta), "text tests ->", OUT)
