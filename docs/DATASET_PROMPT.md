# Prompt for building one synthetic document dataset

Give the whole block below to a model that knows nothing about this project. Change only the five values under "Your assignment". Each model gets a different name, seed and focus, so the datasets are independent of each other and of `make_hard.py`.

```text
You are going to build a synthetic dataset of document page images with exact ground truth, to train and test an OCR model on Italian documents. You know nothing about the project, and you do not need to: everything required is in this message. Do not ask questions. Where something is not specified, choose, and write the choice down in the README you deliver.

## Your assignment

NAME       = itdoc_a            # dataset name, used in file names
SEED       = 4101               # master random seed
N_PAGES    = 1500               # pages to generate
LANG_MIX   = 80% Italian, 20% English
FOCUS      = invoices, delivery notes and purchase orders

## What the dataset is for

An OCR model reads a page image and returns its text. We measure whether specific strings that matter (amounts, dates, bank and tax identifiers) come back exactly. Two failure modes are known and your pages must be able to expose them:

1. The model silently skips whole blocks of small print: invoice detail lines, footnotes, footers, conditions in fine print. It reads the rest of the page perfectly, so nothing looks wrong.
2. The model rewrites Italian number notation as English: 2.100.000,00 becomes 2,100,000.00.

So a page is useful when it holds a lot of text, some of it small, with many exact strings to recover. A page that is half empty is a wasted page.

## Deliverables

1. make_NAME.py: one Python 3.12 script that generates the whole dataset. Allowed libraries: pillow, numpy, and the standard library. Nothing else.
2. The dataset it produces, in a folder called NAME/ (layout below).
3. fonts/: the font files the script uses, and FONTS.md listing each font with its license and source URL.
4. README_NAME.md: what you built, every choice you made, the table of counts printed by the self-check, and anything that does not meet this specification.

If you can run code, run the script, run the self-check, and deliver all four. If you cannot run code, deliver the script, the font list and the README, and say clearly that nothing was executed.

## Folder layout

NAME/
  meta.json
  NAME_0000_<template>_L<level>_<lang>.png
  NAME_0000_<template>_L<level>_<lang>.txt
  NAME_0001_...

- Page id = file name without extension. Index is zero-padded to 4 digits; <lang> is it or en; <level> is 1 to 5.
- PNG: exactly 1240 x 1754 pixels, RGB, 8 bits per channel. That is A4 at 150 dpi. Portrait only. Never rotate a page by 90, 180 or 270 degrees.
- TXT: the ground truth. UTF-8, Unicode NFC, Unix line endings, no markup of any kind.
- meta.json: a JSON list with one object per page, in index order:

  {
    "id": "NAME_0000_invoice_L3_it",
    "gt": "exact",
    "level": 3,
    "tpl": "invoice",
    "lang": "it",
    "font": "name of the main font",
    "rot90": false,
    "seed": 4101000,
    "small_print_px": 15,
    "degradation": {"rotation_deg": 1.2, "blur": 0.8, "noise_sigma": 10, "jpeg_quality": 50},
    "overlays": ["PROTOCOLLO N. 4471"],
    "entities": [
      {"type": "amount", "text": "1.234,56 €"},
      {"type": "iban", "text": "IT60 X054 2811 1010 0000 0123 456"}
    ]
  }

## Ground truth rules

These are the rules that make the dataset usable. A dataset that breaks one is rejected by an automatic checker, whatever its other qualities.

1. The .txt contains exactly the strings your script drew as document content, and nothing else. The ground truth comes from the strings in your code. Never produce it by running OCR or a language model on the image, and never edit a file by hand.
2. One text block per line, in natural reading order: top to bottom; in a multi-column layout the whole left column, then the right; in a table row by row, one line per cell, left to right; footnotes and footers last. A paragraph is one line, however many lines it wraps to on the page.
3. Everything a reader can read as content is in the .txt: headers, footers, page numbers, footnotes, fine print, handwritten notes.
4. Decorative overlays (rubber stamps, watermarks) are NOT in the .txt. Their text goes in "overlays" in meta.json.
5. Every entity in meta.json appears character for character in the .txt of its page, and is fully visible on the page: not clipped by the page edge, not covered by a stamp, not broken across two lines.
6. An entity string appears once per page. Two different entities on a page never have the same text.
7. Text never overflows the page or overlaps other text. Check it with measured text widths, not by eye.

## Entities

Mark these, and only these, as entities. Use exactly these type names.

- amount: money. Italian pages use Italian notation only: "1.234,56 €", "€ 1.234,56", "EUR 1.234,56", with the dot for thousands and the comma for decimals. English pages use "$ 1,234.56", "GBP 1,234.56", "EUR 1,234.56". Include amounts from a few euros to tens of millions, so that many have two or three thousands groups.
- date: Italian "03/07/2025", "03.07.2025", "3 luglio 2025"; English "2025-07-03", "July 3, 2025", "3 July 2025".
- iban: Italian IBAN, 27 characters, with a valid mod-97 check, written in groups of four.
- vat: Italian partita IVA, 11 digits with a valid check digit.
- cf: Italian codice fiscale, 16 characters, valid structure and valid check character.
- cig: 10 alphanumeric characters. cup: 15 characters, one letter, two digits, one letter, eleven digits.
- contract: contract, order, invoice or protocol numbers, such as "CTR/2025/00417", "ORD-2024-118", "Prot. n. 2291/2025".
- pec: certified e-mail addresses, always on invented domains ending in .example.
- percent: "22%", "4,5%" on Italian pages, "4.5%" on English ones.
- hand: a handwritten amount or date added to the page with a handwriting font, in dark blue or black ink.

Generate identifiers algorithmically with valid checksums. Never use an identifier, a company, a person or an address that exists. Invent plausible company and personal names; real Italian city names are fine.

Every page has at least 6 entities, and the dataset averages at least 12 per page. At least a third of all entities must sit inside small print (see below).

## Templates

At least 12 visibly different page layouts, each one a function in the script, each with internal variation (number of rows, of columns, of paragraphs, block positions, alignment, presence of a letterhead or a footer). Spend most of them on your FOCUS, but these six are mandatory whatever the focus, because they are where OCR models fail:

1. Invoice or statement with 12 to 30 detail lines in small print, each line with a description, a quantity and two amounts, followed by totals.
2. Body text with 3 to 8 footnotes in small print at the bottom, each holding an entity.
3. Dense table of 8 to 20 rows with a header that spans columns, right-aligned numbers and a total row.
4. Two or three columns of continuous text.
5. Form with 10 to 20 labelled fields, label in small grey capitals above the value.
6. Letter with a letterhead, body, and a footer in small print carrying company data (vat, cf, iban, pec).

"Small print" means text whose capital letters are 11 to 17 pixels tall on the 1240 x 1754 page (roughly 6 to 8 point). Body text is 21 to 30 pixels. Record the smallest text size of each page in "small_print_px". At least half of the pages must contain a small-print block of 300 characters or more.

Fill the page: text should cover at least half of the printable area on at least 80% of pages.

## Text

- Write real Italian and real English, in the registers of the documents you are imitating: commercial, legal, administrative, banking. No lorem ipsum, no filler, no translationese.
- Italian must be correct: accents (perché, poiché, cioè, è, più, né, già), apostrophes (l'IVA, dell'art., un'offerta), abbreviations (S.r.l., S.p.A., art., n., c.a., D.Lgs. 36/2023, p.es.), capital letters after a full stop only.
- Write the sentences yourself. Do not copy from existing documents, datasets or templates.
- Build text from sentence pools with slots (names, amounts, dates, references), with at least 300 distinct sentence patterns per language. After filling the slots, no identical sentence may appear on more than 2% of the pages.

## Fonts

- Use only fonts with an open license that allows redistribution (for example Liberation, DejaVu, Noto, Source Serif, Source Sans, IBM Plex, Courier Prime, Libertinus, and an open handwriting font). Never use fonts installed with the operating system.
- At least 8 text font families, mixing serif, sans-serif and monospaced, plus at least 2 handwriting fonts.
- The script loads fonts only from fonts/. Downloading them is a separate step (a function run with --fetch-fonts), never part of generation.
- Every font must render every character you use, including à è é ì ò ù À È É and the euro sign. Test it; a missing glyph drawn as an empty box makes the ground truth false.

## Degradation levels

Draw the clean page first, then degrade it according to its level. Pages are split evenly across the five levels.

- L1: clean, as drawn.
- L2: light scan. Rotation up to 0.7 degrees, slight blur, faint noise, JPEG quality around 75.
- L3: bad scan. Rotation up to 1.8 degrees, blur, visible noise, lower contrast, a lighting gradient, downsampled to 75% and back, JPEG quality around 50. One handwritten note.
- L4: L3 plus a rubber stamp partly over the text, perspective warp, stronger noise, downsampled to 60% and back, JPEG quality around 35.
- L5: the edge of legibility. Downsampled to 50% and back, heavy noise, low contrast, a fold shadow, show-through from the back of the sheet, JPEG quality around 25.

The limit for L5: a careful person looking at the page at 100% zoom must still be able to read every entity. If you could not read it yourself, it is too much. Vary the parameters at random inside each level and record the values actually used in "degradation".

## Reproducibility

- Everything random derives from SEED. Page i uses seed SEED * 1000 + i, recorded in meta.json, so any single page can be regenerated alone.
- Same script, same fonts, same seed must give identical .txt and meta.json byte for byte, and identical pixels.
- No network access, no clock, no environment-dependent value during generation.
- python make_NAME.py --out NAME --seed SEED --pages N_PAGES generates the dataset. --pages 24 must work, for a quick look.

## Self-check

The script has a --check mode that reads the generated folder and fails loudly on any violation. It verifies at least:

- every page has its .png, its .txt and its meta entry; image size and mode are right;
- every entity text is in its page's .txt; no duplicate entity text within a page; entity count per page and on average;
- IBAN, partita IVA and codice fiscale checksums are valid;
- on Italian pages no amount is in English notation, and on English pages none is in Italian notation;
- no character in any .txt is missing from the font it was drawn with;
- shares of pages per level, per template and per language; share of pages with a small-print block; share of entities inside small print;
- no sentence on more than 2% of the pages;
- generating the first 24 pages twice gives identical files.

It prints a table of counts. Paste that table into the README.

## Order of work

1. Write the script and generate 24 pages.
2. Look at them. Open at least one page per template and per level and compare the image with its .txt line by line. Fix what is wrong: text off the page, overlapping blocks, empty pages, entities under a stamp, missing glyphs.
3. Generate the full dataset and run --check until it passes.
4. Write the README, including what you could not do.

Report honestly. A dataset with a known, stated limitation is useful. A dataset that claims to meet this specification and does not is worse than none.
```

## Suggested assignments

| NAME | SEED | FOCUS |
|---|---|---|
| `itdoc_a` | 4101 | invoices, delivery notes and purchase orders |
| `itdoc_b` | 4202 | contracts, private agreements and their annexes |
| `itdoc_c` | 4303 | public administration: resolutions, tender notices, minutes, protocol letters |
| `itdoc_d` | 4404 | banks and insurance: statements, policy schedules, payment notices |
| `itdoc_e` | 4505 | quotes, technical reports, forms and receipts |

Decided on 2026-10-12, before any dataset existed: **`itdoc_c` is the frozen synthetic test set**; the other four are for training and tuning (rule 2 of `ROADMAP.md`). A dataset is accepted or rejected by its own `--check` and by ours, never by how a model scores on it. If `itdoc_c` fails the checks, the test set becomes the next one in the table that passes, and the change is written in `LEDGER.md`.

What was delivered, and the role of each set: `DATASETS.md`. Our check is `glmocr check`.
