"""The scorer, fed deliberately broken inputs."""
import pytest

from glmocr import metrics
from glmocr.models import bits_for, parse_variant
from glmocr.score import verdict

ENT = [{"type": "amount", "text": "2.100.000,00 €"}, {"type": "iban", "text": "IT60 X054 2811 1010 0000 0123 456"}]


def test_entity_survives_spacing_and_case():
    assert metrics.found(ENT, "totale 2.100.000,00€\niban it60x0542811 1010 0000 0123 456") == 2


def test_english_notation_is_a_miss():
    assert metrics.found(ENT[:1], "Totale 2,100,000.00 €") == 0


def test_one_wrong_digit_is_a_miss():
    assert metrics.found(ENT[1:], "IT60 X054 2811 1010 0000 0123 457") == 0


def test_fullwidth_digits_are_normalised():
    assert metrics.found([{"text": "123,45"}], "１２３,４５") == 1


def test_loop_by_repetition_and_by_budget():
    page = "Il corrispettivo è dovuto a mezzo bonifico bancario entro sessanta giorni dalla data di ricevimento della fattura. " * 3
    assert not metrics.is_loop(page + "La penale giornaliera è fissata in 1.250,00 € per ogni giorno di ritardo, oltre IVA.", 120)
    assert metrics.is_loop("Art. 7 - Recesso\n" * 80, 900)
    assert metrics.is_loop(page, metrics.MAX_TOKENS)          # budget exhausted: not read to the end
    assert not metrics.is_loop("ab" * 50, 40)                 # too short to judge by compression


def test_skipped_block():
    ref = "riga di dettaglio 12,50 € " * 40
    assert metrics.is_skipped(ref[: len(ref) // 2], ref)
    assert not metrics.is_skipped(ref.replace(" ", "\n"), ref)   # layout is not content
    assert not metrics.is_skipped(ref[: int(len(ref) * 0.95)], ref)


def test_cer_ignores_markup_and_is_capped():
    assert metrics.cer("Totale 10,00 €", "| **Totale** | 10,00 € |") == 0
    assert metrics.cer("abc", "x" * 500) == 1.0
    assert metrics.cer("", "anything") == 0


def test_recall_pools_entities_not_pages():
    counts = {"a": (1, 1), "b": (0, 9)}
    assert metrics.recall(counts, ["a", "b"]) == 10
    assert metrics.recall({}, []) == 0


def test_bootstrap_is_deterministic_and_brackets_the_value():
    counts = {str(i): (i % 4 != 0, 1) for i in range(60)}
    stat = lambda ids: metrics.recall(counts, ids)
    lo, hi = metrics.bootstrap(stat, list(counts))
    assert (lo, hi) == metrics.bootstrap(stat, list(counts))
    assert lo < stat(list(counts)) < hi


@pytest.mark.parametrize("ci, expected", [((0.1, 2.0), "better"), ((-0.3, 0.1), "not worse"), ((-2.5, 1.2), "not shown"),
                                         ((-5.5, -1.1), "worse"), ((-1.5, 0.5), "not shown")])
def test_verdict_follows_rule_4(ci, expected):
    assert verdict(ci) == expected


def test_variant_names():
    assert parse_variant("v8d4e4") == (8, 4, 4, 64)
    assert parse_variant("v6d4e4g128") == (6, 4, 4, 128)
    assert parse_variant("v16d4e4")[0] == 16
    for bad in ("v8d4", "8d4e4", "v8d4e4g", "v12d4e4", "v8d16e8", "v8d4e4 "):
        with pytest.raises(SystemExit):
            parse_variant(bad)


def test_bits_go_to_the_right_part():
    assert bits_for("vision_tower.blocks.3.attn.qkv", 8, 4, 5) == 8
    assert bits_for("language_model.model.layers.0.mlp", 8, 4, 5) == 4
    assert bits_for("language_model.model.embed_tokens", 8, 4, 5) == 5
    assert bits_for("language_model.lm_head", 8, 4, 5) == 5


def test_italian_notation_returned_as_english_is_counted():
    ents = [{"text": "2.100.000,00 €"}, {"text": "1.234,56 €"}, {"text": "850,00 €"}, {"text": "14/03/2025"}]
    assert metrics.notation_flips(ents, "Totale 2,100,000.00 € e 1.234,56 €") == (1, 2)   # one flipped of two that can flip
    assert metrics.notation_flips(ents, "nothing here") == (0, 2)                            # missing is not flipped


def test_accented_words_must_come_back_as_written():
    ref = "La società è tenuta, perché così si evitano più rischi."
    assert metrics.accent_recall(ref, "La società è tenuta, perché così si evitano più rischi.") == (5, 5)
    assert metrics.accent_recall(ref, "La societa e tenuta, perche cosi si evitano piu rischi.") == (0, 5)
    assert metrics.accent_recall("no accents here", "x") == (0, 0)


def test_a_repetition_at_the_end_is_found_where_it_starts():
    page = "Art. 3 - Durata\nIl contratto ha durata di dodici mesi dalla data di sottoscrizione.\n"
    unit = "Il Fornitore garantisce la conformità dei beni.\n"
    start, period = metrics.loop_start(page + unit * 6)
    assert (start, period) == (len(page), len(unit))
    assert metrics.loop_start(page + "0" * 200) == (len(page), 1)
    assert metrics.loop_start(page + unit * 6 + "Fine.") is None      # the repetition must reach the end
    assert metrics.loop_start(page + unit * 2) is None                 # twice is not a loop


def test_a_page_with_similar_rows_is_not_a_loop():
    rows = "".join(f"Servizio B{10 + i}/2025  {i + 1}  {125.5 * (i + 1):.2f} €  {251.0 * (i + 1):.2f} €\n" for i in range(30))
    assert metrics.loop_start(rows) is None
    assert metrics.loop_start("Totale 0,00 €\n" * 3) is None           # three equal lines, under the span
    assert metrics.loop_start("") is None

