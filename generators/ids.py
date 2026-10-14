"""Made-up Italian identifiers with valid check characters, and the lists they are drawn from. Pure Python: imported by the page
generators in either environment. Values are fictitious; a tax code built here can match a real person only by chance."""
import string

ODD = dict(zip(string.digits + string.ascii_uppercase, [1, 0, 5, 7, 9, 13, 15, 17, 19, 21] + [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]))
EVEN = {c: (int(c) if c.isdigit() else ord(c) - 65) for c in string.digits + string.ascii_uppercase}


def codice_fiscale(rng, surname, name, year, month, day, female, place):
    """16 characters, check letter computed with the official odd/even tables"""
    cons = lambda s: [c for c in s.upper() if c.isalpha() and c not in "AEIOU"]
    vow = lambda s: [c for c in s.upper() if c in "AEIOU"]
    sur = (cons(surname) + vow(surname) + ["X"] * 3)[:3]
    c = cons(name)
    nam = [c[0], c[2], c[3]] if len(c) >= 4 else (c + vow(name) + ["X"] * 3)[:3]
    body = "".join(sur + nam) + f"{year % 100:02d}" + "ABCDEHLMPRST"[month - 1] + f"{day + 40 * female:02d}" + place
    s = sum(ODD[ch] if k % 2 == 0 else EVEN[ch] for k, ch in enumerate(body))
    return body + chr(65 + s % 26)


def iban_it(rng):
    """IT + 2 check digits (ISO 13616, mod 97) + CIN letter + ABI + CAB + 12-digit account"""
    abi, cab = f"{rng.randrange(1000, 99999):05d}", f"{rng.randrange(1000, 99999):05d}"
    conto = "".join(rng.choice(string.digits) for _ in range(12))
    odd = [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]
    bban = abi + cab + conto
    cin = chr(65 + sum((odd[int(ch)] if k % 2 == 0 else int(ch)) for k, ch in enumerate(bban)) % 26)
    bban = cin + bban
    num = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in bban + "IT00")
    return f"IT{98 - int(num) % 97:02d}{bban}"


def partita_iva(rng):
    d = [rng.randrange(10) for _ in range(10)]
    s = sum(x if k % 2 == 0 else (2 * x if 2 * x < 10 else 2 * x - 9) for k, x in enumerate(d))
    return "".join(map(str, d)) + str((10 - s % 10) % 10)


# (name as written on forms, province, cadastral code)
COMUNI = [("ROMA", "RM", "H501"), ("MILANO", "MI", "F205"), ("TORINO", "TO", "L219"), ("NAPOLI", "NA", "F839"), ("BOLOGNA", "BO", "A944"),
          ("FIRENZE", "FI", "D612"), ("PALERMO", "PA", "G273"), ("GENOVA", "GE", "D969"), ("VENEZIA", "VE", "L736"), ("VERONA", "VR", "L781"),
          ("BARI", "BA", "A662"), ("CATANIA", "CT", "C351"), ("PADOVA", "PD", "G224"), ("TRIESTE", "TS", "L424"), ("CAGLIARI", "CA", "B354"),
          ("MESSINA", "ME", "F158"), ("SALERNO", "SA", "H703"), ("BERGAMO", "BG", "A794"), ("BRESCIA", "BS", "B157"), ("FERRARA", "FE", "D548"),
          ("MODENA", "MO", "F257"), ("PARMA", "PR", "G337"), ("PRATO", "PO", "G999"), ("LECCE", "LE", "E506"), ("TARANTO", "TA", "L049"),
          ("RIMINI", "RN", "H294"), ("PISA", "PI", "G702"), ("LIVORNO", "LI", "E625"), ("LUCCA", "LU", "E715"), ("SIENA", "SI", "I726"),
          ("AREZZO", "AR", "A390"), ("PERUGIA", "PG", "G478"), ("TERNI", "TR", "L117"), ("ANCONA", "AN", "A271"), ("PESCARA", "PE", "G482"),
          ("MACERATA", "MC", "E783"), ("BOLZANO", "BZ", "A952"), ("TRENTO", "TN", "L378"), ("UDINE", "UD", "L483"), ("COMO", "CO", "C933"),
          ("MONZA", "MB", "F704"), ("NOVARA", "NO", "F952"), ("VICENZA", "VI", "L840"), ("TREVISO", "TV", "L407"), ("REGGIO CALABRIA", "RC", "H224"),
          ("COSENZA", "CS", "D086"), ("SASSARI", "SS", "I452"), ("POTENZA", "PZ", "G942"), ("CAMPOBASSO", "CB", "B519"), ("L'AQUILA", "AQ", "A345"),
          ("AOSTA", "AO", "A326")]
SURNAMES = ["Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno", "Gallo", "Conti",
            "De Luca", "Mancini", "Costa", "Giordano", "Rizzo", "Lombardi", "Moretti", "Barbieri", "Fontana", "Santoro", "Mariani", "Rinaldi",
            "Caruso", "Ferrara", "Galli", "Martini", "Leone", "Longo", "Gentile", "Martinelli", "Vitale", "Lombardo", "Serra", "Coppola",
            "De Santis", "D'Angelo", "Marchetti", "Parisi", "Villa", "Conte", "Ferraro", "Ferri", "Fabbri", "Bianco", "Marini", "Grasso",
            "Valentini", "Messina", "Sala", "De Angelis", "Gatti", "Pellegrini", "Palumbo", "Sanna", "Farina", "Rizzi", "Monti", "Cattaneo",
            "Morelli", "Amato", "Silvestri", "Mazza", "Testa", "Grassi", "Pellegrino", "Carbone", "Giuliani", "Benedetti", "Barone", "Rossetti",
            "Caputo", "Montanari", "Guerra", "Palmieri", "Bernardi", "Martino", "Fiore", "De Rosa", "Ferretti", "Bellini", "Basile", "Riva",
            "Donati", "Piras", "Vitali", "Battaglia", "Sartori", "Neri", "Costantini", "Milani", "Pagano", "Ruggiero", "Sorrentino", "D'Amico",
            "Orlando", "Damico", "Negri", "Kofler", "Zanetti", "Brambilla", "Melis", "Porcu", "Sciortino", "Cuomo", "Iannone", "Tedesco"]
NAMES_M = ["Giuseppe", "Giovanni", "Antonio", "Mario", "Luigi", "Francesco", "Angelo", "Vincenzo", "Pietro", "Salvatore", "Carlo", "Franco",
           "Domenico", "Bruno", "Paolo", "Michele", "Giorgio", "Aldo", "Sergio", "Luciano", "Stefano", "Roberto", "Alessandro", "Andrea",
           "Marco", "Matteo", "Lorenzo", "Leonardo", "Gabriele", "Riccardo", "Tommaso", "Edoardo", "Federico", "Davide", "Simone", "Nicola",
           "Emanuele", "Filippo", "Jacopo", "Alberto", "Massimo", "Fabio", "Claudio", "Enrico", "Raffaele", "Gennaro", "Ciro", "Rocco"]
NAMES_F = ["Maria", "Anna", "Giuseppina", "Rosa", "Angela", "Giovanna", "Teresa", "Lucia", "Carmela", "Caterina", "Francesca", "Antonietta",
           "Laura", "Paola", "Elena", "Chiara", "Sara", "Giulia", "Martina", "Sofia", "Aurora", "Alice", "Ginevra", "Emma", "Giorgia",
           "Beatrice", "Alessia", "Federica", "Valentina", "Silvia", "Roberta", "Cristina", "Simona", "Monica", "Barbara", "Daniela",
           "Raffaella", "Ilaria", "Elisa", "Serena", "Noemi", "Arianna", "Veronica", "Margherita", "Ludovica", "Benedetta", "Vittoria"]
BANKS = ["INTESA SANPAOLO", "UNICREDIT", "BANCO BPM", "BPER BANCA", "CREDIT AGRICOLE ITALIA", "BANCA MONTE DEI PASCHI DI SIENA",
         "BANCA POPOLARE DI SONDRIO", "BANCA SELLA", "CREDEM", "BANCA MEDIOLANUM", "POSTE ITALIANE", "BANCA DI CREDITO COOPERATIVO",
         "FINECO BANK", "BANCA IFIS", "CASSA DI RISPARMIO DI BOLZANO", "BANCA POPOLARE DI CIVIDALE"]
