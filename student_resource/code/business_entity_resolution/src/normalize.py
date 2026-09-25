"""Stage 1 - country-agnostic normalisation of business names and addresses.

Nothing in this module branches on `country`. All maps below are hand-written code
(abbreviations, legal forms, a Brahmic-script transliteration table); no external data.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Brahmic-script transliteration.  Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
# Tamil, Telugu, Kannada and Malayalam share the ISCII-derived layout: the same
# offset inside each 0x80-wide Unicode block encodes the same phoneme, so one
# offset table transliterates all of them.
# ---------------------------------------------------------------------------
_BRAHMIC_BLOCKS = range(0x0900, 0x0D80, 0x80)

_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
               0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
               0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
               0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
               0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
               0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r",
               0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x62: "li", 0x63: "li"}
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA = 0x4D
_SILENT = {0x3C, 0x3D, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x70, 0x71}


def _brahmic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp < 0x0D80:
        return cp & 0x7F
    return None


def transliterate(s):
    """Romanise any Brahmic-script characters in `s`; other characters pass through."""
    if not s or all(ord(c) < 0x0900 or ord(c) >= 0x0D80 for c in s):
        return s
    out = []
    n = len(s)
    i = 0
    while i < n:
        off = _brahmic_offset(s[i])
        if off is None:
            out.append(s[i])
            i += 1
            continue
        if off in _CONSONANTS:
            out.append(_CONSONANTS[off])
            j = i + 1
            while j < n and _brahmic_offset(s[j]) in _SILENT:  # nukta etc.
                j += 1
            nxt = _brahmic_offset(s[j]) if j < n else None
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                i = j + 1
            elif nxt == _VIRAMA:
                i = j + 1
            else:
                # inherent vowel, dropped word-finally (schwa deletion)
                if nxt is not None and (nxt in _CONSONANTS or nxt in _SIGNS):
                    out.append("a")
                i = j
            continue
        if off in _VOWELS:
            out.append(_VOWELS[off])
        elif off in _SIGNS:
            out.append(_SIGNS[off])
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        elif off in _MATRAS:
            out.append(_MATRAS[off])
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Shared cleaning
# ---------------------------------------------------------------------------
_NULLS = {"null", "none", "nan", "n/a", "na", "-"}


def _ascii_fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def base_clean(s):
    """NFKD + accent strip, transliterate, lowercase, & -> and, dots removed,
    other punctuation -> space, whitespace collapsed."""
    if not s:
        return ""
    s = _ascii_fold(transliterate(s)).lower()
    s = s.replace("&", " and ")
    s = s.replace(".", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(t for t in s.split() if t not in _NULLS)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
_LEGAL_MAP = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd", "private": "pvt",
    "company": "co", "companies": "co", "compagnie": "cie", "pvtltd": "pvt ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "ltd": "ltd", "inc": "inc",
    "corp": "corp", "pvt": "pvt", "pte": "pvt", "prv": "pvt", "priv": "pvt", "co": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl", "cie": "cie",
    "sci": "sci", "snc": "snc",
    # romanised Indic-script spellings of the same legal words
    "praivet": "pvt", "prayvet": "pvt", "piraivet": "pvt", "pirayvet": "pvt", "limitet": "ltd", "limited": "ltd", "limiteda": "ltd",
    "elelpi": "llp", "elelapi": "llp", "karporeshan": "corp", "kampani": "co",
}
LEGAL_TOKENS = frozenset(_LEGAL_MAP.values()) | {"and", "the", "of", "ms", "m", "s"}
# honorific / filler prefixes that the sources add or drop freely
_FILLER = frozenset({"sri", "shri", "shree", "sree", "the", "ms"})

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "$": "s", "@": "a"})
_DBA_RE = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|dba|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|"
    r"trading\s+as|doing\s+business\s+as|formerly\s+known\s+as|also\s+known\s+as)\b",
    re.IGNORECASE)
_PRA_LI_RE = re.compile(r"\bpra\s+li\b")  # romanised Indic abbreviation of "pvt ltd"
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|us|io|co)\b",
                        re.IGNORECASE)


def _deleet(tok):
    # only tokens that mix letters and digits are rewritten (c0rp, g1oba, 5ervices)
    if tok.isalpha() or tok.isdigit():
        return tok
    if any(c.isalpha() for c in tok):
        return tok.translate(_LEET)
    return tok


def _clean_name_part(s):
    s = s.strip()
    m = _DOMAIN_RE.match(s)
    if m:
        s = m.group(1).replace("-", " ")
    s = base_clean(s.replace("#", " "))
    s = _PRA_LI_RE.sub(" pvt ltd ", s)
    toks = []
    for t in s.split():
        t = _deleet(t)
        toks.extend(_LEGAL_MAP.get(t, t).split())
    return " ".join(toks)


def normalize_name(raw):
    """Returns (full_name, core_name, variants) where variants are the DBA/aka parts."""
    raw = _ascii_fold(transliterate(raw or ""))
    parts = [p for p in _DBA_RE.split(raw) if p and p.strip()]
    variants = [v for v in (_clean_name_part(p.replace("/", " ")) for p in parts) if v]
    full = " ".join(variants)
    core = core_of(full)
    return full, core, variants


def core_of(name):
    toks = [t for t in name.split() if t not in LEGAL_TOKENS and t not in _FILLER]
    return " ".join(toks) if toks else name


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
_ADDR_MAP = {
    "road": "rd", "street": "st", "str": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "bld": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
    "square": "sq", "highway": "hwy", "parkway": "pkwy", "circle": "cir", "terrace": "ter",
    "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "fl", "near": "nr",
    "opposite": "opp", "opp": "opp", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "saint": "st", "sainte": "ste", "route": "rte", "chemin": "ch",
    "impasse": "imp", "allee": "all", "rue": "r", "faubourg": "fbg", "sector": "sec",
    "nagar": "ngr", "colony": "col", "marg": "mg", "cross": "crs", "main": "mn",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th", "ground": "gr", "gf": "gr fl",
}
# house-number prefixes that sources add or drop freely
_ADDR_NOISE = frozenset({"no", "number", "hno", "h", "door", "unit", "plot", "house", "flat", "nos"})
_NUM_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")


def normalize_address(raw):
    """Returns (address, numbers(sorted unique str list), postal or '')."""
    s = base_clean(raw)
    # leading zeros are source formatting noise (0201 ~ 201)
    toks = [(t.lstrip("0") or "0") if t.isdigit() else _ADDR_MAP.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in _ADDR_NOISE]
    addr = " ".join(toks)
    nums = sorted(set(_NUM_RE.findall(addr)))
    postal = _POSTAL_RE.findall(addr)
    return addr, nums, (postal[-1] if postal else "")


# ---------------------------------------------------------------------------
# Phonetic skeleton: makes romanised Indic spellings and English spellings of the same
# word collide (gud treding ~ good trading, purotyuchar ~ producer).  Voiced/unvoiced
# pairs merged, sibilants merged, vowels / y / h dropped, repeats collapsed.
# ---------------------------------------------------------------------------
_PHON_SUBS = [(re.compile(p), r) for p, r in [
    (r"tion", "sn"), (r"sion", "sn"), (r"chh", "s"), (r"ch", "s"), (r"sh", "s"), (r"ph", "p"),
    (r"ck", "k"), (r"c(?=[eiy])", "s"), (r"c", "k"), (r"q", "k"), (r"x", "ks"), (r"z", "s"),
    (r"w", "v"), (r"f", "p"), (r"b", "p"), (r"d", "t"), (r"g", "k"), (r"j", "s"),
    (r"[aeiouyh]", ""), (r"(.)\1+", r"\1"),
]]


def phonetic(text):
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.append(tok)
            continue
        for pat, rep in _PHON_SUBS:
            tok = pat.sub(rep, tok)
        if tok:
            out.append(tok)
    return " ".join(out)


def normalize_record(name, address):
    full, core, variants = normalize_name(name)
    addr, nums, postal = normalize_address(address)
    return full, core, "|".join(variants), addr, " ".join(nums), postal, phonetic(core)
