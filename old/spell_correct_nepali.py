# Spell correction for Nepali text using a combination of edit distance and n-gram language model scoring.


import pickle, re, heapq, math
from collections import defaultdict
from typing import List, Tuple, Dict



NGRAM_MODEL_PATH = "../model_dicts/nepali_trigram_model.pkl.pkl"
DELETES_MAP_PATH = "../model_dicts/deletes_map.pkl"
LEXICON_PATH = "../model_dicts/lexicon_freq.tsv"

# Devanagari block includes danda "।" (U+0964). Exclude it from "word" matches.
DEV_WORD_RE = re.compile(r"^[\u0900-\u0963\u0966-\u097F]+$")  # exclude 0964-0965
WORD_RE = re.compile(r"^([\u0900-\u0963\u0966-\u097F]+|[०-९0-9]+|[A-Za-z]+)$")
PUNCT_TOKENS = {"।", ",", ".", "!", "?", ":", ";", ")", "]", "}", "”", "\"", "'", "’", "—", "-", "“", "(", "[", "{"}

def is_word(tok: str) -> bool:
    return bool(WORD_RE.match(tok))

def is_dev_word(tok: str) -> bool:
    return bool(DEV_WORD_RE.match(tok))

def tokenize(text: str) -> List[str]:
    # Force-separate common punctuation (including danda)
    text = re.sub(r"([।\.\,\!\?\:\;\(\)\[\]\{\}“”\"'’—\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Now split by spaces; punctuation stays as separate tokens
    return text.split(" ")

def join_tokens(tokens: List[str]) -> str:
    no_space_before = {",", ".", ")", "]", "}", ":", ";", "।", "!", "?", "”", "\"", "'", "’"}
    no_space_after  = {"(", "[", "{", "“", "\"", "'", "’"}
    out = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t); continue
        prev = tokens[i-1]
        if t in no_space_before:
            out.append(t)
        elif prev in no_space_after:
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)

def load_lexicon(path=LEXICON_PATH) -> Dict[str,int]:
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            lex[w] = int(c)
    return lex

def load_deletes_map(path=DELETES_MAP_PATH):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["max_edit"], obj["deletes_map"]

def load_ngram_model(path=NGRAM_MODEL_PATH):
    with open(path, "rb") as f:
        m = pickle.load(f)
    tri = {k: dict(v) for k, v in m["tri_next_words"].items()}
    bi  = {k: dict(v) for k, v in m["bi_next_words"].items()}
    return {"tri": tri, "bi": bi, "START": m["START"]}

def levenshtein(a: str, b: str, max_dist: int = 3) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            cur[j] = min(ins, dele, sub)
            row_min = min(row_min, cur[j])
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]

def generate_candidates(miss: str, lexicon: Dict[str,int], deletes_map: Dict[str, List[str]], max_edit: int) -> List[Tuple[str,int]]:
    if not is_dev_word(miss):
        return []
    if miss in lexicon:
        return [(miss, 0)]

    dels = {miss}
    queue = {miss}
    for _ in range(max_edit):
        nq = set()
        for w in queue:
            if len(w) <= 1:
                continue
            for i in range(len(w)):
                d = w[:i] + w[i+1:]
                if d not in dels:
                    dels.add(d)
                    nq.add(d)
        queue = nq

    # also treat deletes themselves as possible candidates (handles very short words like 'छ')
    cands = []
    seen = set()
    for d in dels:
        if d in lexicon and d not in seen:
            seen.add(d)
            cands.append((d, levenshtein(miss, d, max_dist=max_edit)))
    for d in dels:
        for w in deletes_map.get(d, []):
            if w in seen:
                continue
            seen.add(w)
            dist = levenshtein(miss, w, max_dist=max_edit)
            if dist <= max_edit:
                cands.append((w, dist))

    cands.sort(key=lambda x: (x[1], -lexicon.get(x[0], 0)))
    return cands[:150]

def lm_score(ng, prev2: Tuple[str,str], prev1: str, cand: str) -> float:
    tri = ng["tri"]; bi = ng["bi"]
    c3 = tri.get(prev2, {}).get(cand, 0)
    if c3 > 0:
        return math.log1p(c3)
    c2 = bi.get((prev1,), {}).get(cand, 0)
    if c2 > 0:
        return 0.7 * math.log1p(c2)
    return 0.0

COMMON_FUNCTION_WORDS = {"छ", "छन्", "हो", "थियो", "थिए", "हुन्छ", "गर्छ", "गरे", "गरेको"}

def correct_sentence(
    text: str,
    lexicon: Dict[str,int],
    deletes_map: Dict[str, List[str]],
    max_edit: int,
    ng,
    *,
    dist_weight: float = 3.2,
    margin: float = 0.8,
    suggest_only: bool = True,
) -> Tuple[str, List[Dict]]:
    toks = tokenize(text)
    edits = []
    corrected = toks[:]

    word_positions = [i for i,t in enumerate(toks) if is_word(t) and t not in PUNCT_TOKENS]
    words = [toks[i] for i in word_positions]

    for wi, pos in enumerate(word_positions):
        w = corrected[pos]

        if not is_dev_word(w):
            continue

        # SAFETY: don't touch single-character tokens
        if len(w) == 1:
            continue

        if w in lexicon:
            continue

        prev1 = words[wi-1] if wi-1 >= 0 else ng["START"]
        prev2 = words[wi-2] if wi-2 >= 0 else ng["START"]

        cands = generate_candidates(w, lexicon, deletes_map, max_edit)
        if not cands:
            continue

        scored = []
        for cand, dist in cands:
            lm = lm_score(ng, (prev2, prev1), prev1, cand)
            freq_bonus = 0.18 * math.log1p(lexicon.get(cand, 1))

            # Prefer common function words slightly (prevents "राम्रो काम" type mistakes)
            if cand in COMMON_FUNCTION_WORDS:
                freq_bonus += 0.6

            score = lm + freq_bonus - dist_weight * dist
            scored.append((score, cand, dist))

        scored.sort(reverse=True, key=lambda x: x[0])
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None

        # Only allow distance 1 corrections by default (more realistic typos)
        if best[2] != 1:
            continue

        if second and (best[0] - second[0] < margin):
            continue

        if suggest_only:
            edits.append({"index": wi, "from": w, "suggest": best[1], "edit_distance": best[2]})
        else:
            corrected[pos] = best[1]
            words[wi] = best[1]
            edits.append({"index": wi, "from": w, "to": best[1], "edit_distance": best[2]})

    return join_tokens(corrected), edits

if __name__ == "__main__":
    print("Loading resources...")
    lex = load_lexicon()
    max_edit, dmap = load_deletes_map()
    ng = load_ngram_model()

    while True:
        try:
            s = input("\nType a Nepali sentence (Devanagari). Enter to exit:\n> ").strip()
        except EOFError:
            break
        if not s:
            break

        out, edits = correct_sentence(s, lex, dmap, max_edit, ng, suggest_only=True)
        print("\nText:\n", out)
        if edits:
            print("\nSuggestions:")
            for e in edits:
                print(f" - {e['from']}  ->  {e['suggest']}  (d={e['edit_distance']})")
        else:
            print("\nNo suggestions.")