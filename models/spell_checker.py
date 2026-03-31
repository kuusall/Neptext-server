from __future__ import annotations

import heapq
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

from config.env import get_settings


DEV_WORD_RE = re.compile(r"^[\u0900-\u0963\u0966-\u097F]+$")
WORD_RE = re.compile(r"^([\u0900-\u0963\u0966-\u097F]+|[०-९0-9]+|[A-Za-z]+)$")
PUNCT_TOKENS = {
	"।",
	",",
	".",
	"!",
	"?",
	":",
	";",
	")",
	"]",
	"}",
	'"',
	"'",
	"-",
	"(",
	"[",
	"{",
}
COMMON_FUNCTION_WORDS = {
	"छ",
	"छन्",
	"हो",
	"थियो",
	"थिए",
	"हुन्छ",
	"गर्छ",
	"गरे",
	"गरेको",
}


def is_word(token: str) -> bool:
	return bool(WORD_RE.match(token))


def is_dev_word(token: str) -> bool:
	return bool(DEV_WORD_RE.match(token))


def tokenize(text: str) -> List[str]:
	spaced = re.sub(r"([।\.\,\!\?\:\;\(\)\[\]\{\}\"'\-])", r" \1 ", text)
	spaced = re.sub(r"\s+", " ", spaced).strip()
	if not spaced:
		return []
	return spaced.split(" ")


def join_tokens(tokens: List[str]) -> str:
	no_space_before = {",", ".", ")", "]", "}", ":", ";", "।", "!", "?", '"', "'"}
	no_space_after = {"(", "[", "{", '"', "'"}

	out: List[str] = []
	for index, token in enumerate(tokens):
		if index == 0:
			out.append(token)
			continue

		prev = tokens[index - 1]
		if token in no_space_before or prev in no_space_after:
			out.append(token)
		else:
			out.append(" " + token)

	return "".join(out)


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
			ins = cur[j - 1] + 1
			dele = prev[j] + 1
			sub = prev[j - 1] + (ca != cb)
			cur[j] = min(ins, dele, sub)
			row_min = min(row_min, cur[j])
		if row_min > max_dist:
			return max_dist + 1
		prev = cur

	return prev[-1]


class SpellChecker:
	def __init__(self) -> None:
		settings = get_settings()
		base_dir = Path(__file__).resolve().parent.parent
		model_dir = base_dir / settings.model_dicts_dir

		self.lexicon_path = model_dir / "lexicon_freq.tsv"
		self.deletes_map_path = model_dir / "deletes_map.pkl"
		self.ngram_model_path = model_dir / "nepali_trigram_model.pkl"

		self.lexicon = self._load_lexicon()
		self.max_edit, self.deletes_map = self._load_deletes_map()
		self.ng = self._load_ngram_model()

	def _load_lexicon(self) -> Dict[str, int]:
		lexicon: Dict[str, int] = {}
		with self.lexicon_path.open("r", encoding="utf-8") as file:
			next(file)
			for line in file:
				word, count = line.rstrip("\n").split("\t")
				lexicon[word] = int(count)
		return lexicon

	def _load_deletes_map(self) -> Tuple[int, Dict[str, List[str]]]:
		with self.deletes_map_path.open("rb") as file:
			payload = pickle.load(file)
		return payload["max_edit"], payload["deletes_map"]

	def _load_ngram_model(self) -> Dict[str, object]:
		with self.ngram_model_path.open("rb") as file:
			payload = pickle.load(file)
		tri = {k: dict(v) for k, v in payload["tri_next_words"].items()}
		bi = {k: dict(v) for k, v in payload["bi_next_words"].items()}
		return {"tri": tri, "bi": bi, "START": payload["START"]}

	def _generate_candidates(self, misspelled: str) -> List[Tuple[str, int]]:
		if not is_dev_word(misspelled):
			return []
		if misspelled in self.lexicon:
			return [(misspelled, 0)]

		deletes = {misspelled}
		frontier = {misspelled}
		for _ in range(self.max_edit):
			next_frontier = set()
			for word in frontier:
				if len(word) <= 1:
					continue
				for idx in range(len(word)):
					deleted = word[:idx] + word[idx + 1 :]
					if deleted not in deletes:
						deletes.add(deleted)
						next_frontier.add(deleted)
			frontier = next_frontier

		candidates: List[Tuple[str, int]] = []
		seen = set()

		for deleted in deletes:
			if deleted in self.lexicon and deleted not in seen:
				seen.add(deleted)
				candidates.append((deleted, levenshtein(misspelled, deleted, max_dist=self.max_edit)))

		for deleted in deletes:
			for candidate in self.deletes_map.get(deleted, []):
				if candidate in seen:
					continue
				seen.add(candidate)
				dist = levenshtein(misspelled, candidate, max_dist=self.max_edit)
				if dist <= self.max_edit:
					candidates.append((candidate, dist))

		return heapq.nsmallest(
			150,
			candidates,
			key=lambda item: (item[1], -self.lexicon.get(item[0], 0)),
		)

	def _lm_score(self, prev2: Tuple[str, str], prev1: str, candidate: str) -> float:
		tri = self.ng["tri"]
		bi = self.ng["bi"]

		tri_count = tri.get(prev2, {}).get(candidate, 0)
		if tri_count > 0:
			return math.log1p(tri_count)

		bi_count = bi.get((prev1,), {}).get(candidate, 0)
		if bi_count > 0:
			return 0.7 * math.log1p(bi_count)

		return 0.0

	def correct(
		self,
		text: str,
		suggest_only: bool = True,
		dist_weight: float = 3.2,
		margin: float = 0.8,
	) -> Tuple[str, List[Dict[str, object]]]:
		tokens = tokenize(text)
		corrected = tokens[:]
		edits: List[Dict[str, object]] = []

		positions = [idx for idx, token in enumerate(tokens) if is_word(token) and token not in PUNCT_TOKENS]
		words = [tokens[idx] for idx in positions]

		for word_index, pos in enumerate(positions):
			word = corrected[pos]

			if not is_dev_word(word):
				continue
			if len(word) == 1:
				continue
			if word in self.lexicon:
				continue

			prev1 = words[word_index - 1] if word_index > 0 else self.ng["START"]
			prev2 = words[word_index - 2] if word_index > 1 else self.ng["START"]

			candidates = self._generate_candidates(word)
			if not candidates:
				continue

			scored: List[Tuple[float, str, int]] = []
			for candidate, dist in candidates:
				lm = self._lm_score((prev2, prev1), prev1, candidate)
				freq_bonus = 0.18 * math.log1p(self.lexicon.get(candidate, 1))
				if candidate in COMMON_FUNCTION_WORDS:
					freq_bonus += 0.6
				score = lm + freq_bonus - dist_weight * dist
				scored.append((score, candidate, dist))

			scored.sort(reverse=True, key=lambda item: item[0])
			best = scored[0]
			second = scored[1] if len(scored) > 1 else None

			if best[2] != 1:
				continue
			if second and (best[0] - second[0] < margin):
				continue

			chosen = best[1]
			if not suggest_only:
				corrected[pos] = chosen
				words[word_index] = chosen

			edits.append(
				{
					"index": word_index,
					"from": word,
					"suggest": chosen,
					"edit_distance": best[2],
				}
			)

		return join_tokens(corrected), edits
