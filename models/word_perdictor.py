from __future__ import annotations

import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config.env import get_settings


TOKEN_RE = re.compile(r"[\u0900-\u097F]+|[A-Za-z]+|[0-9]+")
VALID_TOKEN_RE = re.compile(r"[\u0900-\u097Fa-z0-9]", re.IGNORECASE)


def _normalize_token(token: str) -> str:
	normalized = unicodedata.normalize("NFKC", token or "")
	normalized = normalized.replace("\u200d", "").strip().lower()
	normalized = re.sub(r"\s+", "", normalized)
	return normalized


def _is_valid_prediction_token(token: str) -> bool:
	if not token or len(token) > 40:
		return False
	if token in {"<s>", "</s>"}:
		return False
	return bool(VALID_TOKEN_RE.search(token))


def _pairs_to_count_dict(pairs: Iterable[Tuple[object, object]]) -> Dict[str, float]:
	counts: Dict[str, float] = {}
	for pair in pairs:
		if not isinstance(pair, (tuple, list)) or len(pair) != 2:
			continue

		word = _normalize_token(str(pair[0]))
		if not _is_valid_prediction_token(word):
			continue

		try:
			count = float(pair[1])
		except (TypeError, ValueError):
			continue

		if count <= 0:
			continue
		counts[word] = counts.get(word, 0.0) + count

	return counts


def _counts_to_probs(counts: Dict[str, float]) -> Dict[str, float]:
	if not counts:
		return {}
	total = float(sum(counts.values()))
	if total <= 0.0:
		return {}
	return {word: value / total for word, value in counts.items()}


class WordPredictor:
	def __init__(self) -> None:
		settings = get_settings()
		base_dir = Path(__file__).resolve().parent.parent
		model_dir = base_dir / settings.model_dicts_dir

		self.ngram_model_path = model_dir / "nepali_trigram_model.pkl"
		self.lexicon_path = model_dir / "lexicon_freq.tsv"

		self.ng = self._load_ngram_model()
		self.lexicon = self._load_lexicon()
		self._max_lexicon_count = max(self.lexicon.values(), default=1)
		self._top_lexicon_words = [
			word
			for word, _ in sorted(self.lexicon.items(), key=lambda item: item[1], reverse=True)[:1500]
		]

	def _load_ngram_model(self) -> Dict[str, object]:
		with self.ngram_model_path.open("rb") as file:
			payload = pickle.load(file)

		tri_raw = payload.get("tri_next_words", {})
		bi_raw = payload.get("bi_next_words", {})

		tri: Dict[Tuple[str, str], Dict[str, float]] = {}
		for key, values in dict(tri_raw).items():
			if not isinstance(key, (tuple, list)) or len(key) != 2:
				continue
			left = _normalize_token(str(key[0]))
			right = _normalize_token(str(key[1]))
			if not _is_valid_prediction_token(left) or not _is_valid_prediction_token(right):
				continue

			counts = _pairs_to_count_dict(values)
			if counts:
				tri[(left, right)] = counts

		bi: Dict[Tuple[str], Dict[str, float]] = {}
		for key, values in dict(bi_raw).items():
			if not isinstance(key, (tuple, list)) or len(key) != 1:
				continue
			token = _normalize_token(str(key[0]))
			if not _is_valid_prediction_token(token):
				continue

			counts = _pairs_to_count_dict(values)
			if counts:
				bi[(token,)] = counts

		start = _normalize_token(str(payload.get("START", "<s>"))) or "<s>"
		return {"tri": tri, "bi": bi, "START": start}

	def _load_lexicon(self) -> Dict[str, int]:
		lexicon: Dict[str, int] = {}
		with self.lexicon_path.open("r", encoding="utf-8") as file:
			next(file)
			for line in file:
				word_raw, count = line.rstrip("\n").split("\t")
				word = _normalize_token(word_raw)
				if not _is_valid_prediction_token(word):
					continue
				lexicon[word] = int(count)
		return lexicon

	def _tokenize_context(self, text: str) -> List[str]:
		tokens = [_normalize_token(token) for token in TOKEN_RE.findall(text)]
		return [token for token in tokens if _is_valid_prediction_token(token)]

	def _lexicon_prior(self, word: str) -> float:
		freq = float(self.lexicon.get(word, 0))
		if freq <= 0.0:
			return 0.0
		return math.log1p(freq) / math.log1p(float(self._max_lexicon_count))

	def _add_prob_scores(
		self,
		scores: Dict[str, float],
		candidates: set[str],
		counts: Dict[str, float],
		weight: float,
	) -> None:
		for word, prob in _counts_to_probs(counts).items():
			scores[word] += weight * prob
			candidates.add(word)

	def _context_aware_scores(self, context: List[str]) -> Dict[str, float]:
		scores: Dict[str, float] = defaultdict(float)
		candidates: set[str] = set()
		tri = self.ng["tri"]
		bi = self.ng["bi"]

		# Blend multiple recent bigrams/trigrams instead of only last token window.
		if len(context) >= 2:
			start_idx = max(1, len(context) - 4)
			for idx in range(start_idx, len(context)):
				pair = (context[idx - 1], context[idx])
				recency = (len(context) - 1) - idx
				weight = 1.0 * (0.67 ** recency)
				self._add_prob_scores(scores, candidates, tri.get(pair, {}), weight)

		if context:
			for offset, token in enumerate(reversed(context[-4:])):
				weight = 0.72 * (0.62 ** offset)
				self._add_prob_scores(scores, candidates, bi.get((token,), {}), weight)

		if not context:
			self._add_prob_scores(scores, candidates, bi.get((self.ng["START"],), {}), 0.9)

		# If context is very sparse, seed candidates from frequent lexicon words.
		if not candidates:
			for rank, word in enumerate(self._top_lexicon_words[:200]):
				scores[word] += 0.03 / (1.0 + rank)
				candidates.add(word)

		context_tail = context[-12:]
		context_counter = Counter(context_tail)
		last_token = context[-1] if context else ""

		for word in list(candidates):
			score = scores[word]

			# Add a lexical prior so context and corpus frequency both matter.
			score += 0.12 * self._lexicon_prior(word)

			# Boost words that are topical to the current context.
			if word in context_counter:
				score += min(0.18, 0.07 * context_counter[word])

			# Avoid immediate word loops.
			if last_token and word == last_token:
				score *= 0.35

			scores[word] = score

		return scores

	def _normalize(self, counts: Dict[str, int], top_k: int) -> List[Dict[str, float]]:
		if not counts or top_k < 1:
			return []

		sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
		filtered_items = [(word, score) for word, score in sorted_items if score > 0][: max(20, top_k * 6)]

		if not filtered_items:
			filtered_items = sorted_items[: max(20, top_k * 6)]

		words = [word for word, _ in filtered_items]
		score_values = np.asarray([float(score) for _, score in filtered_items], dtype=np.float64)

		temperature = 0.85
		score_values = score_values / temperature
		max_value = float(np.max(score_values))
		exps = np.exp(score_values - max_value)
		sum_exps = float(np.sum(exps))

		if sum_exps <= 0.0:
			probs = np.ones_like(exps) / float(len(exps))
		else:
			probs = exps / sum_exps

		top_idx = np.argsort(-probs)[:top_k]

		return [
			{"word": words[int(idx)], "probability": round(float(probs[int(idx)]), 6)}
			for idx in top_idx
		]

	def predict(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
		context = self._tokenize_context(text.strip())
		if top_k < 1:
			top_k = 1

		scores = self._context_aware_scores(context)
		if scores:
			return self._normalize(scores, top_k)

		fallback = {word: float(self.lexicon.get(word, 0)) for word in self._top_lexicon_words[: top_k * 10]}
		return self._normalize(fallback, top_k)
