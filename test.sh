#!/usr/bin/env bash

set -u

BASE_URL="${BASE_URL:-https://neptext-server-production.up.railway.app}"
API_PREFIX="${API_PREFIX:-}"

PASS_COUNT=0
FAIL_COUNT=0

if ! command -v curl >/dev/null 2>&1; then
	echo "Error: curl is required but not installed."
	exit 1
fi

print_result() {
	local ok="$1"
	local name="$2"
	local details="$3"

	if [ "$ok" -eq 0 ]; then
		echo "[PASS] $name"
		PASS_COUNT=$((PASS_COUNT + 1))
	else
		echo "[FAIL] $name"
		echo "       $details"
		FAIL_COUNT=$((FAIL_COUNT + 1))
	fi
}

run_test() {
	local name="$1"
	local method="$2"
	local path="$3"
	local data="$4"
	local expected_status="$5"
	local expected_text="$6"

	local url="${BASE_URL}${API_PREFIX}${path}"
	local response
	local body
	local status

	if [ -n "$data" ]; then
		response=$(curl -sS -X "$method" "$url" -H "Content-Type: application/json" -d "$data" -w "\n%{http_code}")
	else
		response=$(curl -sS -X "$method" "$url" -w "\n%{http_code}")
	fi

	body="$(echo "$response" | sed '$d')"
	status="$(echo "$response" | tail -n 1)"

	if [ "$status" != "$expected_status" ]; then
		print_result 1 "$name" "Expected HTTP $expected_status, got $status. Body: $body"
		return
	fi

	if [[ "$body" != *"$expected_text"* ]]; then
		print_result 1 "$name" "Expected response to contain '$expected_text'. Body: $body"
		return
	fi

	print_result 0 "$name" ""
}

echo "Testing API at ${BASE_URL}${API_PREFIX}"
echo

run_test \
	"Health Check" \
	"GET" \
	"/health" \
	"" \
	"200" \
	'"status":"ok"'

run_test \
	"Sentiment Analysis" \
	"POST" \
	"/sentiment" \
	'{"text":"यो movie ramro cha 😊 10/10"}' \
	"200" \
	'"sentiment"'

run_test \
	"Spell Correction Suggest-Only" \
	"POST" \
	"/spell-correct" \
	'{"text":"म नेपाल जाान्छु।","suggest_only":true}' \
	"200" \
	'"suggestions"'

run_test \
	"Spell Correction Auto-Correct" \
	"POST" \
	"/spell-correct" \
	'{"text":"म नेपाल जाान्छु।","suggest_only":false}' \
	"200" \
	'"corrected_text"'

run_test \
	"Word Prediction" \
	"POST" \
	"/word-predict" \
	'{"text":"नेपाल एक","top_k":5}' \
	"200" \
	'"predictions"'

echo
echo "Summary: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"

if [ "$FAIL_COUNT" -gt 0 ]; then
	exit 1
fi

exit 0
