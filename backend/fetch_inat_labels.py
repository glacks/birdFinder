from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.inaturalist.org/v1"
USER_AGENT = "bird-finder-label-sync/0.1"


def http_get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = f"?{urlencode(params or {}, doseq=True)}" if params else ""
    url = f"{BASE_URL}{path}{query}"
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def normalize_english_label(class_name: str) -> str:
    return class_name.replace("_", " ")


def score_candidate(query_en: str, candidate: dict[str, Any]) -> int:
    query = query_en.lower().strip()
    common = str(candidate.get("preferred_common_name") or "").lower().strip()
    name = str(candidate.get("name") or "").lower().strip()
    score = 0
    if common == query:
        score += 10
    if query and query in common:
        score += 5
    if query and query in name:
        score += 2
    rank = str(candidate.get("rank") or "")
    if rank == "species":
        score += 3
    iconic = (candidate.get("iconic_taxon_name") or "").lower()
    if iconic == "aves":
        score += 3
    return score


def query_taxa_autocomplete(english_name: str) -> list[dict[str, Any]]:
    payload = http_get_json(
        "/taxa/autocomplete",
        {
            "q": english_name,
            "per_page": 15,
            "locale": "en",
            "taxon_id": 3,
        },
    )
    return payload.get("results") or []


def query_taxa_search(english_name: str) -> list[dict[str, Any]]:
    payload = http_get_json(
        "/taxa",
        {
            "q": english_name,
            "per_page": 15,
            "order_by": "observations_count",
            "taxon_id": 3,
            "is_active": "true",
        },
    )
    return payload.get("results") or []


def query_variants(english_name: str) -> list[str]:
    variants = [english_name]
    if " × " in english_name:
        variants.append(english_name.replace(" × ", " x "))
        variants.append(english_name.split(" × ", 1)[0])
    if " x " in english_name:
        variants.append(english_name.split(" x ", 1)[0])
    if "-" in english_name:
        variants.append(english_name.replace("-", " "))
    cleaned = []
    seen = set()
    for v in variants:
        vv = " ".join(v.split()).strip()
        if vv and vv not in seen:
            cleaned.append(vv)
            seen.add(vv)
    return cleaned


def find_best_taxon_id(english_name: str) -> int | None:
    results: list[dict[str, Any]] = []
    for query in query_variants(english_name):
        results.extend(query_taxa_autocomplete(query))
        results.extend(query_taxa_search(query))

    if not results:
        return None
    dedup = {}
    for c in results:
        cid = c.get("id")
        if cid is not None:
            dedup[cid] = c
    ranked = sorted(dedup.values(), key=lambda c: score_candidate(english_name, c), reverse=True)
    top = ranked[0]
    return top.get("id")


def get_taxon_names(taxon_id: int) -> tuple[str | None, str | None]:
    ko_payload = http_get_json(f"/taxa/{taxon_id}", {"locale": "ko"})
    en_payload = http_get_json(f"/taxa/{taxon_id}", {"locale": "en"})

    ko_results = ko_payload.get("results") or []
    en_results = en_payload.get("results") or []
    ko = ko_results[0].get("preferred_common_name") if ko_results else None
    en = en_results[0].get("preferred_common_name") if en_results else None
    return (ko, en)


def list_class_names(dataset_dir: Path) -> list[str]:
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Korean/English labels from iNaturalist")
    parser.add_argument("--dataset-dir", default="../data_raw")
    parser.add_argument("--output", default="artifacts/class_labels.json")
    parser.add_argument("--sleep-ms", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = list_class_names(dataset_dir)
    if args.limit > 0:
        class_names = class_names[: args.limit]
    labels: dict[str, dict[str, Any]] = {}

    for idx, class_name in enumerate(class_names, start=1):
        english_query = normalize_english_label(class_name)
        ko_name = english_query
        en_name = english_query
        taxon_id = None

        try:
            taxon_id = find_best_taxon_id(english_query)
            if taxon_id is not None:
                ko, en = get_taxon_names(taxon_id)
                if ko:
                    ko_name = ko
                if en:
                    en_name = en
        except Exception:
            pass

        labels[class_name] = {
            "ko": ko_name,
            "en": en_name,
            "display": f"{ko_name} ({en_name})",
            "inat_taxon_id": taxon_id,
        }
        print(f"[{idx}/{len(class_names)}] {class_name} -> {labels[class_name]['display']}")
        time.sleep(max(0, args.sleep_ms) / 1000.0)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
