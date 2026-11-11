"""Download of the PragTic material database (https://www.pragtic.com/vmat.php, J. Papuga, CTU).

The database has been read-only since 2014. The page shows one record; the records of one designation
are browsed with the buttons < > within a PHP session. Procedure: each designation from the list → "Show" →
the 1st record → then all the records in turn. The raw HTML is stored in a cache (repeated runs download only the missing parts),
the result is written to `data/pragtic_vmat.json`.

    python -m fea_core.fatigue.pragtic_fetch [--cache scratch/pragtic_raw] [--parse-only]
"""
from __future__ import annotations

import argparse
import html
import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://www.pragtic.com/vmat.php"
OUT = Path(__file__).with_name("data") / "pragtic_vmat.json"
RE_OPTION = re.compile(r'<option value="(\d+)">([^<]*)</option>')
RE_POS = re.compile(r"<b>(\d+)\. record from (\d+) records of (.*?) \(ID=(\d+)\)</b>")


class Session:
    def __init__(self, delay: float = 0.25):
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.opener.addheaders = [("User-Agent", "FEAStudio/1.0 (material import)")]
        self.delay = delay

    def post(self, **form) -> str:
        data = urllib.parse.urlencode(form).encode()
        for attempt in range(4):
            try:
                time.sleep(self.delay)
                with self.opener.open(URL, data=data, timeout=60) as r:
                    return r.read().decode("utf-8", "replace")
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("not reachable")

    def get(self) -> str:
        with self.opener.open(URL, timeout=60) as r:
            return r.read().decode("utf-8", "replace")


def position(page: str) -> tuple[int, int, str, int]:
    m = RE_POS.search(page)
    if not m:
        raise ValueError("the page has no record")
    return int(m[1]), int(m[2]), html.unescape(m[3]), int(m[4])


def download(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    s = Session()
    options = RE_OPTION.findall(s.get())
    print(f"{len(options)} marks in the list")
    done_marks: set[str] = set()
    index = json.loads((cache / "index.json").read_text()) if (cache / "index.json").exists() else {}
    for oi, (oid, mark) in enumerate(options):
        if index.get(oid) and all((cache / f"{rid}.html").exists() for rid in index[oid]):
            continue
        page = s.post(material=oid, send="Show")
        k, n, mark_page, rid = position(page)
        while k > 1:                                    # to the first record of the designation
            page = s.post(send="<")
            k, n, _, rid = position(page)
        ids = []
        while True:
            (cache / f"{rid}.html").write_text(page, encoding="utf-8")
            ids.append(rid)
            if k >= n:
                break
            page = s.post(send=">")
            k2, n, _, rid = position(page)
            if k2 != k + 1:
                raise RuntimeError(f"{mark}: unexpected order of records {k} → {k2}")
            k = k2
        index[oid] = ids
        (cache / "index.json").write_text(json.dumps(index))
        done_marks.add(mark_page)
        print(f"[{oi + 1}/{len(options)}] {mark_page}: {len(ids)} records", flush=True)


# ---------------------------------------------------------------------- parsing
def _text(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _num(s: str):
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return s or None


def _cell(page: str, label: str) -> str:
    m = re.search(rf"<td[^>]*>{re.escape(label)}</td>\s*<td[^>]*>(.*?)</td>", page, re.S)
    v = _text(m[1]) if m else ""
    return "" if v == "---" else v


def parse_record(page: str) -> dict:
    k, n, mark, rid = position(page)
    rec = {"id": rid, "mark": mark,
           "alt_mark": _cell(page, "Alternate mark:"), "standard": _cell(page, "Standard:"),
           "group": _cell(page, "Group:"), "treatment": _cell(page, "Treatment:"),
           "comment": _cell(page, "Commentary:"), "references": [], "data": {}}
    m = re.search(r"References:</td>\s*<td[^>]*>(.*?)</td>", page, re.S)
    if m:
        for li in re.findall(r"<li>(.*?)</li>", m[1], re.S):
            t = _text(re.sub(r"<b>\s*<a [^>]*>Manager</a>:</b>.*", "", li, flags=re.S))
            if t:
                rec["references"].append(t)
    for title, body in re.findall(r'<th colspan=4>([^<]+)</th>(.*?)</table>', page, re.S):
        rows = {}
        for mk, val, unit, desc in re.findall(
                r"<tr>\s*<td>([^<]*)</td>\s*<td>([^<]*)</td>\s*<td>([^<]*)</td>\s*<td>([^<]*)</td>\s*</tr>", body):
            rows[mk.strip()] = {"value": _num(val.strip()), "unit": unit.strip(), "desc": desc.strip()}
        if rows and title.strip() != "Chemical composition":
            rec["data"][title.strip()] = rows
    chem = {}
    m = re.search(r"Chemical composition \[%\]:</td>(.*?)</table>", page, re.S)
    if m:
        heads = [_text(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", m[1], re.S)]
        vals = [_text(v) for v in re.findall(r"<td(?: colspan=2)?>(.*?)</td>", m[1], re.S)]
        for h, v in zip(heads, vals):
            if h == "balance":
                if v and v != "---":
                    chem[h] = v
            elif _num(v) not in (0.0, None, ""):
                chem[h] = _num(v)
    if chem:
        rec["chemistry"] = chem
    return rec


def parse(cache: Path) -> list[dict]:
    index = json.loads((cache / "index.json").read_text())
    recs, seen = [], set()
    for ids in index.values():
        for rid in ids:
            if rid in seen:
                continue
            seen.add(rid)
            recs.append(parse_record((cache / f"{rid}.html").read_text(encoding="utf-8")))
    recs.sort(key=lambda r: (r["mark"].lower(), r["id"]))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="scratch/pragtic_raw")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    if not a.parse_only:
        download(cache)
    recs = parse(cache)
    OUT.parent.mkdir(exist_ok=True)
    payload = {"source": URL, "author": "Jan Papuga, PragTic – Fatigue Data Repository",
               "downloaded": time.strftime("%Y-%m-%d"), "records": recs}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(recs)} records → {OUT}")


if __name__ == "__main__":
    main()
