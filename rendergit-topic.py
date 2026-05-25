#!/usr/bin/env python3
"""
rendergit_topic_batch_v2.py

Enumerate ALL public repositories for a GitHub topic and run karpathy's `rendergit` (binary on PATH)
for each repo, extracting the LLM/CXML view.

Key improvements vs naive halving:
 - Samples repository density by month (then day/hour if necessary).
 - Groups months into partitions each with <= 1000 search results.
 - Caches counts per bucket for fast re-runs.
 - Rate-limit aware and resumable.
 - Strictly uses rendergit (must be on PATH).
"""

from __future__ import annotations
import argparse, json, logging, math, os, re, subprocess, sys, time, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import calendar
import requests

# ---------- Config ----------
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
MAX_PER_PAGE = 100
SEARCH_PAGE_CAP = 10            # 100 * 10 = 1000 cap
RENDERGIT_CMD = "rendergit"
DEFAULT_WORKERS = 3
DEFAULT_MAX_BYTES = 50 * 1024
RATE_LIMIT_SAFETY = 5           # seconds extra when sleeping for reset
CACHE_FILENAME = "counts_cache.json"
RETRY_BACKOFF_BASE = 3
RENDER_RETRY_LIMIT = 2
WORKER_DELAY = 0.08

# ---------- Helpers ----------
def which_rendergit() -> Optional[str]:
    from shutil import which
    return which(RENDERGIT_CMD)

def run_cmd(cmd: List[str], env: Optional[dict] = None, timeout: int = 900) -> Tuple[int, str, str]:
    env = env or os.environ.copy()
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"TimeoutExpired: {e}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name_from_full(full: str) -> str:
    fn = full.rstrip("/")
    if fn.startswith("http://") or fn.startswith("https://"):
        fn = fn.split("://", 1)[1]
    fn = fn.replace(":", "_").replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", fn)

# ---------- GitHub search wrapper (rate-limit aware) ----------
def _headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def _do_search(params: dict, token: Optional[str], retries: int = 3) -> requests.Response:
    headers = _headers(token)
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(GITHUB_SEARCH_URL, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            if attempt <= retries:
                wait = 2 ** attempt
                logging.warning("Search request exception: %s. retry in %ds", e, wait)
                time.sleep(wait)
                continue
            raise

        # handle rate-limit and retry logic
        remaining = int(r.headers.get("X-RateLimit-Remaining", "0") or 0)
        reset = int(r.headers.get("X-RateLimit-Reset", str(int(time.time()) + 60)))
        if r.status_code == 200:
            if remaining < 5:
                wait = max(0, reset - int(time.time())) + RATE_LIMIT_SAFETY
                logging.info("Approaching rate limit (remaining=%d). Sleeping %ds.", remaining, wait)
                time.sleep(wait)
            return r
        if r.status_code == 422:
            # Validation error (often happens when page > allowed or malformed query)
            return r
        if r.status_code in (403, 429):
            wait = max(0, reset - int(time.time())) + RATE_LIMIT_SAFETY
            logging.warning("Rate limited (HTTP %d). Sleeping %ds then retrying.", r.status_code, wait)
            time.sleep(wait)
            continue
        if attempt <= retries:
            wait = 2 ** attempt
            logging.warning("Search HTTP %d; retrying in %ds (attempt %d)", r.status_code, wait, attempt)
            time.sleep(wait)
            continue
        r.raise_for_status()

def total_count_for_range(topic: str, token: Optional[str], start_iso: str, end_iso: str) -> int:
    q = f"topic:{topic} created:{start_iso}..{end_iso}"
    params = {"q": q, "per_page": 1}
    r = _do_search(params, token)
    if r.status_code == 200:
        return int(r.json().get("total_count", 0))
    logging.warning("total_count query returned %d for %s..%s; treating as 0", r.status_code, start_iso, end_iso)
    return 0

def list_pages_for_range(topic: str, token: Optional[str], start_iso: str, end_iso: str) -> List[dict]:
    q = f"topic:{topic} created:{start_iso}..{end_iso}"
    params = {"q": q, "per_page": MAX_PER_PAGE, "page": 1}
    r = _do_search(params, token)
    if r.status_code == 200:
        js = r.json()
        total = int(js.get("total_count", 0))
        pages = min(SEARCH_PAGE_CAP, max(1, math.ceil(total / MAX_PER_PAGE)))
        items = js.get("items", [])
        for page in range(2, pages + 1):
            params["page"] = page
            r2 = _do_search(params, token)
            if r2.status_code == 200:
                items += r2.json().get("items", [])
            elif r2.status_code == 422:
                logging.warning("Hit 422 on page %d for range %s..%s", page, start_iso, end_iso)
                break
            else:
                r2.raise_for_status()
        return items
    else:
        if r.status_code == 422:
            raise RuntimeError("Validation error listing pages; range may be too large.")
        r.raise_for_status()

# ---------- time-bucket helpers ----------
def month_ranges(start: date, end: date) -> List[Tuple[date, date]]:
    """Return list of (month_start_date, month_end_date) covering start..end inclusive."""
    ranges = []
    y, m = start.year, start.month
    cur = date(y, m, 1)
    while cur <= end:
        y2, m2 = cur.year, cur.month
        last_day = calendar.monthrange(y2, m2)[1]
        month_end = date(y2, m2, last_day)
        if month_end > end:
            month_end = end
        ranges.append((cur, month_end))
        # next month
        if m2 == 12:
            cur = date(y2 + 1, 1, 1)
        else:
            cur = date(y2, m2 + 1, 1)
    return ranges

def day_ranges(start: date, end: date) -> List[Tuple[date, date]]:
    cur = start
    out = []
    while cur <= end:
        out.append((cur, cur))
        cur = cur + timedelta(days=1)
    return out

# ---------- sampling + partitioning ----------
def load_cache(cache_path: Path) -> Dict[str,int]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf8"))
        except Exception:
            return {}
    return {}

def save_cache(cache_path: Path, cache: Dict[str,int]):
    try:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf8")
    except Exception:
        logging.debug("Could not write cache to %s", cache_path)

def iso_date(d: date) -> str:
    return d.isoformat()

def sample_counts_by_month(topic: str, token: Optional[str], start: date, end: date, cache: Dict[str,int], cache_path: Path) -> List[Tuple[date,date,int]]:
    months = month_ranges(start, end)
    results = []
    for s,e in months:
        key = f"month:{s.isoformat()}..{e.isoformat()}"
        if key in cache:
            count = cache[key]
            logging.debug("Cache hit %s -> %d", key, count)
        else:
            count = total_count_for_range(topic, token, iso_date(s), iso_date(e))
            cache[key] = count
            save_cache(cache_path, cache)
        results.append((s,e,count))
    return results

def sample_counts_by_day(topic: str, token: Optional[str], start: date, end: date, cache: Dict[str,int], cache_path: Path) -> List[Tuple[date,date,int]]:
    days = day_ranges(start, end)
    results = []
    for s,e in days:
        key = f"day:{s.isoformat()}"
        if key in cache:
            count = cache[key]
        else:
            count = total_count_for_range(topic, token, iso_date(s), iso_date(e))
            cache[key] = count
            save_cache(cache_path, cache)
        results.append((s,e,count))
    return results

def partition_buckets_by_target(buckets: List[Tuple[date,date,int]], target: int = 1000) -> List[Tuple[date,date,int]]:
    """
    Group consecutive buckets into partitions where sum(count) <= target.
    Returns list of partitions described as (partition_start, partition_end, partition_total_count).
    """
    partitions = []
    cur_start=None
    cur_end=None
    cur_sum=0
    for (s,e,c) in buckets:
        if c > target:
            # single bucket too big: we must emit it as its own partition (caller should split further)
            if cur_sum>0:
                partitions.append((cur_start, cur_end, cur_sum))
                cur_start=None; cur_end=None; cur_sum=0
            partitions.append((s,e,c))
            continue
        if cur_sum + c > target:
            # finalize current partition
            partitions.append((cur_start, cur_end, cur_sum))
            cur_start,cur_end,cur_sum = s,e,c
        else:
            if cur_sum==0:
                cur_start,cur_end,cur_sum = s,e,c
            else:
                # extend
                cur_end = e
                cur_sum += c
    if cur_sum>0:
        partitions.append((cur_start,cur_end,cur_sum))
    return partitions

def build_partitions(topic: str, token: Optional[str], start: date, end: date, out_dir: Path) -> List[Tuple[date,date]]:
    """
    Return list of (start_date, end_date) partitions such that each partition is likely <=1000 results.
    Strategy:
      - sample monthly counts
      - group months until partition sum <= 1000
      - if a month > 1000, sample daily inside that month and group days
      - if a day > 1000, attempt hour splitting (rare)
    """
    cache_path = out_dir / CACHE_FILENAME
    cache = load_cache(cache_path)
    months = sample_counts_by_month(topic, token, start, end, cache, cache_path)
    logging.info("Sampled %d months", len(months))

    partitions = []
    month_partitions = partition_buckets_by_target(months, 1000)
    for (ms, me, mcount) in month_partitions:
        if mcount <= 1000:
            partitions.append((ms, me))
            continue
        # month chunk exceeded 1000 -> split by day
        logging.info("Month range %s..%s has %d (>1000): splitting days", ms.isoformat(), me.isoformat(), mcount)
        days = sample_counts_by_day(topic, token, ms, me, cache, cache_path)
        day_partitions = partition_buckets_by_target(days, 1000)
        for (ds,de,dcount) in day_partitions:
            if dcount <= 1000:
                partitions.append((ds,de))
                continue
            # day still >1000 -> split hours (rare). We'll binary-split the day into halves iteratively.
            logging.info("Day %s has %d (>1000): splitting by binary halves", ds.isoformat(), dcount)
            queue = [(ds,de)]
            while queue:
                a,b = queue.pop(0)
                # compute midpoint datetime (date -> midday)
                a_dt = datetime.combine(a, datetime.min.time(), tzinfo=timezone.utc)
                b_dt = datetime.combine(b, datetime.max.time(), tzinfo=timezone.utc)
                mid_dt = a_dt + (b_dt - a_dt) / 2
                # format iso timestamps (without microseconds) - GitHub accepts ISO 8601 timestamps
                a_iso = a_dt.isoformat().replace("+00:00","Z")
                mid_iso = mid_dt.isoformat().replace("+00:00","Z")
                b_iso = b_dt.isoformat().replace("+00:00","Z")
                # note: total_count_for_range supports timestamps too (best-effort)
                left_count = total_count_for_range(topic, token, a_iso, mid_iso)
                right_count = total_count_for_range(topic, token, mid_iso, b_iso)
                if left_count <= 1000:
                    partitions.append((a, a))
                else:
                    # if still large and range is less than an hour, accept partial (can't split further reliably)
                    # But attempt further splitting by halving datetime range
                    queue.append((a, a))  # fallback - best-effort
                if right_count <= 1000:
                    partitions.append((b, b))
                else:
                    queue.append((b, b))
            # end day splitting
    # deduplicate and merge adjacent partitions (safety)
    merged = []
    for s,e in partitions:
        if not merged:
            merged.append((s,e))
        else:
            last_s,last_e = merged[-1]
            if (last_e + timedelta(days=1)) >= s:
                # merge contiguous
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s,e))
    logging.info("Built %d partitions", len(merged))
    return merged

# ---------- enumerate repositories per partition ----------
def enumerate_repos_for_partitions(topic: str, token: Optional[str], partitions: List[Tuple[date,date]]) -> List[str]:
    all_repos = []
    seen=set()
    for (s,e) in partitions:
        logging.info("Fetching partition %s..%s", s.isoformat(), e.isoformat())
        items = list_pages_for_range(topic, token, s.isoformat(), e.isoformat())
        for it in items:
            fn = it.get("full_name")
            if fn and fn not in seen:
                seen.add(fn)
                all_repos.append(fn)
    return all_repos

# ---------- rendergit run + extraction ----------
def extract_llm_view(html_path: Path) -> Optional[str]:
    raw = html_path.read_text(encoding="utf8", errors="replace")
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        textareas = soup.find_all("textarea")
        if textareas:
            largest = max(textareas, key=lambda t: len(t.get_text() or ""))
            return largest.get_text()
        pres = soup.find_all(["pre","code"])
        if pres:
            largest = max(pres, key=lambda t: len(t.get_text() or ""))
            return largest.get_text()
        for attr in ("id","class","data-view","aria-label"):
            els = soup.find_all(attrs={attr: re.compile(r"llm|cxml|LLM|CXML", re.I)})
            if els:
                cand = max(els, key=lambda e: len(e.get_text() or ""))
                return cand.get_text()
        if soup.body:
            return soup.body.get_text(separator="\n")
        return raw
    except Exception:
        m = re.search(r"<textarea[^>]*>([\s\S]*?)</textarea>", raw)
        if m:
            return re.sub(r"&#x([0-9A-Fa-f]+);", lambda mo: chr(int(mo.group(1),16)), m.group(1))
        m = re.search(r"<pre[^>]*>([\s\S]*?)</pre>", raw)
        if m:
            return re.sub(r"<[^>]+>","",m.group(1))
        m = re.search(r"<body[^>]*>([\s\S]*)</body>", raw)
        if m:
            return re.sub(r"<[^>]+>","",m.group(1))
        return raw

def run_rendergit_for_repo(full_name: str, out_root: Path, max_bytes: Optional[int], no_open: bool, tmpbase: Optional[str], retry_limit: int) -> Dict[str,Any]:
    safe = safe_name_from_full(full_name)
    repo_out = out_root / safe
    ensure_dir(repo_out)
    repo_url = f"https://github.com/{full_name}" if not (full_name.startswith("http://") or full_name.startswith("https://")) else full_name
    out_html = str((repo_out / "rendergit.html").resolve())
    cmd = [RENDERGIT_CMD]
    if no_open: cmd.append("--no-open")
    if max_bytes and max_bytes>0:
        cmd += ["--max-bytes", str(max_bytes)]
    cmd += ["-o", out_html, repo_url]

    env = os.environ.copy()
    if tmpbase:
        tmpd = Path(tmpbase) / f"rg_tmp_{safe}_{int(time.time())}"
        tmpd.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmpd)

    attempt = 0
    last_out = last_err = ""
    rc = None
    while attempt <= retry_limit:
        attempt += 1
        logging.info("rendergit %s (attempt %d)", full_name, attempt)
        rc, out, err = run_cmd(cmd, env=env, timeout=900)
        last_out, last_err = out, err
        if rc == 0 and Path(out_html).exists():
            logging.info("rendergit done for %s", full_name)
            break
        wait = RETRY_BACKOFF_BASE * attempt
        logging.warning("rendergit failed for %s rc=%s; sleeping %ds", full_name, rc, wait)
        time.sleep(wait)

    manifest = {
        "repo": full_name,
        "repo_url": repo_url,
        "cmd": cmd,
        "attempts": attempt,
        "exit_code": rc,
        "stdout": (last_out or "")[:200000],
        "stderr": (last_err or "")[:200000],
        "timestamp": int(time.time())
    }

    p = Path(out_html)
    if p.exists():
        try:
            llm = extract_llm_view(p)
            if not llm:
                manifest["extract_ok"] = False
                manifest["error"] = "empty_llm"
            else:
                (repo_out / "export.cxml.txt").write_text(llm, encoding="utf8")
                (repo_out / "export.md").write_text("# Rendered by rendergit\n\n```\n" + llm[:20000] + "\n```\n", encoding="utf8")
                # copy html into repo_out if outside
                try:
                    if p.resolve().parent != repo_out.resolve():
                        dest = repo_out / p.name
                        if not dest.exists(): dest.write_bytes(p.read_bytes())
                except Exception:
                    pass
                manifest["extract_ok"] = True
        except Exception as e:
            manifest["extract_ok"] = False
            manifest["error"] = f"extract_exc:{e}"
    else:
        manifest["extract_ok"] = False
        manifest["error"] = "html_not_found"

    try:
        (repo_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    except Exception:
        logging.exception("Failed to write manifest for %s", full_name)
    return manifest

# ---------- main orchestration ----------
def main():
    ap = argparse.ArgumentParser(description="Partition GitHub topic by time buckets and run rendergit for each repo.")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--out", default="exports")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--start-date", default="2008-01-01")
    ap.add_argument("--end-date", default=date.today().isoformat())
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tmpbase", help="base TMPDIR for rendergit runs")
    ap.add_argument("--token", help="GITHUB_TOKEN (or set env var)")
    ap.add_argument("--workers-delay", type=float, default=WORKER_DELAY)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    token = args.token or os.environ.get("GITHUB_TOKEN")
    rg = which_rendergit()
    if not rg:
        logging.error("rendergit not found on PATH (expected 'rendergit'). Install it and retry.")
        sys.exit(1)
    logging.info("Using rendergit at: %s", rg)

    out_root = Path(args.out)
    ensure_dir(out_root)

    repo_list_path = out_root / "repo_list.json"
    if repo_list_path.exists():
        logging.info("Loading existing repo_list.json (resume mode)")
        repo_list = json.loads(repo_list_path.read_text(encoding="utf8"))
    else:
        # build balanced partitions
        start = datetime.fromisoformat(args.start_date).date()
        end = datetime.fromisoformat(args.end_date).date()
        partitions = build_partitions(args.topic, token, start, end, out_root)
        logging.info("Enumerating repositories across %d partitions...", len(partitions))
        repo_list = enumerate_repos_for_partitions(args.topic, token, partitions)
        repo_list_path.write_text(json.dumps(repo_list, indent=2), encoding="utf8")
        logging.info("Enumerated %d unique repos and saved to %s", len(repo_list), repo_list_path)

    # apply resume filtering
    if args.resume:
        filtered=[]
        for r in repo_list:
            safe = safe_name_from_full(r)
            if (out_root / safe / "manifest.json").exists():
                logging.info("Skipping already-processed: %s", r)
                continue
            filtered.append(r)
        repo_list = filtered
        logging.info("After resume filter, %d repos remain", len(repo_list))

    # run rendergit concurrently
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures={}
        for fullname in repo_list:
            fut = exe.submit(run_rendergit_for_repo, fullname, out_root, (args.max_bytes if args.max_bytes>0 else None), True, args.tmpbase, RENDER_RETRY_LIMIT)
            futures[fut]=fullname
            time.sleep(args.workers_delay)
        for fut in as_completed(futures):
            fullname=futures[fut]
            try:
                m = fut.result()
                results.append(m)
                logging.info("Processed %s -> %s", fullname, "OK" if m.get("extract_ok") else "FAIL")
            except Exception as e:
                logging.exception("Unhandled error for %s: %s", fullname, e)
                results.append({"repo": fullname, "error": str(e)})

    succ=sum(1 for r in results if r.get("extract_ok"))
    fail=len(results)-succ
    summary={"topic": args.topic, "total": len(results), "succeeded": succ, "failed": fail, "timestamp": int(time.time())}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    logging.info("Done. exports in %s: %d success, %d fail", out_root, succ, fail)

if __name__ == "__main__":
    main()
