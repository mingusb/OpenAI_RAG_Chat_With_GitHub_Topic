#!/usr/bin/env python3
"""
openai_file_search_upload.py

Robust uploader: find all rendergit.html files and upload them into an OpenAI
Vector Store (File Search). Works with multiple client SDK shapes by probing
for the methods available at runtime and using safe fallbacks.

Requirements:
    pip install --upgrade openai
    export OPENAI_API_KEY="sk-..."

Usage:
    python openai_file_search_upload.py \
        --root . \
        --store-name rendergit-html-corpus \
        --max-upload-mb 450 \
        --chunk-mb 200 \
        --test-query "reverse engineering"

Notes:
- max-upload-mb: maximum single-file upload size to accept as-is. Files bigger than
  that will be split into chunk files of size chunk-mb and uploaded instead.
- chunk-mb: chunk size used when splitting oversized files.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from openai import OpenAI
except Exception as e:
    print("ERROR: openai package not found. Install with: pip install --upgrade openai")
    raise

LOG = logging.getLogger("openai_file_search_upload")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0:
            return f"{nbytes:0.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:0.2f} PB"


def find_rendergit_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("rendergit.html") if p.is_file()]


def create_or_get_vector_store(client: OpenAI, name: str) -> str:
    """
    Creates a vector store named `name` if it doesn't exist; otherwise returns an existing id.
    Uses client.vector_stores.list/create where possible.
    """
    vs_client = getattr(client, "vector_stores", None)
    if vs_client is None:
        raise RuntimeError("This OpenAI client does not expose `vector_stores`. "
                           "Upgrade the 'openai' package or consult the SDK docs.")

    # Try to find existing store with same name
    try:
        if hasattr(vs_client, "list"):
            LOG.debug("Listing vector stores to detect existing store name.")
            lst = vs_client.list()
            # lst may be a response object or a list; try to detect
            candidates = []
            if isinstance(lst, dict) and "data" in lst:
                candidates = lst["data"]
            elif hasattr(lst, "__iter__"):
                candidates = list(lst)
            for item in candidates:
                # item might be object with .id/.name or dict
                iname = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
                iid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
                if iname == name:
                    LOG.info("Found existing vector store with name=%s id=%s", name, iid)
                    return iid
    except Exception:
        LOG.debug("Could not list vector stores (SDK differences); continuing to create.", exc_info=True)

    # Create
    try:
        LOG.info("Creating vector store named '%s'...", name)
        created = vs_client.create(name=name)
        # created may be object or dict
        vs_id = getattr(created, "id", None) or (created.get("id") if isinstance(created, dict) else None)
        if not vs_id:
            # maybe API returns full object with nested id or simply returns string
            vs_id = str(created)
        LOG.info("Vector store created: id=%s", vs_id)
        return vs_id
    except Exception as e:
        LOG.exception("Failed to create vector store: %s", e)
        raise


def split_file_to_chunks(path: Path, chunk_bytes: int, tmpdir: Path) -> List[Path]:
    """
    Split a file into chunk files under tmpdir and return the list of chunk file paths.
    This is a binary-safe split; chunk boundaries are by bytes (not semantically optimal),
    but safe for upload in cases where service enforces per-file size limits.
    """
    LOG.info("Splitting %s into %d-byte chunks under %s", path, chunk_bytes, tmpdir)
    parts: List[Path] = []
    base_name = path.name
    idx = 0
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk_bytes)
            if not data:
                break
            idx += 1
            outp = tmpdir / f"{base_name}.part{idx:04d}"
            with outp.open("wb") as outfh:
                outfh.write(data)
            parts.append(outp)
    LOG.info("Created %d chunk files for %s", len(parts), path)
    return parts


def upload_files_to_vector_store(client: OpenAI, vs_id: str, file_paths: List[Path]):
    """
    Upload files to a vector store using vector_stores.file_batches.upload_and_poll if available,
    otherwise attempt a create + basic poll fallback.
    """
    vs_client = client.vector_stores
    fb = getattr(vs_client, "file_batches", None)
    # Convert to file objects for upload; keep track to close later.
    file_objs = []
    try:
        for p in file_paths:
            file_objs.append(open(str(p), "rb"))

        if fb and hasattr(fb, "upload_and_poll"):
            LOG.info("Using file_batches.upload_and_poll() to upload %d files...", len(file_objs))
            # Some SDKs provide upload_and_poll as a context manager (as in earlier examples).
            try:
                # Try as context manager first
                with fb.upload_and_poll(vector_store_id=vs_id, files=file_objs) as batch:
                    LOG.info("Batch status: %s", getattr(batch, "status", str(batch)))
                    LOG.info("Batch file counts: %s", getattr(batch, "file_counts", str(batch)))
                    return batch
            except TypeError:
                # Maybe not a context manager - call directly
                batch = fb.upload_and_poll(vector_store_id=vs_id, files=file_objs)
                LOG.info("upload_and_poll returned: %s", batch)
                return batch
        else:
            # Fallback: try fb.create or vs_client.files.create
            LOG.info("file_batches.upload_and_poll not available; falling back to create + poll.")
            if fb and hasattr(fb, "create"):
                LOG.info("Using file_batches.create(...)")
                batch = fb.create(vector_store_id=vs_id, files=file_objs)
                LOG.info("Created batch: %s. Attempting to poll status...", batch)
                # Attempt to poll for readiness - best-effort
                batch_id = getattr(batch, "id", None) or (batch.get("id") if isinstance(batch, dict) else None)
                if batch_id and hasattr(fb, "retrieve"):
                    for _ in range(60):
                        time.sleep(2)
                        status = fb.retrieve(batch_id)
                        st = getattr(status, "status", None) or (status.get("status") if isinstance(status, dict) else None)
                        LOG.info("Batch %s status: %s", batch_id, st)
                        if st in ("succeeded", "ready", "completed"):
                            return status
                    LOG.warning("Batch did not reach ready status in the allotted polling time.")
                    return status
                return batch
            else:
                # Last resort: try using vs_client.files.create (if exists)
                if hasattr(vs_client, "files"):
                    LOG.info("Using vector_stores.files.create for each file as a last-resort fallback.")
                    created_files = []
                    for f in file_objs:
                        # Some SDK shapes expect 'file' param; others may differ
                        try:
                            resp = vs_client.files.create(file=f, vector_store_id=vs_id)
                        except TypeError:
                            resp = vs_client.files.create(file=f)
                        created_files.append(resp)
                        LOG.info("Created file: %s", getattr(resp, "id", str(resp)))
                    return created_files
                else:
                    raise RuntimeError("No supported file upload API found on vector_stores.")
    finally:
        # Close file objects
        for fo in file_objs:
            try:
                fo.close()
            except Exception:
                pass


def try_search_sample(client: OpenAI, vs_id: str, query: str, top_k: int = 3):
    """
    Try a quick search to validate that the vector store is usable.
    We probe for multiple method names (search, retrieve, query) to be robust.
    """
    vs_client = client.vector_stores
    search_methods = ["search", "retrieve", "query"]
    last_exc = None
    for m in search_methods:
        if hasattr(vs_client, m):
            LOG.info("Attempting vector_stores.%s(...)", m)
            func = getattr(vs_client, m)
            try:
                # Try a few common parameter patterns used in different SDK versions
                for param_set in (
                    {"vector_store_id": vs_id, "query": query, "top_k": top_k},
                    {"id": vs_id, "query": query, "top_k": top_k},
                    {"vector_store_id": vs_id, "query": query},
                    {"id": vs_id, "query": query},
                ):
                    try:
                        resp = func(**{k: v for k, v in param_set.items() if v is not None})
                    except TypeError:
                        continue
                    # Print a compact summary of results
                    LOG.info("Search method '%s' returned: %s", m, type(resp))
                    try:
                        items = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else resp)
                        if isinstance(items, list):
                            LOG.info("Top %d hits:", min(len(items), top_k))
                            for i, it in enumerate(items[:top_k], start=1):
                                # it may have .text, .file_id, .score or be a dict/payload
                                text = getattr(it, "text", None) or (it.get("text") if isinstance(it, dict) else None)
                                score = getattr(it, "score", None) or (it.get("score") if isinstance(it, dict) else None)
                                fid = getattr(it, "file_id", None) or (it.get("file_id") if isinstance(it, dict) else None)
                                LOG.info("  %d) file_id=%s score=%s text_preview=%s", i, fid, score, (text or "")[:200].replace("\n", " "))
                        else:
                            LOG.info("Search response shape unexpected: %s", resp)
                    except Exception:
                        LOG.info("Could not introspect search response.")
                    return resp
            except Exception as e:
                LOG.debug("Search method %s failed: %s", m, e, exc_info=True)
                last_exc = e
    raise RuntimeError("All search method attempts failed.") from last_exc


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Upload rendergit.html files into OpenAI File Search (Vector Store).")
    p.add_argument("--root", default=".", help="Root directory to search for rendergit.html")
    p.add_argument("--store-name", default="rendergit-html-corpus", help="Name for the vector store")
    p.add_argument("--max-upload-mb", type=int, default=500, help="Maximum size (MB) allowed for a single file upload; oversized files will be split")
    p.add_argument("--chunk-mb", type=int, default=200, help="Chunk size (MB) used when splitting oversized files")
    p.add_argument("--test-query", default="reverse engineering", help="A small test query to validate the store after upload")
    p.add_argument("--dry-run", action="store_true", help="Do not perform uploads; just list files and estimated sizes")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv or sys.argv[1:])

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    root = Path(args.root).resolve()
    if not root.exists():
        LOG.error("Root path does not exist: %s", root)
        sys.exit(2)

    client = OpenAI()

    # Find files
    files = find_rendergit_files(root)
    LOG.info("Found %d rendergit.html files under %s", len(files), root)
    if not files:
        LOG.info("Nothing to upload. Exiting.")
        return

    total_bytes = sum(p.stat().st_size for p in files)
    LOG.info("Total bytes to consider: %s", human(total_bytes))

    if args.dry_run:
        for pth in files:
            LOG.info("Would upload: %s (%s)", pth, human(pth.stat().st_size))
        return

    # Create/reuse vector store
    vs_id = create_or_get_vector_store(client, args.store_name)

    # Prepare a temporary dir for chunk files
    tmpdir = Path(tempfile.mkdtemp(prefix="openai-file-upload-"))
    created_temp_parts: List[Path] = []
    try:
        upload_candidates: List[Path] = []
        max_upload_bytes = int(args.max_upload_mb * 1024 * 1024)
        chunk_bytes = int(args.chunk_mb * 1024 * 1024)

        for p in files:
            size = p.stat().st_size
            if size <= max_upload_bytes:
                upload_candidates.append(p)
            else:
                LOG.warning("File %s is %s, exceeds max-upload %s; splitting into chunks of %s", p, human(size), human(max_upload_bytes), human(chunk_bytes))
                parts = split_file_to_chunks(p, chunk_bytes, tmpdir)
                created_temp_parts.extend(parts)
                upload_candidates.extend(parts)

        LOG.info("Total upload candidate files (after chunking): %d", len(upload_candidates))
        for c in upload_candidates:
            LOG.debug("Candidate: %s (%s)", c, human(c.stat().st_size))

        # Upload in batches: to avoid memory pressure, upload in chunks of N files
        BATCH_FILES = 32
        for i in range(0, len(upload_candidates), BATCH_FILES):
            batch = upload_candidates[i:i + BATCH_FILES]
            LOG.info("Uploading batch %d..%d (count=%d)...", i + 1, i + len(batch), len(batch))
            result = upload_files_to_vector_store(client, vs_id, batch)
            LOG.info("Upload batch result: %s", getattr(result, "status", str(result)))

        # Final verification search
        LOG.info("Running verification search for query: %s", args.test_query)
        try:
            resp = try_search_sample(client, vs_id, args.test_query, top_k=3)
            LOG.info("Verification search succeeded.")
        except Exception as e:
            LOG.error("Verification search failed: %s", e)
            raise

        LOG.info("Upload process complete. Vector store id: %s", vs_id)
        LOG.info("You can now attach this vector store to an Assistant or query programmatically.")
    finally:
        # cleanup temporary parts
        if created_temp_parts:
            LOG.debug("Cleaning up %d temporary part files ...", len(created_temp_parts))
            for fp in created_temp_parts:
                try:
                    fp.unlink()
                except Exception:
                    pass
        # remove tmpdir
        try:
            tmpdir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
