#!/usr/bin/env python3
"""
Ingest a company's PDFs end-to-end: extract -> chunk -> embed -> index.

Usage:
    python scripts/ingest_company_pdfs.py --company company_a

Local-first: a PDF is skipped if its SHA-256 matches the manifest AND its
clean/{sha256}.md file still exists. Kill this mid-run and re-run the same
command to resume — completed files aren't reprocessed.

PDF text extraction is CPU-bound, so it runs in a process pool
(Settings.max_pdf_workers workers) rather than a thread pool, sidestepping
the GIL for the actual parsing work.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.checkpoint import Checkpoint
from rag_platform.chunker import RecursiveCharacterChunker
from rag_platform.company_registry import CompanyNotFoundError, CompanyRegistry
from rag_platform.config import get_settings
from rag_platform.config_loader import ConfigValidationError, load_company_config
from rag_platform.embedder import Embedder
from rag_platform.logging_config import bind_company, configure_logging, get_logger
from rag_platform.manifest import ManifestEntry, ManifestStatus, ManifestStore, sha256_file
from rag_platform.pdf_extractor import PDFExtractionError, extract_pdf
from rag_platform.vector_store import VectorStore

log = get_logger("ingest_company_pdfs")


def find_pdfs(company_config) -> list[Path]:
    pdfs: list[Path] = []
    for source in company_config.sources:
        if source.type != "pdf":
            continue
        source_dir = Path(source.path)
        if not source_dir.exists():
            log.warning("source_path_missing", path=str(source_dir))
            continue
        pdfs.extend(sorted(source_dir.glob(source.glob)))
    return pdfs


async def process_one_pdf(
    pdf_path: Path,
    company_id: str,
    settings,
    company_config,
    manifest: ManifestStore,
    chunker: RecursiveCharacterChunker,
    embedder: Embedder,
    vector_store: VectorStore,
    executor: ProcessPoolExecutor,
) -> dict:
    """Returns {"status": "ok"|"scanned"|"error", "chunk_count": int, "error": str|None}."""
    loop = asyncio.get_running_loop()
    file_hash = sha256_file(pdf_path)

    try:
        doc = await loop.run_in_executor(executor, extract_pdf, pdf_path)
    except PDFExtractionError as exc:
        log.error("pdf_extraction_failed", file=str(pdf_path), error=str(exc))
        await manifest.add(
            ManifestEntry(
                source_path=str(pdf_path),
                file_sha256=file_hash,
                company_id=company_id,
                status=ManifestStatus.ERROR,
                error=str(exc),
            )
        )
        return {"status": "error", "chunk_count": 0, "error": str(exc)}

    if doc.is_scanned:
        log.warning("scanned_pdf_skipped", file=str(pdf_path), page_count=doc.page_count)
        await manifest.add(
            ManifestEntry(
                source_path=str(pdf_path),
                file_sha256=file_hash,
                company_id=company_id,
                status=ManifestStatus.SKIPPED_SCANNED,
                page_count=doc.page_count,
                extractor_used=doc.extractor_used,
            )
        )
        return {"status": "scanned", "chunk_count": 0, "error": None}

    clean_path = settings.clean_dir / f"{file_hash}.md"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = clean_path.with_suffix(".md.tmp")
    tmp_path.write_text(doc.markdown, encoding="utf-8")
    tmp_path.replace(clean_path)  # atomic write

    # NOTE (C1 fix): index the new chunks FIRST, then remove stale ones.
    # The old code deleted by source *before* embedding/upsert — if either
    # threw, the file's chunks were gone from the index while the manifest
    # still said OK, so the PDF was skipped forever with no signal.
    chunks = chunker.split(doc.markdown, metadata={"source_path": str(pdf_path), "company_id": company_id})
    if chunks:
        texts = [c.text for c in chunks]
        vectors = await loop.run_in_executor(None, embedder.embed_documents, texts)
        ids = [f"{file_hash}:{c.index}" for c in chunks]
        metadatas = [
            {
                "source_path": str(pdf_path),
                "company_id": company_id,
                "page_number": c.page_number if c.page_number is not None else -1,
                "chunk_index": c.index,
                "is_code": c.is_code,
                "file_sha256": file_hash,
            }
            for c in chunks
        ]
        vector_store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

        # Now remove chunks from previous versions of this same file
        # (same source_path, different file hash) — stale data can't linger.
        vector_store.delete_stale_versions(str(pdf_path), file_hash)

    await manifest.add(
        ManifestEntry(
            source_path=str(pdf_path),
            file_sha256=file_hash,
            clean_path=str(clean_path),
            company_id=company_id,
            status=ManifestStatus.OK,
            page_count=doc.page_count,
            chunk_count=len(chunks),
            extractor_used=doc.extractor_used,
        )
    )

    log.info("pdf_ingested", file=str(pdf_path), pages=doc.page_count, chunks=len(chunks))
    return {"status": "ok", "chunk_count": len(chunks), "error": None}


async def main(company_id: str) -> int:
    configure_logging()
    bind_company(company_id)

    settings = get_settings(company_id)
    settings.ensure_dirs()

    try:
        company_config = load_company_config(company_id, settings.configs_dir)
    except ConfigValidationError as exc:
        log.error("config_load_failed", error=str(exc))
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    registry = CompanyRegistry(settings.registry_path)
    if not registry.exists(company_id):
        log.warning("company_not_registered", hint="run scripts/manage_companies.py add first")

    pdfs = find_pdfs(company_config)
    if not pdfs:
        log.info("no_pdfs_found")
        print("No PDFs found for this company. Check your config's sources[].path.")
        return 0

    manifest = ManifestStore(settings.manifest_path)
    manifest.load()

    to_process = [p for p in pdfs if manifest.needs_processing(p)]
    skipped_unchanged = len(pdfs) - len(to_process)
    log.info("ingestion_plan", total_pdfs=len(pdfs), to_process=len(to_process), already_up_to_date=skipped_unchanged)

    checkpoint = Checkpoint.load(settings.checkpoint_path, company_id=company_id)
    checkpoint.seed([str(p) for p in to_process])
    checkpoint.save(settings.checkpoint_path)

    chunker = RecursiveCharacterChunker(
        chunk_size=company_config.processing.chunk_size,
        chunk_overlap=company_config.processing.chunk_overlap,
        min_chunk_length=company_config.processing.min_chunk_length,
    )
    embedder = Embedder(
        model_name=company_config.embedding.model,
        batch_size=company_config.embedding.batch_size,
        device=company_config.embedding.device,
    )
    vector_store = VectorStore(persist_dir=settings.vector_db_dir, collection_name=company_config.vector_store.collection_name)

    started_at = time.monotonic()
    ok_count = scanned_count = error_count = total_chunks = 0

    with ProcessPoolExecutor(max_workers=settings.max_pdf_workers) as executor:
        for pdf_path in [Path(p) for p in checkpoint.pending]:
            try:
                result = await process_one_pdf(
                    pdf_path, company_id, settings, company_config, manifest, chunker, embedder, vector_store, executor
                )
                if result["status"] == "ok":
                    ok_count += 1
                    total_chunks += result["chunk_count"]
                    checkpoint.mark_completed(str(pdf_path))
                elif result["status"] == "scanned":
                    scanned_count += 1
                    checkpoint.mark_skipped_scanned(str(pdf_path))
                else:
                    error_count += 1
                    checkpoint.mark_failed(str(pdf_path))
            except Exception as exc:  # noqa: BLE001 — one bad PDF must never kill the whole run
                log.error("unexpected_ingest_error", file=str(pdf_path), error=str(exc))
                error_count += 1
                checkpoint.mark_failed(str(pdf_path))
            finally:
                checkpoint.save(settings.checkpoint_path)

    elapsed = time.monotonic() - started_at

    if registry.exists(company_id):
        registry.update_stats(
            company_id,
            total_pdfs=len(manifest.all_entries()),
            total_chunks=sum(e.chunk_count or 0 for e in manifest.all_entries()),
            mark_ingestion_now=True,
        )

    log.info(
        "ingestion_run_complete",
        ok=ok_count,
        scanned_skipped=scanned_count,
        errors=error_count,
        already_up_to_date=skipped_unchanged,
        total_chunks_this_run=total_chunks,
        elapsed_seconds=round(elapsed, 1),
    )

    print(f"\n=== Ingestion summary for {company_id} ===")
    print(f"  PDFs processed:      {ok_count}")
    print(f"  Skipped (scanned):   {scanned_count}")
    print(f"  Errors:              {error_count}")
    print(f"  Already up to date:  {skipped_unchanged}")
    print(f"  Chunks indexed:      {total_chunks}")
    print(f"  Time:                {elapsed:.1f}s")

    return 1 if error_count and ok_count == 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True, help="Company id, matching configs/{company_id}.yaml")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.company)))
