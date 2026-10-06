"""Standard vector search baseline (not Vestige).

One document per ingested commit. The text layout below is fixed:
subject, a blank line, the body, a blank line, then changed paths.
Nothing here is tuned. Ranking is plain cosine, highest first.
"""

import os
import subprocess
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer

# Frozen before any search result was read. Do not edit commit_document.
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 9


def stop(text):
    print("stopped: %s" % text, file=sys.stderr)
    sys.exit(1)


def commit_document(subject, body, paths):
    subject = subject.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    body = body.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    cleaned = []
    for path in paths:
        path = path.replace("\r", "").strip()
        if path:
            cleaned.append(path)
    return subject + "\n\n" + body + "\n\n" + "\n".join(cleaned)


def git_text(repo, args):
    run = subprocess.run(
        ["git", "-C", repo] + args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if run.returncode != 0:
        stop("git failed while reading a commit")
    return run.stdout.decode("utf-8", "replace")


def load_lines(path):
    try:
        raw = open(path, "r", encoding="utf-8").read()
    except OSError:
        stop("missing %s" % path)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def cosine_scores(query_vec, doc_vecs):
    query_vec = np.asarray(query_vec, dtype=np.float64)
    doc_vecs = np.asarray(doc_vecs, dtype=np.float64)
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    if query_norm == 0 or np.any(doc_norms == 0):
        stop("a vector had length zero")
    query_vec = query_vec / query_norm
    doc_vecs = doc_vecs / doc_norms[:, None]
    return doc_vecs @ query_vec


def main():
    if len(sys.argv) != 7:
        stop("usage: search.py REPO SHAS QUERY_FILE CAUSE_FILE COUNT_FILE WALK_FILE")
    repo, shas_path, query_path, cause_path, count_path, walk_path = sys.argv[1:]
    shas = load_lines(shas_path)
    cause_lines = load_lines(cause_path)
    count_lines = load_lines(count_path)
    if len(cause_lines) != 1 or len(count_lines) != 1:
        stop("cause or count file is malformed")
    cause = cause_lines[0]
    try:
        expected = int(count_lines[0])
    except ValueError:
        stop("count file is malformed")
    if len(shas) != expected:
        stop("corpus size does not match the ingested commit count")
    try:
        query = open(query_path, "r", encoding="utf-8").read()
    except OSError:
        stop("missing the failure text")
    if not query:
        stop("the failure text is empty")

    documents = []
    for sha in shas:
        if len(sha) != 40:
            stop("a commit id is malformed")
        subject = git_text(repo, ["log", "-1", "--format=%s", sha])
        body = git_text(repo, ["log", "-1", "--format=%b", sha])
        paths = git_text(
            repo, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha]
        ).splitlines()
        document = commit_document(subject, body, paths)
        if not document.startswith(subject.rstrip("\n")) or "\n\n" not in document:
            stop("document format broke")
        documents.append(document)

    model = SentenceTransformer(MODEL_NAME)
    doc_vecs = model.encode(documents, show_progress_bar=False, convert_to_numpy=True)
    query_vec = model.encode(
        [query], show_progress_bar=False, convert_to_numpy=True
    )[0]
    scores = cosine_scores(query_vec, doc_vecs)
    order = sorted(range(len(shas)), key=lambda i: (-scores[i], i))

    print("Standard vector search baseline (not Vestige)")
    print("model: %s" % MODEL_NAME)
    print("corpus commits: %d" % len(shas))
    print("query:")
    print(query)
    print("top %d:" % TOP_K)
    shown = order[:TOP_K]
    for rank, index in enumerate(shown, start=1):
        print("#%d %s cosine=%.6f" % (rank, shas[index], scores[index]))
    cause_rank = None
    for rank, index in enumerate(order, start=1):
        if shas[index] == cause:
            cause_rank = rank
            break
    if cause_rank is None:
        print("cause %s rank: absent" % cause)
        sys.exit(1)
    print("cause %s rank: %d" % (cause, cause_rank))

    # Same scores as the full corpus. This only restricts which commits are ranked.
    walked = load_lines(walk_path)
    if not walked:
        stop("the walk returned no commits")
    positions = {}
    for index, sha in enumerate(shas):
        positions[sha] = index
    subset = []
    seen = set()
    for sha in walked:
        if sha in seen:
            continue
        seen.add(sha)
        if sha not in positions:
            stop("a walk commit is outside the corpus")
        subset.append(positions[sha])
    subset_order = sorted(subset, key=lambda i: (-scores[i], i))
    subset_rank = None
    for rank, index in enumerate(subset_order, start=1):
        if shas[index] == cause:
            subset_rank = rank
            break
    if subset_rank is None:
        print(
            "among the %d commits causal_walk returned: %s rank absent"
            % (len(subset_order), cause)
        )
        sys.exit(1)
    print(
        "among the %d commits causal_walk returned: %s rank %d"
        % (len(subset_order), cause, subset_rank)
    )
    for rank, index in enumerate(subset_order, start=1):
        print("#%d %s cosine=%.6f" % (rank, shas[index], scores[index]))


if __name__ == "__main__":
    main()
