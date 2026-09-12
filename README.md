# D2 selected sanitized evidence package

**Local candidate, not yet published.** This is a sanitized derivative of selected frozen evidence, not the original research tree and not a runnable training repository. No new experiment or scientific-result regeneration was performed.

## Identity and verification

- Private/local scientific freeze: `e3e11d0cfebcd0893f5cd5bae46cf7f48ba8429a`.
- Private/local packaging freeze: `781a6f148a46d59a89ea40806de8e10eb7b98941`.
- Public derivative identity: not assigned until this package is committed independently; do not substitute either original SHA as its public URL.
- [PROVENANCE.json](PROVENANCE.json) maps every included source file to its original and distributed SHA-256, byte sizes, encoding and transformation.
- [AUDIT.json](AUDIT.json) records local automated checks, not a guarantee of complete secret detection or remote availability.

Original hashes inside historical manifests identify original private files. A changed manifest/log has a different public hash listed in PROVENANCE; do not use historical embedded hashes to validate the sanitized file. Checkpoint hashes identify external assets, not weights shipped here.

Run the offline distributed-file check:

```text
python verify_package.py
```

This validates the package's selected files against its manifest. It does not prove access to the private source, validate model execution, or certify authenticity of the manifest itself. After publication, use the new commit identity plus this manifest.

## Start here

1. [Final result](experiments/rhino_d2/results/p2_normalized_response_formal/final_result.md).
2. [Paired results](experiments/rhino_d2/results/p2_normalized_response_formal/paired_results.json): No detectable change; not equivalence.
3. [Mechanism summary](experiments/rhino_d2/results/p2_normalized_response_formal/mechanism_summary.csv): aggregate secondary support, not causal mediation.
4. [Original P2-07 protocol](experiments/rhino_d2/DINOV3_P2_NORMALIZED_RESPONSE_FORMAL_PROTOCOL.md): byte-preserved historical text, NOT execution authorization.
5. [Sanitized B24 recovery disclosure](experiments/rhino_d2/P2_07_RECOVERY_DISCLOSURE.md): interrupted 33-epoch run, rejected restart attempt, subsequent fresh epoch-zero run, extra compute and governance limitations retained.

## Selection and privacy rules

84 source files are selected: P2-07 tables/raw secondary CSV/epoch CSV, six-arm and pilot summaries/manifests/checkpoint hashes, complete selected logs, configuration, software/hardware version records and recovery notes. Of these, 41 are byte-identical and 43 replace absolute path prefixes with logical placeholders. All 16 files designated mandatory byte-preserved (CSV tables, paired JSON, final result and P2-07 protocol) are unchanged. Scientific values and all existing SHA-256 references are retained.

Placeholders distinguish user home, repository, workspace and small/large Teacher asset roots. They are not executable paths. Python/PyTorch/CUDA versions, GPU model and driver versions are retained; no original user-directory identifiers, absolute drive paths, known token signatures, IP/host identifiers matched the final configured scanner.

Omitted: per-batch JSONL, binary assets/plots, model weights, dataset images, training/probe/queue scripts and most earlier-stage evidence. Full raw evidence is **not** claimed publicly available. Some historical documents refer to these omitted files or earlier stages; their paths remain historical references, not working links promised by this selected package. The main recommendation is historical context; direct P2-07 entries above are the package's supported navigation.

Historical protocol and result files were not rewritten to repair archive-only links, preserving their source hashes. Recovery/other documents may be path-sanitized as explicitly recorded. The protocol's historical implementation is not the later hardened integration API.

No original Git history is to be included when committing this package. Do not push either private freeze ref. Publication, author metadata and anonymous URL availability must be audited separately before the integration README can claim public evidence is available.
