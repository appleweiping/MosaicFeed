# Release process

A push of a semantic-version tag such as `v0.3.0` invokes the release
workflow. The workflow refuses a tag that differs from `project.version`,
installs the committed `uv.lock`, runs the static and coverage-gated suite,
builds both source and wheel distributions with the locked build backend, and
installs each distribution into its own clean environment.

The GitHub Release contains the distributions, a CycloneDX 1.5 runtime
dependency SBOM, and `SHA256SUMS`. GitHub also records build-provenance
attestations for every asset. All third-party workflow actions are pinned to
full commit hashes and the job receives only the write permissions needed to
attest and create the release.

Verify a downloaded file with:

```bash
sha256sum --check SHA256SUMS
gh attestation verify mosaicfeed-0.3.0-py3-none-any.whl \
  --repo appleweiping/MosaicFeed
```

PyPI publication is intentionally not claimed or automated until a project
owner configures Trusted Publishing and tests it. A GitHub Release is the
authoritative distribution channel meanwhile.
