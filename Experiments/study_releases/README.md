# Clickworker study releases

`build_study_release_manifest.py` creates one immutable audit manifest for a
candidate Clickworker deployment. It hashes the exact V2 bandit database, V3
PPO checkpoint, ten-cell aggregate simulator diagnostic, all 160 device/traffic
reference strata, training datasets, archetype configuration, catalogue
databases, relevant code/configuration, and
dependency manifests. It also records Git state, runtime versions, the frozen
deployment settings, and immutable container identifiers.

The builder is deliberately fail-closed. A manifest can be written while it is
a draft, but `ready_for_recruitment` remains `false` when any prerequisite is
unresolved. In particular, a dirty tree, missing required file, substantive
`TO_BE_FILLED` preregistration field, absent/mutable image identifier, catalogue
mismatch, stale simulator baseline, missing dependency lock, or incomplete
embedded policy-build provenance is a blocker. Non-empty shop event/decision
tables or a non-empty dispatcher ledger are also blockers, preventing pilot or
test sessions from being silently pooled into the confirmatory study.

## Build a candidate manifest

Use a new release ID on every attempt. The command refuses to overwrite a
non-empty release directory.

```powershell
.\.venv\Scripts\python.exe Experiments\build_study_release_manifest.py `
  --release-id clickworker_release_20260721_01 `
  --image v2=sha256:<64-hex-image-id> `
  --image v3=sha256:<64-hex-image-id> `
  --image dispatcher=sha256:<64-hex-image-id> `
  --image nginx=nginx@sha256:<64-hex-manifest-digest> `
  --image certbot=certbot@sha256:<64-hex-manifest-digest>
```

The equivalent Make target is:

```text
make study-release-manifest STUDY_RELEASE_ID=clickworker_release_20260721_01 \
  STUDY_IMAGE_ARGS="--image v2=sha256:... --image v3=sha256:... --image dispatcher=sha256:... --image nginx=nginx@sha256:... --image certbot=certbot@sha256:..."
```

`--image` values must be immutable IDs/digests, not tags such as `latest`. The
script does not read or archive `.env`, participant data, credentials, or other
secrets. Repeat `--training-dataset LABEL=PATH` only when replacing both
default training-dataset records with an explicitly labelled set.

## Release sequence

1. Rebuild/export both policies with embedded dataset, code, seed, dynamics,
   and Git provenance; never reuse legacy policy artifacts that cannot prove
   their origin.
2. Lock dependencies and build all images from the intended clean commit.
3. Generate the 2 x 5 aggregate diagnostic and all 160 context-stratified
   simulator references from those exact policy files.
4. Fill and lock the preregistration, including the intended release commit
   and manifest path, before recruitment.
5. Archive any pilot/test volumes and create clean V2, V3, and dispatcher data
   volumes while preserving the frozen policy/catalogue contents.
6. Obtain immutable image IDs/digests, run this builder, and inspect every
   entry in `readiness.blockers`.
7. Recruit only when the manifest says `READY_FOR_RECRUITMENT` and the scripted
   deployment smoke tests pass.

## Current draft

`clickworker_draft_20260721/release_manifest.json` is a diagnostic snapshot,
not a deployment authorization. Its simulator reference is current, includes
all 160 required device/traffic strata, and matches the current policy and code
hashes. Its power analysis also matches that reference. Nevertheless, the
manifest has 11 fail-closed blockers:

1. the worktree is dirty;
2. substantive preregistration fields remain unfilled;
3. the preregistration is not locked;
4. a reproducible dependency lock is missing;
5. the current V2/V3/dispatcher data volumes contain pilot or test rows;
6. the V2 artifact lacks embedded build provenance;
7. the V3 checkpoint does not encode the full finite serving vocabulary;
8. the V3 artifact lacks embedded training provenance;
9. the V2 policy has no trained evidence for some visited contexts/actions;
10. the V3 policy encounters out-of-vocabulary serving states; and
11. immutable identifiers are missing for all five deployment images.

The two state-coverage failures require new policy artifacts or an explicitly
approved change to the serving state domain; provenance metadata alone will
not repair them. Archive the test data rather than pooling it with the
confirmatory sample. After resolving every blocker, generate a new release
directory and rerun the checks; do not edit a manifest by hand.
