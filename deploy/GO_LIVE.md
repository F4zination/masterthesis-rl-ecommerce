# Go-live runbook — human validation study

Follow this in order on the VPS after pulling the current repository state.
Every step has a verification; do not continue past a failed one.

Assumed VPS layout (from the Makefile defaults): the repository lives at
`/root/MasterThesisProject`. Substitute your path if it differs.

```bash
export STUDY=/root/MasterThesisProject
```

---

## 0. Before you leave your laptop

**The frozen policy artifacts are NOT in git.** `.gitignore` excludes
`OfflineTraining/outputs/` and every `*/data/` directory, so `git pull` will
not bring them. If you skip Step 1 the VPS will keep serving whatever stale
artifacts are already there, and they will not match the hashes recorded in
the release manifest.

Push the current state first:

```bash
git push
```

Confirm on your laptop that these three files are the ones the manifest
recorded:

```bash
cd ~/development/masterion/MasterThesisProject
sha256sum OfflineTraining/outputs/ppo_policy.pt \
          OfflineTraining/outputs/trained_policy.json \
          DemoSiteV2/data/demosite.db
```

Expected (verified against `clickworker_release_20260803`):

| File | SHA-256 (first 16) |
|---|---|
| `OfflineTraining/outputs/ppo_policy.pt` | `249ae69b1dfd720f` |
| `OfflineTraining/outputs/trained_policy.json` | `2ee5e02cf17c2ede` |
| `DemoSiteV2/data/demosite.db` | `3fd34b5e17302493` |

The PPO **weights** hash (`model_state_sha256`) is `69d30a27c7cbc1ca…` and is
the reproducible provenance key; the file-byte hash differs on every retrain
and is expected to.

### Getting the artifacts onto the machine that can reach the VPS

The laptop is dual-boot and the two halves have different capabilities: the repo
checkout and the frozen artifacts sit on the Windows partition, but the SSH
identity for `thedemoshop.live` exists only on the Linux one. Windows ships an
OpenSSH client, but with no key for the VPS and `ssh-agent` disabled it cannot
run Step 1. Do not copy the private key over to Windows to work around this —
move the three artifacts to Linux instead.

Stage them on Windows in a directory that mirrors the repo-relative paths, so
the Step 1 commands work unchanged, and outside the repo so nothing can dirty
`git status`:

```powershell
$src = "C:\Users\finnr\development\masterion\MasterThesisProject"
$dst = "C:\Users\finnr\go-live-payload"
New-Item -ItemType Directory -Force -Path "$dst\OfflineTraining\outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\DemoSiteV2\data" | Out-Null
Copy-Item "$src\OfflineTraining\outputs\ppo_policy.pt"       "$dst\OfflineTraining\outputs\"
Copy-Item "$src\OfflineTraining\outputs\trained_policy.json" "$dst\OfflineTraining\outputs\"
Copy-Item "$src\DemoSiteV2\data\demosite.db"                 "$dst\DemoSiteV2\data\"
```

Write a `SHA256SUMS` beside them so both later verifications become one command.
It must use LF endings and no BOM or `sha256sum -c` will reject it:

```powershell
$dst = "C:\Users\finnr\go-live-payload"
$rel = "OfflineTraining/outputs/ppo_policy.pt",
       "OfflineTraining/outputs/trained_policy.json",
       "DemoSiteV2/data/demosite.db"
$lines = foreach ($r in $rel) {
  $p = Join-Path $dst $r.Replace("/", "\")
  "$((Get-FileHash $p -Algorithm SHA256).Hash.ToLower())  $r"
}
[IO.File]::WriteAllText("$dst\SHA256SUMS", (($lines -join "`n") + "`n"),
                        (New-Object Text.UTF8Encoding $false))
```

Reboot into Linux. Confirm Fast Startup is off first — with
`HiberbootEnabled=1` or a `C:\hiberfil.sys` present, Windows only half-shuts-down
and Linux will refuse the NTFS mount or force it read-only. Restarting rather
than shutting down also guarantees a full shutdown.

Identify the Windows partition by size and model; Linux lives on the Crucial
P310, so do not touch that one:

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MODEL
```

Mount it read-only — nothing here needs writing, and `ro` cannot damage the
volume. Take the device name from `lsblk`, because Windows' partition numbering
does not map onto Linux device names:

```bash
sudo mkdir -p /mnt/win && sudo mount -o ro /dev/nvme0n1p3 /mnt/win
```

```bash
cp -r /mnt/win/Users/finnr/go-live-payload ~/go-live-payload
```

```bash
cd ~/go-live-payload && sha256sum -c SHA256SUMS && sudo umount /mnt/win
```

Three `OK` lines, matching the table above, and this step's verification is
done. Run Step 1 from `~/go-live-payload`.

> `trained_policy.json` is a 1.5 MB text file with a pinned byte hash. Any route
> that rewrites line endings — copy-paste, a text-mode transfer, `core.autocrlf`
> — changes the bytes and breaks the hash, and it would not surface until the
> manifest disagrees at Step 8. `cp` and `scp` are binary-safe. A USB stick is an
> equally valid route; the three files together are under 4 MB, so exFAT or even
> FAT32 is fine.

---

## 1. Copy the frozen artifacts to the VPS

Run from your laptop, not the VPS.

```bash
scp OfflineTraining/outputs/ppo_policy.pt      root@<study-host>:/root/MasterThesisProject/OfflineTraining/outputs/
scp OfflineTraining/outputs/trained_policy.json root@<study-host>:/root/MasterThesisProject/OfflineTraining/outputs/
scp DemoSiteV2/data/demosite.db                 root@<study-host>:/root/MasterThesisProject/DemoSiteV2/data/
```

> `DemoSiteV2/data/demosite.db` is the frozen bandit policy **and** the V2
> event ledger in one file. Copying it replaces both, which is what you want
> here: it installs the locked policy and a ledger containing no sessions.
> Step 6 is where you clear the ledger again after the pilot, and it must
> preserve the policy tables.

**Verify on the VPS** that the hashes match your laptop:

```bash
cd $STUDY
sha256sum OfflineTraining/outputs/ppo_policy.pt \
          OfflineTraining/outputs/trained_policy.json \
          DemoSiteV2/data/demosite.db
```

If you staged a `SHA256SUMS` as in Step 0, send it too and let the machine do
the comparison instead of your eyes — its paths are repo-relative, so it checks
out from `$STUDY`:

```bash
scp SHA256SUMS root@<study-host>:/tmp/
```

```bash
cd $STUDY && sha256sum -c /tmp/SHA256SUMS
```

> Keep it in `/tmp`, **not** in `$STUDY`. The manifest builder reads
> `git status --porcelain=v1 --untracked-files=all`, so an untracked
> `SHA256SUMS` inside the repo raises `dirty_worktree` at Step 8 and fails the
> clean-worktree item in Step 11.

---

## 2. Pull and configure

```bash
cd $STUDY
git pull
git log --oneline -1     # expect the commit you pushed in Step 0
git status --porcelain   # expect empty
```

Edit `$STUDY/.env` and set:

```bash
# Recruit only the three cells of Design Revision 1. The other two scenarios
# stay deployed and digest-locked but are never assigned.
STUDY_PERSONAS=fastbuyer,detailedcomparator,windowshopper

# No per-cell quota: recruitment stops on the hard end date, not on counts.
# Set high enough that the dispatcher never closes a cell.
TARGET_PER_CELL=100000
```

Leave every other value as it is. Do not commit `.env`; it is gitignored and
the manifest never reads it.

**Verify** the value in `.env` resolves to the intended subset (the loader fails
closed on an unknown or duplicated name):

```bash
cd $STUDY
env $(grep -E '^STUDY_PERSONAS=' .env | xargs) python3 -c "
import sys; sys.path.insert(0, '.')
from StudyDispatcher.app import config
print('recruiting:', config.PERSONAS)
"
```

Expect exactly `['fastbuyer', 'detailedcomparator', 'windowshopper']`. A typo
raises `ValueError` here rather than silently recruiting a cell the shops
would reject.

After Step 3 has started the container, re-check the value the **running**
dispatcher actually holds, which is the one that matters:

```bash
cd $STUDY
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" \
  exec -T dispatcher python -c "from app import config; print(config.PERSONAS)"
```

---

## 3. Build and deploy

```bash
cd $STUDY
make update-shops
```

That target checks the gateway, verifies the frozen artifacts are present and
non-empty, validates the V3 checkpoint inside the image, rebuilds v2 and v3,
then reloads the shared nginx.

The dispatcher is not covered by `update-shops`; rebuild it explicitly:

```bash
cd $STUDY
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" \
  up -d --build --no-deps dispatcher
```

**Verify** all services are up:

```bash
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" ps
```

> The V3 image installs `torch` from PyPI, which on Linux pulls the CUDA
> runtime stack (~2–3 GB). The build is slow and disk-hungry the first time.
> This is what `DemoSiteV3/requirements.lock` records, so it is expected — but
> check free disk before building: `df -h /`.

---

## 4. Smoke test the frozen policies

Both conditions must serve their intended policy and never fall back.

```bash
cd $STUDY
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" \
  logs v3 --tail 40 | grep -i "policy runtime\|fallback\|checkpoint"
```

Expect `policy_mode=ppo_only`, `timing_enabled=False`, and no fallback lines.
`REQUIRE_PPO_CHECKPOINT=true` means V3 refuses to start rather than silently
degrade, so a running container is itself the check.

Walk one session by hand in a browser on each domain:

- `https://start.thedemoshop.live` → enter a nickname → you are redirected to
  v2 or v3 with a persona
- the scenario modal appears **once**; returning to the home page mid-session
  must not show it again
- the footer legal links stay on the shop domain and return you to the shop,
  not to the study entry form
- product images render (check a product page and the recommendation widget)
- `SAVE10` can be entered in the promo box on cart/checkout
- ending the session reveals a completion code

---

## 5. Pilot — up to 20 sessions

Preregistration Section 2: the pilot is excluded from confirmatory analyses,
and any change it motivates must be made **before** main recruitment and
documented as a preregistration revision.

Run up to 20 real sessions across both conditions and all three personas.
Watch for anything that would corrupt the confirmatory data: broken images,
double assignments, missing events, fallback decisions.

**Verify** the pilot recorded what you expect:

```bash
cd $STUDY
sqlite3 StudyDispatcher/data/dispatcher.db \
  "SELECT condition, persona, COUNT(*) FROM assignments GROUP BY 1,2 ORDER BY 1,2;"
sqlite3 DemoSiteV2/data/demosite.db "SELECT COUNT(*) FROM events;"
sqlite3 DemoSiteV3/data/demosite.db "SELECT COUNT(*) FROM events;"
```

Confirm only the three recruited personas appear.

---

## 6. Archive the pilot and release clean ledgers

**Stop the stack first** so nothing writes mid-archive.

```bash
cd $STUDY
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" down
```

Archive the volumes (never delete — the pilot is evidence):

```bash
cd $STUDY
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p archive/pilot_$STAMP
cp DemoSiteV2/data/demosite.db      archive/pilot_$STAMP/v2_demosite.db
cp DemoSiteV3/data/demosite.db      archive/pilot_$STAMP/v3_demosite.db
cp StudyDispatcher/data/dispatcher.db archive/pilot_$STAMP/dispatcher.db
echo "Pilot archived $(date -Is), commit $(git rev-parse HEAD)" > archive/pilot_$STAMP/README.txt
```

Now clear **only** the collected-data tables. The V2 file also holds the
frozen bandit policy and the catalogue, which must survive:

```bash
cd $STUDY
for DB in DemoSiteV2/data/demosite.db DemoSiteV3/data/demosite.db; do
  sqlite3 "$DB" "
    DELETE FROM events;
    DELETE FROM orders;
    DELETE FROM order_items;
    DELETE FROM cart_items;
    DELETE FROM decision_logs;
    DELETE FROM session_discounts;
    DELETE FROM sqlite_sequence;
    VACUUM;
  "
done
sqlite3 StudyDispatcher/data/dispatcher.db "DELETE FROM assignments; VACUUM;"
```

**Verify** the ledgers are empty and the policy and catalogue survived:

```bash
cd $STUDY
for DB in DemoSiteV2/data/demosite.db DemoSiteV3/data/demosite.db; do
  echo "== $DB"
  sqlite3 "$DB" "SELECT 'events',COUNT(*) FROM events
    UNION ALL SELECT 'orders',COUNT(*) FROM orders
    UNION ALL SELECT 'decision_logs',COUNT(*) FROM decision_logs
    UNION ALL SELECT 'cart_items',COUNT(*) FROM cart_items
    UNION ALL SELECT 'session_discounts',COUNT(*) FROM session_discounts
    UNION ALL SELECT 'products (keep 249)',COUNT(*) FROM products;"
done
sqlite3 DemoSiteV2/data/demosite.db \
  "SELECT 'bandit_arm_stats (keep 5088)', COUNT(*) FROM bandit_arm_stats;"
sqlite3 StudyDispatcher/data/dispatcher.db "SELECT 'assignments', COUNT(*) FROM assignments;"
```

The five ledger tables must read 0, `products` 249, and V2's
`bandit_arm_stats` 5088. **If `bandit_arm_stats` is not 5088 you have wiped
the frozen policy — restore `DemoSiteV2/data/demosite.db` from Step 1 and
redo the deletes.**

Bring the stack back up:

```bash
cd $STUDY
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" up -d
```

---

## 7. Capture immutable image identifiers

```bash
cd $STUDY
for img in demosite-v2:latest demosite-v3:latest study-dispatcher:latest; do
  printf '%-26s %s\n' "$img" "$(docker image inspect --format '{{.Id}}' $img)"
done
for img in nginx:1.27-alpine certbot/certbot:latest; do
  printf '%-26s %s\n' "$img" "$(docker image inspect --format '{{index .RepoDigests 0}}' $img)"
done
```

Locally built images have no registry digest, so their `.Id` is the immutable
identifier. `nginx` and `certbot` are pulled, so use the `name@sha256:` digest.
Tags such as `latest` are rejected by the builder.

---

## 8. Generate the final release manifest

Use a fresh release id — the builder refuses to overwrite a non-empty
directory.

```bash
cd $STUDY
make study-release-manifest \
  STUDY_RELEASE_ID=clickworker_release_final \
  STUDY_BASELINE_ID=clickworker_release_20260803 \
  STUDY_IMAGE_ARGS="--image v2=<ID> --image v3=<ID> --image dispatcher=<ID> --image nginx=nginx@sha256:<DIGEST> --image certbot=certbot@sha256:<DIGEST>"
```

Then read every blocker:

```bash
cd $STUDY
python3 -c "
import json
d=json.load(open('Experiments/study_releases/clickworker_release_final/release_manifest.json'))
print(d['status'], d['readiness']['blocker_count'])
for b in d['readiness']['blockers']: print(' -', b['code'])
print('ledgers clean:', d['pre_recruitment_data']['clean_for_confirmatory_recruitment'])
"
```

**Expected to have cleared by now:** `dependency_lock_missing`,
`pre_recruitment_data_not_empty`, `image_identifiers_missing`.

**Expected to remain — see Section 10 below.**

---

## 9. Fill and lock the preregistration

Only now do the two `TO_BE_FILLED` fields have values.

In `Experiments/ClickworkerPreregistration.md`:

1. **Study release commit** → the commit you deployed (`git rev-parse HEAD`)
2. **Reference-data manifest** → `Experiments/study_releases/clickworker_release_final/release_manifest.json`
3. **Status** → change `draft; commit and timestamp before confirmatory recruitment` to `locked YYYY-MM-DD`
4. Section 13.1 → replace both `RECORD_REFERENCE_PENDING` entries with the
   identifier or date of the ethics and supervisor records
5. Section 13.2 → tick the boxes you have completed

Then commit and push. **This commit is the timestamp of the lock**, so make it
before the first participant.

```bash
cd $STUDY
git add Experiments/ClickworkerPreregistration.md
git commit -m "docs: lock the preregistration for recruitment"
git push
```

---

## 10. What will still be blocked, and why that is acceptable

Three blockers cannot be cleared, and the reason is structural rather than a
defect to fix:

- `v2_policy_provenance_mismatch`
- `v3_policy_provenance_mismatch`
- `baseline_commit_mismatch`

All three compare a *commit hash* recorded inside an artifact against
`HEAD`. The policies and the simulator reference were built at commit
`2d672c3`, and HEAD has since advanced by the commits that carry the
preregistration text and the reference itself. A version-controlled reference
can never have been generated at the commit that contains it — the hash would
have to predict itself. Regenerating does not converge; it only trades these
three for `dirty_worktree`.

Every substantive check inside them passes, and this is worth quoting in the
thesis rather than paraphrasing:

- `policy_was_built_from_clean_worktree`: **true** for both policies
- build/training script hashes match the release: **true**
- training-dataset hash matches the selected input: **true**
- baseline input hashes (`v2_db`, `v3_checkpoint`): **true** — `archetype_config`
  holds on the laptop but not on the VPS; see the subsection below
- baseline code hashes: **all match on the laptop** — three of the five diverge
  on the VPS; see the subsection below

So the artifacts, the reference and the code are provably consistent with each
other; only the self-referential commit equality fails. Record that in the
methods chapter and move on.

`preregistration_commit_mismatch` may also appear for the same reason: the
declared release commit cannot equal the HEAD created by declaring it.

### Two further blockers, if you generate the manifest on the VPS

Step 8 run on the VPS additionally raises:

- `baseline_artifact_mismatch`
- `baseline_code_mismatch`

Both are line-ending artifacts rather than code changes. The laptop has
`core.autocrlf=true` and `.gitattributes` carries no `text=auto` rule, so git
stores LF, materialises CRLF on Windows, and materialises LF on the VPS. The
baseline in `clickworker_release_20260803` was generated on Windows
(`runtime.platform = Windows-11-…`), so it recorded SHA-256s of a CRLF working
copy — `CustomerSimulation/config/archetypes.yaml` wholly CRLF, and
`simulation/state.py`, `shared_schema/constants.py` and `shared_schema/features.py`
with mixed endings, where tooling had written bare LFs into files git checked out
as CRLF. The VPS recomputes those same files from the canonical LF checkout and
necessarily gets different digests for identical content.

Do **not** reach for `core.autocrlf=true` on the VPS to make the digests agree.
It would also put CRLF into `deploy/init-letsencrypt.sh` and
`deploy/prepare-study-artifacts.sh`, which then fail with
`bad interpreter: /bin/bash^M`.

Prove the content is identical using blob object ids instead. Git computes those
from normalized content, so they are the same on both machines. Run this at the
same commit on the laptop and on the VPS, and compare:

```bash
for p in CustomerSimulation/config/archetypes.yaml \
         CustomerSimulation/simulation/state.py \
         SharedSchema/shared_schema/constants.py \
         SharedSchema/shared_schema/features.py; do
  printf '%-46s %s\n' "$p" "$(git rev-parse HEAD:$p)"
done
```

Identical object ids on both machines mean the tracked content is identical and
only its on-disk line-ending representation differs. That, together with an empty
`git status --porcelain` on both, is the record to cite: what is
environment-dependent is the manifest's byte-level digest comparison, not the
code it is comparing.

The permanent fix is a `.gitattributes` carrying `* text=auto eol=lf`, a
`git add --renormalize .`, and a baseline regenerated on Linux. That is a change
to make after the study, not days before locking the preregistration.

---

## 11. Go / no-go

Recruit only when all of these are true:

- [ ] Frozen artifact hashes on the VPS match your laptop (Step 1)
- [ ] `git status --porcelain` on the VPS is empty
- [ ] Dispatcher recruits exactly the three personas (Step 2)
- [ ] All services running; V3 started with `REQUIRE_PPO_CHECKPOINT=true` and logs no fallback
- [ ] Manual session walkthrough passes on both domains (Step 4)
- [ ] Pilot run and archived (Steps 5–6)
- [ ] Ledgers read 0 / products 249 / bandit arms 5088 (Step 6)
- [ ] Manifest generated with immutable image IDs, and the only remaining blockers are the ones in Section 10
- [ ] Preregistration filled, status locked, committed and pushed **before** the first participant

---

## 12. During recruitment

Check periodically:

```bash
cd $STUDY
sqlite3 StudyDispatcher/data/dispatcher.db \
  "SELECT condition, persona, COUNT(*) FROM assignments GROUP BY 1,2 ORDER BY 1,2;"
sqlite3 DemoSiteV3/data/demosite.db \
  "SELECT COUNT(*) FROM decision_logs WHERE json_extract(context_json,'\$.policy_source') <> 'ppo_policy';"
```

The second must stay **0**. Preregistration Section 3: if a technical fallback
is detected, pause recruitment, preserve already-assigned sessions, verify the
frozen release, and only then resume.

Do **not** run the analysis script against partial data. Looking at outcomes
mid-collection and then continuing is an outcome-dependent stopping decision,
which is exactly what the hard end date exists to prevent.

Recruitment ends **24.08.2026** regardless of how many participants you have.

If you complete Steps 1–9 including the pilot before 03.08.2026, the
preregistered start date holds and no amendment is needed. If the start will
slip, amend the dates *before* 03.08.2026 passes — an amendment made in
advance is a protocol change, one made afterwards is a protocol deviation, and
the difference costs nothing except doing it in the right order.

---

## 13. If something goes wrong

Roll back to the previous image and investigate before touching data:

```bash
cd $STUDY
docker compose --project-directory "$STUDY" \
  -f "$STUDY/docker-compose.yml" -f "$STUDY/docker-compose.vps.yml" down
git log --oneline -5
git checkout <previous-good-commit>
make update-shops
```

Never edit a release manifest by hand, and never delete a data volume —
archive it as in Step 6 and start a fresh one.
