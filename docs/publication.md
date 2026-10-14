# Publish the prepared repository and input-data release

The source export and verification are complete before publication. Authenticate
GitHub CLI with the account that should own the repository:

```bash
gh auth login
python scripts/publish_release.py --repository OWNER/NFGS \
  --archive release_data/nfpt_comparison_fields.tar
```

The script records the real repository URL in the checksum manifest, commits and
pushes the public code, and publishes release v0.1.0 with the approximately
550 MB archived comparison-data file. It never adds that data file to Git
history. It also produces English and Korean availability text in
`docs/manuscript_code_availability.md` for the manuscript.

The authoring environment includes a locally installed GitHub CLI at
`.local/bin/gh`; that directory and its optional `.local/gh` authentication
configuration are excluded from Git.
