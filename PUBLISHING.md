# Publishing the Prepared Branch

The project is ready to push to `jingxuxie/rl-memory-selection` on branch
`agent/decision-relevant-memory`. From an authenticated machine with `git` and
`rsync`, run:

```bash
./scripts/publish_branch.sh
```

The script removes the obsolete packed-bootstrap files, preserves the official
AAAI author kit inherited from `main`, commits the complete project, and pushes
it to the existing pull request branch.
