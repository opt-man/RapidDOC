# Uploading this prepared repository

This folder is the repository root. Do not upload its parent directory, which contains paper-edit backups and unrelated working files.

The local repository is connected to `https://github.com/opt-man/RapidDOC.git`. To publish a reviewed update from this folder:

```bash
git add .
git diff --cached --stat
git commit -m "Use manuscript figures and equations; remove manuscript files"
git push origin main
```

The included demo MP4 is about 18.5 MB, below GitHub's 100 MB per-file limit. The README offers both a YouTube player link and the MP4 download. CI performs offline artifact checks; it does not download native solvers or launch training/hardware.

Before pushing, inspect publication metadata and third-party notices. Frozen manifests may contain historical local paths, retained for provenance. They are not runtime configuration. The full manuscript and LaTeX sources are excluded. The README retains the author/contact information, video, manuscript equations and original figure assets. Removing files in a new commit does not erase copies in earlier Git history.
