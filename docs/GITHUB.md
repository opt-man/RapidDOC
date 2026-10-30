# Uploading this prepared repository

This folder is the repository root. Do not upload its parent directory, which contains paper-edit backups and unrelated working files.

The local Git repository has no remote and no commit created by this preparation. Review the files, choose an explicit code license with the authors, then create an empty GitHub repository using your account. From this folder:

```bash
git add .
git diff --cached --stat
git commit -m "Initial RapidDOC research artifact"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL; none is invented here. The included demo MP4 is about 18.5 MB, below GitHub's 100 MB per-file limit. The README offers both a YouTube player link and the MP4 download. CI performs offline artifact checks; it does not download native solvers or launch training/hardware.

Before pushing, inspect publication metadata and third-party notices. Frozen manifests may contain historical local paths, retained for provenance. They are not runtime configuration. The latest paper in `paper/` contains the four authors, both corresponding-author markers, the confirmed contact email and the video link.
