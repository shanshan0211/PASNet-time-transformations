# Publishing the prepared folder

This folder is not uploaded. The repository owner supplies the GitHub account,
repository name, visibility and licenses. No credentials belong in this folder.

1. Choose licenses for code and for data/checkpoints. Add the agreed license
   files; do not automatically apply an open-source license to every material.
2. Create an empty GitHub repository. Keep it private while reviewing the
   contents if desired. When creating it, avoid generating a conflicting README.
3. From this folder, run the following commands, replacing the remote URL with
   the actual repository URL:

```text
git init -b main
git add .
git status --short
git commit -m "Prepare PASNet code and reproducibility evidence"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

GitHub authentication is handled by the user's Git client. Evidence is tracked as ordered archive parts under data/; restored results/
files are ignored. Browser upload is supported: upload the data/ folder with
all parts, then update README.md, reproduce.py and the input/upload manifests.
The old raw-results gzip files and old result batches are not used by this version.

4. Validate a fresh checkout with `python reproduce.py`. Preserve the exact
   commit used for the paper; create a release tag after author review.
5. Archive that fixed release with a service such as Zenodo and obtain a DOI.
   Then update the paper's availability statement and data/software citation.

The existing validation report covers the local candidate's saved-record and
numerical integrity checks. It does not represent a new training run, independent
external review, author approval, or completed public release.
