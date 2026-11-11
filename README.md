# Comic photo tidy-up

- The original phone photos are in the repo root. They are left untouched.
- **`aligned/`** has the tidied versions, under the same file names. Each one is 3840×2160.
- **`previews/`** has contact sheets for comparison: `before.jpg`, `after.jpg` (with red guide lines) and `dust_before_after.jpg`.

## What is done to each photo
1. **Find the paper.** The clear page holder covers the middle of each spread, so the page's
   rectangle is measured from the top and bottom parts of the paper, where the edges are clean.
2. **Align it.** Each photo is rotated, uniformly scaled, shifted and cropped. There is no warping:
   no perspective correction or stretching.
   - Every page is level, centred (spine on the centre line) and the same height.
   - Every image is the same size, cropped close around the pages and clips with an even margin.
3. **Black background.** The mouse mat is taken down to pure black, consistently in every photo,
   so dust specks disappear. Nothing is painted in or copied, and the comic and clips are kept as shot.

## Re-running
```
pip install -r tools/requirements.txt
python3 tools/align.py <photo folder> aligned
```
Photos are processed in name order (1, 2, ... 10, 11).
