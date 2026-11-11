# Comic photo tidy-up

- The original phone photos are in the repo root. They are left untouched.
- **`aligned/`** has the tidied versions, under the same file names. Each one is 3840×2160.
- **`previews/`** has contact sheets for comparison: `before.jpg`, `after.jpg` (with red guide lines) and `dust_before_after.jpg`.

## What was done to each photo
1. **Found the paper.** The clear page holder covers the middle of each spread, so the page's
   rectangle is measured from the top and bottom parts of the paper, where the edges are clean.
2. **Aligned it.** Each photo is rotated, uniformly scaled and shifted. It is then cropped so that:
   - the paper is level,
   - it is centred, with the spine on the centre line,
   - the page is the same height (82% of the frame) in every image.

   There is **no warping**: no perspective correction or stretching, so straight lines stay straight.
3. **Removed dust from the mat.** Small bright specks on the black mouse mat are detected and filled
   in from the surrounding mat texture. The detection threshold adapts to each photo's noise level.
   The paper (plus a margin) and the clear holder are never touched.
4. **Filled the edges.** Where a photo doesn't reach the edge of the frame (mostly the portrait cover
   shots), the gap is filled by extending a strip of plain mat.

## Re-running
```
pip install -r tools/requirements.txt
python3 tools/align.py . aligned
```
