# Comic photo tidy-up

- **`1.jpg` … `11.jpg`** in the repo root are the chosen photos. They are left untouched.
- **`alignednew/`** has the final tidied versions, `1.jpg` … `11.jpg`. Each one is 3326×1972.
- `aligned/` is the earlier attempt (mat cleaned by in-painting).
- **`previews/alignednew.jpg`** is a contact sheet of the final set.

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
