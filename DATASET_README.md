# Sign Language Detection — Dataset

## Quick start (no login, no Kaggle account)

```bash
pip install pyarrow pillow opencv-python mediapipe
python setup_dataset.py
```

That downloads 74 MB and builds the full `dataset/` tree below in about a minute.
Safe to rerun — it skips the download if the file is already there.

Direct file link if you'd rather grab it yourself:
`https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet`

## What's here

```
dataset/
  alphabet/                 10,873 real photos, 26 ASL letter classes (A–Z)
    train/<A..Z>/*.jpg      7,609
    val/<A..Z>/*.jpg        1,628
    test/<A..Z>/*.jpg       1,636
  words/<class>/            EMPTY — you fill this with collect_words.py
  manifest_alphabet.csv     path, label, split, width, height
  preview_alphabet.png      sample grid, sanity check

setup_dataset.py            ONE COMMAND: download + unpack + split everything
collect_words.py            webcam collector for your own word signs
build_splits.py             quality screen + stratified split + health report
extract_dataset.py          how alphabet/ was produced (rerun if needed)
```

All images are 224×224 JPEG, class-balanced, stratified 70/15/15. No image
appears in more than one split.

## Why two datasets

The assignment says "recognize some known words of your choice." Public
**word**-level sign data is all video (WLASL, the 6.6 GB ASL word set) and needs
a sequence model — far more work than this task calls for. Public **static image**
data is all alphabet-level.

So the plan is:

| Dataset | Source | Use |
|---|---|---|
| `alphabet/` | [Marxulia/asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) on Hugging Face | Get the whole pipeline working end-to-end tonight. Train, hit ~95%, prove the GUI works. |
| `words/` | You, via `collect_words.py` | The actual deliverable. 9 word classes recorded from your webcam. |

Train on `alphabet/` first. It de-risks everything — if accuracy is bad you know
it's your code, not your data. Then swap in `words/` and retrain; the only thing
that changes is the folder path and the number of output classes.

## The 9 word classes

All chosen to be **static** and **one-handed**, so a single-frame image
classifier can separate them. Motion-based signs need video models — avoided.

`hello` · `yes` · `no` · `thanks` · `please` · `stop` · `help` · `iloveyou` · `nothing`

Keep `nothing` (empty frame / random hand shapes). Without it the model must
output a word for every frame, and your live demo will flicker through
predictions with nobody signing. This is the single most common mistake in this
project.

## Collecting your word data

```bash
pip install opencv-python mediapipe
python collect_words.py
```

`n`/`p` switch word, hold `SPACE` to record a burst, `c` single shot, `d` undo,
`q` quit. MediaPipe crops to your hand automatically, so the model learns the
hand and not your bedroom wall.

**Target 300–500 images per word.** Below ~150 the model memorises. Vary distance,
hand angle, position in frame, lighting and sleeves while recording, and split it
across 4–6 short sessions in different spots — one long session means one
background, and the model will learn the background.

Then:

```bash
python build_splits.py          # → dataset/words_split/ + health report
```

It drops corrupt files, exact and near-duplicates (perceptual hash) and blurry
frames, then splits 70/15/15 per class and prints how many images each word still
needs. The dedup matters: burst capture makes near-identical frames, and if
copies land in both train and test your test accuracy is fiction.

## Known limitations

- The alphabet set includes a handful of line drawings and one frame with a
  printed letter in it — minor label noise, harmless at this scale.
- Source images are small (~120×170) and upscaled to 224, so they look soft.
  Fine for training; your own webcam data will be sharper.
- `J` and `Z` are motion signs in real ASL. They're single frames here, so
  expect them to be the weakest classes.

## Next step

Training script (transfer learning on MobileNetV2), then the GUI with upload +
live video and the 6 PM–10 PM time gate.
