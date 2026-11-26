# Steel guitar samples

Fifteen plucked notes of a 2017 Martin HD-28 Vintage Series steel-string guitar,
every third semitone from E2 to B5 (16-bit mono WAV, 44.1 kHz).

- Recorded by: Jeff Learman, for Kinwie's Discord SFZ GM Bank
- Source: https://github.com/sfzinstruments/Discord-SFZ-GM-Bank
  (`Discord GM/Melodic/026-Acoustic Guitar (steel)/`)
- Pinned commit: `7a9c478fe331f94f246d33332f0adedb25bbbe27`
- Retrieved: 2026-11-25
- License: Creative Commons CC0 1.0 (public domain dedication)

The repository mixes CC0 and CC BY material and has no repository-wide licence, so the
CC0 dedication for these files is the header of the instrument's own SFZ file, kept
here unchanged as `026-Acoustic Guitar (steel).sfz`:

```
// GM Acoustic Guitar
// 2017 Martin HD28 Vintage Series
// Author: Jeff Learman, for Kinwie's Discord SFZ GM: https://github.com/kinwie/Discord-SFZ-GM-Bank
// License: Creative Commons CC0
```

`air_harp.py` reads the MIDI note number from each file name (`MartinGM2_<note>_...`),
plays the nearest recorded note and resamples it to the requested pitch.
