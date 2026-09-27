# HomePod (1st gen) replacement mesh

A parametric generator for printable replacement acoustic mesh for the Apple HomePod
(1st generation). It writes STL and 3MF for a seamless diamond-lattice shell, in either a
rigid PLA version that slides over the chassis or a foaming-TPU version that stretches on
like a sock.

Everything is driven from caliper measurements of a real pod, so the same script produces a
part that fits whether the original mesh is still on the speaker or has been stripped off.

```
python3 homepod_shell.py --arcs --max-flare 85 --pitch 2.4 --rib 0.72 --dmax 141.3 \
  --split 28 --stretch 2 --fit 1.0 --gap 0.4 --slide --thick 2.0 \
  --hem 0 --hem-bottom 0 --cap-trim --cord 15 --cord-z 28 -o out
```

---

## Contents

- [What it makes](#what-it-makes)
- [The pod it is built around](#the-pod-it-is-built-around)
- [Install](#install)
- [Recipes](#recipes)
- [How it works](#how-it-works)
- [Option reference](#option-reference)
- [Printing](#printing)
- [Post-processing: strip_gapfill.py](#post-processing-strip_gapfillpy)
- [Fitting](#fitting)
- [Known limits](#known-limits)

---

## What it makes

The shell is a **surface of revolution** carrying two counter-rotating families of helical
ribs. Because each rib is swept in theta and simply wraps around, **there is no seam
anywhere** in the pattern — no start, no stop, no join line.

| variant | how it goes on | flags |
| --- | --- | --- |
| **Rigid, two pieces** | lower half from below, upper half from above, interlocking rebate | `--split 28 --slide --gap 0.4` |
| **Rigid, one piece** | slides down over the whole pod | `--slide --gap 0.4 --cord-slot` |
| **TPU sock** | stretches on | `--fit 0.95` (no `--slide`, no `--gap`) |
| **Test band** | a short ring for dialling in settings | `--band 60 80` |

Four mesh densities are built in (`--variants`), or set `--pitch` and `--rib` yourself:

| name | pitch | rib | ribs (approx) | open area |
| --- | --- | --- | --- | --- |
| fine | 2.1 | 0.65 | ~326 | ~48 % |
| mid-fine | 2.4 | 0.72 | ~288 | ~49 % |
| medium | 3.5 | 1.00 | ~196 | ~51 % |
| coarse | 5.0 | 1.40 | ~138 | ~52 % |

Rib count scales with the circumference, so it shifts a little with `--dmax`; the run report
prints the exact figure.

Each run prints a report: rib count and angle, overall size, open fraction, filament volume,
face count, whether the mesh came out watertight, and two printability checks.

---

## The pod it is built around

All measured with calipers on a real 1st-gen HomePod. These are the script's defaults.

| dimension | value | flag |
| --- | --- | --- |
| Max diameter, bare chassis | 136.6 mm | `--dmax` |
| Max diameter, original mesh still on | 140.0 mm | `--dmax` |
| Bottom roll meets the barrel | z = 28 mm | `--z-bottom` |
| Barrel starts curving in | z = 133 mm | `--z-top` |
| Rubber base diameter | 86.0 mm | `--d-bottom` |
| Plastic top cap diameter | 88.5 mm | `--cap-dia` |
| Body height, base to top surface | 163 mm | `--height` |
| Gap between top surface and cap underside | 4 mm | — |
| Depth of the cap ledge | 3 mm | — |

The 163 mm body height is 172 overall minus the 5 mm cap and the 4 mm slot beneath it.

---

## Install

```
pip install numpy scipy shapely trimesh manifold3d
```

`manifold3d` is not optional — every boolean (the rib union, the cord cut, the top trim) goes
through it. Python 3.9+.

---

## Recipes

### Rigid PLA, two pieces — the usual choice

```
python3 homepod_shell.py --arcs --max-flare 85 --flare-blend 1.5 \
  --pitch 2.4 --rib 0.72 --thick 2.0 --dmax 141.3 \
  --split 28 --joint 4 --stretch 2 --fit 1.0 --gap 0.4 --slide \
  --hem 0 --hem-bottom 0 --cap-trim \
  --cord 15 --cord-z 28 -o out
```

Writes `homepod_p2.4_lower.stl` and `homepod_p2.4_upper.stl`. The halves meet at z = 28 with a
half-wall rebate, and a 15 mm cord hole straddles the seam so each half carries a semicircle.

Use `--dmax 141.3` over the original mesh, `--dmax 137.4` on a bare chassis (chassis diameter
plus 2 × `--gap`).

### Rigid PLA, one piece

```
python3 homepod_shell.py --arcs --max-flare 85 --flare-blend 1.5 \
  --pitch 2.4 --rib 0.72 --thick 2.0 --dmax 141.3 \
  --stretch 2 --fit 1.0 --gap 0.4 --slide \
  --hem 0 --hem-bottom 0 --bottom-border 2 --cap-trim \
  --cord 15 --cord-z 28.5 --cord-slot -o out
```

`--cord-slot` opens the cord hole downward to the bottom edge, so the shell slides on over a
cord that is still plugged in.

### Foaming TPU stretch sock

```
python3 homepod_shell.py --arcs --pitch 2.4 --rib 0.72 --thick 1.2 \
  --dmax 136.6 --fit 0.95 --max-flare 65 --flare-blend 1.5 \
  --hem 0 --hem-bottom 4 --hem-thick 2 --cap-trim -o out
```

No `--slide`, no `--gap`, no `--split` — it hugs the profile and stretches about 4 % onto the
bare chassis. The bottom cuff stops a bare lattice edge curling off the bed.

### Test band

```
python3 homepod_shell.py --arcs --band 60 80 --pitch 2.4 --rib 0.72 ... -o out
```

Twenty minutes instead of eleven hours. Use it to check flow, rib width and fit before
committing to a full print.

### Fit gauge

`--gauge` also writes a flat profile template. Hold it against the bare chassis and look for
light under the edge — it is the fastest way to check the profile against your own pod.

---

## How it works

### The profile

`--arcs` builds the silhouette analytically: a straight barrel at `--dmax`, with a true
circular arc at each end, tangent to the barrel so the joins are smooth. Both radii fall out
of the measurements — nothing is fitted:

```
R  = dmax/2                       barrel
Rb = (z_bottom^2 + A^2) / (-2A)   A = d_bottom/2 - R      bottom roll
Rt = (L^2 + B^2) / (-2B)          B = d_top/2 - R,  L = height - z_top   top shoulder
```

For a stock pod that gives a 28.0 mm bottom roll and a 30.3 mm top shoulder. Each arc is
sampled uniformly in angle so the steep ends are not under-sampled.

Without `--arcs` the profile is instead traced from the HomePod-shaped business card outline
in `homepod_card.py`. That is the original method and it has the right shape, but it reads
`r(z)` off a polygon and comes out slightly stair-stepped, which the spline faithfully
reproduces as a visible flat around z 152–156. **`--arcs` is recommended.**

`--profile file.json` overrides both with your own `[z, diameter]` table.

### The ribs

Each rib is a rectangular tube of `--rib` × `--thick` swept along a helix on the surface. The
rib angle is nudged from `--angle` so that a rib shifted by one full turn lands exactly on its
neighbour — that is what makes the pattern close on itself.

Sampling is uniform along the surface but refined over the end curves, where a coarse chord
would push the bore out by millimetres right at the cap.

### Printability

Two checks run on every build.

**Rib creep** — how far a rib steps sideways per layer: `layer / tan(angle)`. At the default
52° and 0.2 mm layers that is 0.157 mm against a 0.72 mm rib, so each bead lands about 78 % on
the one below and the whole lattice prints unsupported, like a vase.

**Outward creep** — how fast the wall grows outward near the base, against `--thick`. The pod's
bottom genuinely curls under (the true arc reaches 69:1 at z = 0, i.e. 13.9 mm of radius per
0.2 mm layer), which cannot be printed base-down. `--max-flare` caps it by replacing the last
fraction of a millimetre with a cone:

| `--max-flare` | straight flat | diameter lost | outward creep |
| --- | --- | --- | --- |
| 75° | 3.54 mm | 6.59 mm | 0.75 mm/layer |
| 80° | 1.55 mm | 4.13 mm | 1.13 mm/layer |
| **85°** | **0.36 mm** | **1.71 mm** | **2.29 mm/layer** |
| 90° (true arc) | 0.00 mm | 0.00 mm | unclamped — 13.7 measured |

85° is within 0.48 mm of the true arc anywhere on the curve while keeping the worst layer at
2.29 mm. Only the bottom ~0.6 mm — three layers — actually needs help; a short support ring
under the outer lip is enough, and many machines will bridge it unaided.

`--flare-blend` fillets the join where the clamp meets the arc. Without it the clamp line
*crosses* the body rather than touching it, and the slope drops in a single step — a visible
crease.

### The top

`--cap-trim` cuts the shell where its **inner face** reaches `--cap-dia` + `--cap-gap`, so the
opening lands on the plastic cap (88.9 mm by default) regardless of wall thickness. The inner
face is half a wall in *along the surface normal*, which near the cap is nowhere near radial —
hence the trim solves for that rather than for a radius.

There is no solid rim at either opening. If you want one, add it in the slicer as a modifier
object: it is a flat washer, it is trivial to place there, and it keeps the generator out of
the business of guessing how it should meet the lattice.

### The bottom border

`--bottom-border 2` makes the first 2 mm of arc at the base solid, at full wall thickness,
instead of lattice. It is taken **out of** the existing mesh rather than added below it, so the
overall height and diameter do not change — measured at 0.000 mm on both.

The swept rib tubes are capped square to the helix, not to the rim, so their ends hang about
0.23 mm **below** the band — a ragged fringe under an otherwise solid border, and a first layer
of ~280 unsupported islands. So the base is also cut flat on the band's own underside. The
bottom face is then one solid annulus, 875 mm², from the very first layer. `--no-bottom-flat`
turns that off.

The flat cut is the one thing that touches overall height: 167.607 mm becomes 167.381 mm, a
loss of 0.226 mm — 0.14%, and only material that was jagged anyway. Diameter is unchanged.

The cord slot is cut after all of this, so it goes straight through the border and the shell
still slides on over a plugged-in cord.

One-piece builds only; the split halves ignore it.

### The split

`--split Z` cuts the shell in two with a **half-wall rebate**: the lower piece carries the
inner half of the wall up past the cut, the upper carries the outer half down over it, so the
two slide together and locate. `--joint` sets the overlap, `--joint-gap` the clearance.

The lower piece is built conformal (no `--slide`), because it goes on from below where the pod
only widens. Its top swells over 4 mm to meet the upper piece's widened bore.

### The cord opening

`--cord D` cuts a hole of diameter D through the wall at `--cord-z`, on the +X side.

- `--cord-slot` opens it downward to the bottom edge, so a one-piece shell slides on over a
  plugged-in cord.
- `--cord-border` (default 2 mm) frames the opening with a solid band. Without it you get cut
  rib ends hanging in mid air; with it the top of the hole becomes a proper arch.
- Setting `--cord-z` equal to `--split` puts the hole on the seam, giving each half a
  semicircle that reassembles into the full circle.

---

## Option reference

### Profile

| flag | default | meaning |
| --- | --- | --- |
| `--arcs` | off | build the profile from two true arcs (recommended) |
| `--profile FILE` | — | JSON `[[z, diameter], ...]` instead of the built-in profile |
| `--dmax` | 136.6 | max diameter of the body being wrapped |
| `--height` | 163 | rubber base to top surface |
| `--d-bottom` | 86.0 | diameter at the rubber base |
| `--d-top` | 88.5 | diameter at the cap slot |
| `--z-bottom` | 28 | where the bottom roll meets the barrel (`--arcs`) |
| `--z-top` | 133 | where the barrel starts curving in (`--arcs`) |
| `--stretch` | 0 | add height in the straight midsection, curvature untouched |
| `--straight-bottom Z` | — | drop the bottom roll; hold the diameter straight down |
| `--collar` | 0 | straight ring above the top |

### Lattice

| flag | default | meaning |
| --- | --- | --- |
| `--pitch` | 2.1 | rib spacing |
| `--rib` | 0.65 | rib width |
| `--thick` | 2.0 | wall thickness |
| `--angle` | 52 | rib angle from horizontal (snapped so the pattern closes) |
| `--variants` | off | write fine / mid-fine / medium / coarse in one run |
| `--band Z0 Z1` | — | only this height range |
| `--no-union` | off | leave the ribs as separate bodies |

### Fit

| flag | default | meaning |
| --- | --- | --- |
| `--fit` | 0.95 | radial scale; < 1 for a stretch fit, 1.0 for rigid |
| `--gap` | 0 | clearance on the bore, for a rigid shell |
| `--slide` | off | widen the bore to the running maximum so a rigid shell can pass the belly |
| `--layer` | 0.2 | layer height, for the printability checks |
| `--max-flare` | from wall/layer | cap the bottom roll angle off vertical |
| `--flare-blend` | 1.0 | fillet where the clamp meets the body |

### Ends

| flag | default | meaning |
| --- | --- | --- |
| `--hem` | 4 | solid cuff at the top, mm of arc |
| `--hem-bottom` | = `--hem` | solid cuff at the bottom |
| `--hem-thick` | 3.0 | cuff wall thickness |
| `--bottom-border` | 0 | solid border at the base, mm of arc, full wall thickness (one-piece only) |
| `--no-bottom-flat` | off | leave the ragged rib ends hanging below the border instead of cutting the base flat |
| `--hem-thick-top` | = `--hem-thick` | top band wall thickness |
| `--cap-dia` | 88.5 | plastic cap diameter |
| `--cap-gap` | 0.4 | clearance on the cap |
| `--cap-trim` | off | trim the top so the lattice edge ends on the plastic cap |
| `--no-cap-trim` | off | never trim to the cap |
| `--top-z Z` | — | cut the shell flat at this height |

### Split and cord

| flag | default | meaning |
| --- | --- | --- |
| `--split Z` | — | cut in two at this height with an interlocking rebate |
| `--joint` | 4 | rebate overlap length |
| `--joint-gap` | 0.15 | clearance in the rebate |
| `--cord D` | 0 | cord hole diameter |
| `--cord-z` | 28.5 | cord hole centre height |
| `--cord-border` | 2.0 | solid frame around the opening |
| `--cord-slot` | off | open the hole down to the bottom edge |

### Output

| flag | default | meaning |
| --- | --- | --- |
| `-o, --out` | out | output directory (STL + 3MF per piece) |
| `--gauge` | off | also write a flat profile template |

---

## Printing

Full settings for a Snapmaker U1 with a 0.4 mm nozzle are in **`u1_pla_settings.md`**. The
short version:

**Line width is not a free choice.** A horizontal slice through a rib is `rib / sin(angle)`
wide, and you want that to be an exact whole number of beads:

| variant | rib | island in a layer | walls | line width |
| --- | --- | --- | --- | --- |
| fine | 0.65 | 0.826 mm | 2 | **0.413** |
| mid-fine | 0.72 | 0.914 mm | 2 | **0.457** |
| medium | 1.00 | 1.269 mm | 3 | **0.423** |
| coarse | 1.40 | 1.777 mm | 4 | **0.444** |

Get this wrong and the slicer leaves a sliver of gap fill in every island, and the ribs print
narrower than modelled.

**The other settings that matter most**, because this part is hundreds of thousands of
sub-millimetre segments rather than a normal solid:

- **Classic wall generator, not Arachne.** Arachne emits variable-width beads on a 0.9 mm
  island — continuous flow modulation on every rib, which softens the edges.
- **Layer height 0.20 mm.** The self-support margin is computed for it.
- **Slow down for overhangs: on.** Only ~1 % of wall length is flagged, at the apex of each
  diamond row, and it costs minutes.
- **Avoid crossing walls: off.** With hundreds of islands per layer the pathfinding costs
  enormous time and buys nothing.
- **Z-hop 0.4 mm, ramped.** ~270,000 travels past thin ribs.
- **XY compensation 0.** Any offset changes rib width and the fit.
- **Supports off, prime tower off, arc fitting on.** Arc fitting matters — files run to 80 MB+.

Expect roughly 9–11 hours and 45–80 g for a full shell in PLA at 0.2 mm.

---

## Post-processing: strip_gapfill.py

Orca emits gap fill between wall loops no matter what "Apply gap fill" is set to. On a lattice
that is roughly a third of the print time for a few grams of sub-0.2 mm slivers.

```
python3 strip_gapfill.py sliced.gcode                 -> sliced_nogap.gcode
python3 strip_gapfill.py --keep-bottom 8 --keep-top 3 sliced.gcode
python3 strip_gapfill.py --keep-below 4.5 --keep-above 161 sliced.gcode
```

It consumes each gap-fill run whole — Orca tags one once and then keeps going across retracts
and travels without re-tagging — and puts back one travel plus a net E correction so the next
feature starts with the right pressure. Requires relative E (Orca's default).

By default it leaves the bottom and top 5 mm alone: in a solid band such as a cuff or
`--bottom-border`, one of the band's own walls is tagged as gap fill, and stripping it makes
that band unprintable. Prints shorter than
`keep_bottom + keep_top + 2` are stripped entirely, so card plates are unaffected.

---

## Fitting

**Rigid, two pieces.** Lower half up from below, upper half down from above, press together at
the rebate. Glue is optional; the rebate locates them on its own.

**Rigid, one piece.** Feed the cord through the slot and slide the shell down. `--slide`
guarantees the bore never narrows going down, so it cannot jam on the belly.

**TPU sock.** Roll it on from the top. It stretches about 4 % on the barrel; the top opening
has to pass the widest point during installation, which is the limiting hoop.

The top edge tucks into the 4 mm slot between the pod's top surface and the underside of the
plastic cap. The bottom sits against the rubber base.

---

## Known limits

- **The bottom cannot be a true arc and still print base-down.** The pod's base curls under at
  69:1. `--max-flare 85` is within half a millimetre of the real shape; the literal arc needs
  `--max-flare 90` plus supports for the first three layers.
- **The diamonds narrow toward the cap.** The circumference drops 36 % from barrel to cap while
  the rib count is fixed, so cells go from 3.14 mm wide to 2.02 mm while staying 3.90 mm tall.
  Holding the size constant would mean terminating about a third of the ribs in a visible
  decrease round. It is confined to the top 15 mm.
- **The top shoulder is the hard part of the print.** It turns near-horizontal as it closes on
  the cap. Halving the layer height and dropping the nozzle ~15 °C over the last 5 mm fixes
  most of it.
- **The cord hole interrupts the split rebate** for 15 mm of circumference when `--cord-z`
  equals `--split`.
- **STLs reload as non-watertight** in some tools. The union reports watertight; the handful of
  edges shared by more than two faces are a float32 round-trip artefact and slice fine.

---

## Files

| file | what it is |
| --- | --- |
| `homepod_shell.py` | the generator |
| `strip_gapfill.py` | removes gap fill from sliced gcode |
| `u1_pla_settings.md` | full slicer settings for a Snapmaker U1, 0.4 mm nozzle |
| `homepod_card.py` | the HomePod-shaped business card, source of the original profile |
| `homepod_mesh_gcode.py` | gcode post-processor that writes the card's mesh layers |

---

## Credits

**Design by Nic Splattstoesser / Nic's Fix (nicsfix.com)** — mail-in HomePod repair.
All dimensions measured from a real 1st-gen HomePod.

Not affiliated with or endorsed by Apple. "HomePod" is a trademark of Apple Inc., used here
only to identify the device this part fits.

## License

**Noncommercial, attribution required.** This is source-available, not open source — no
OSI-approved licence prohibits commercial use.

| what | licence | file |
| --- | --- | --- |
| Source code (`*.py`) | PolyForm Noncommercial 1.0.0 | `LICENSE` |
| Models, STL/3MF output, profile data, docs | CC BY-NC-SA 4.0 | `LICENSE-MODELS` |

**This includes anything the generator produces.** STL and 3MF files you create by running
these scripts are covered by CC BY-NC-SA 4.0 on the same terms.

Attribute as: **Design by Nic Splattstoesser / Nic's Fix (nicsfix.com)**

You may print these for yourself, modify them, and share them — as long as you credit the
author, do not sell them or use them commercially, and license your derivatives under the
same terms.

Selling printed parts, printing them as part of a repair or resale business, or including
them in a paid product all count as commercial use. For a commercial licence, get in touch
via [nicsfix.com](https://nicsfix.com).
