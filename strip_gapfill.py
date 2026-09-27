#!/usr/bin/env python3
"""
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

Delete gap-fill extrusions from Orca gcode - except near the solid rings.

Neither 'Apply gap fill = Nowhere' nor 'Filter out tiny gaps' reaches the gap fill the
perimeter generator emits between wall loops, so on a lattice it can be a third of the
print time to lay a few grams of sub-0.2 mm slivers.

The solid cuff at the bottom and the rim at the top are different: there the generator
calls one of the band's own walls gap fill, so stripping it leaves the ring a single
thin loop and it will not print. Those two zones are left alone; only the lattice in
between is stripped.

Orca tags a run ';TYPE:Gap infill' once and then keeps going across retracts, wipes and
travels to the next island without re-tagging, so a stripped run is consumed whole:
every line is dropped until the next feature, and one travel plus a net E correction is
put back so the following feature starts from the right place with the right pressure.

Needs relative E (Orca's default, use_relative_e_distances = 1).

    python3 strip_gapfill.py plamesh.gcode                 -> plamesh_nogap.gcode
    python3 strip_gapfill.py --keep-bottom 6 --keep-top 3 plamesh.gcode
    python3 strip_gapfill.py --keep-below 4.5 --keep-above 161 plamesh.gcode
    python3 strip_gapfill.py --keep-bottom 0 --keep-top 0 plamesh.gcode   # old behaviour
"""
import re, sys, os, argparse

END = (';TYPE:', '; stop printing', '; printing object', ';LAYER_CHANGE',
       ';BEFORE_LAYER_CHANGE', ';AFTER_LAYER_CHANGE', '; CP TOOLCHANGE', 'M600')
NUM = r'(-?\d*\.?\d+)'
ZCOM = re.compile(r'^;Z:\s*' + NUM)
ZMOV = re.compile(r'^G[0-3]\b.*?\bZ' + NUM)

def z_extent(L):
    """print height from the gcode itself, so --keep-top can be measured from the top"""
    lo, hi = None, None
    for ln in L:
        s = ln.strip()
        m = ZCOM.match(s) or ZMOV.match(s)
        if m:
            v = float(m.group(1))
            if v < -50 or v > 2000: continue
            lo = v if lo is None else min(lo, v)
            hi = v if hi is None else max(hi, v)
    return (0.0 if lo is None else lo), (0.0 if hi is None else hi)

def strip(path, travel=30000, keep_bottom=5.0, keep_top=5.0,
          keep_below=None, keep_above=None):
    L = open(path).read().split('\n')
    zlo, zhi = z_extent(L)
    if (keep_below is None and keep_above is None
            and zhi - zlo < keep_bottom + keep_top + 2):
        keep_bottom = keep_top = 0.0        # too short to have rings (a card plate): strip it all
    lo = keep_below if keep_below is not None else zlo + keep_bottom
    hi = keep_above if keep_above is not None else zhi - keep_top
    out = []; gap = False; x = y = z = None; esum = 0.0
    dropped = runs = kept = 0
    curz = zlo

    def flush():
        nonlocal esum
        if z is not None: out.append(f'G1 Z{z}')
        if x is not None: out.append(f'G1 X{x} Y{y} F{travel}')
        if abs(esum) > 1e-6: out.append(f'G1 E{esum:.5f} F1800')      # keep the E state honest
        esum = 0.0

    for ln in L:
        s = ln.strip()
        if gap and (any(s.startswith(e) for e in END) or re.fullmatch(r'T\d', s)):
            flush(); gap = False
        if not gap:
            m = ZCOM.match(s) or ZMOV.match(s)                        # follow the layer height
            if m:
                v = float(m.group(1))
                if -50 < v < 2000: curz = v
            if s.startswith(';TYPE:'):
                if s[6:].strip() == 'Gap infill':
                    if lo <= curz <= hi:                              # lattice: strip it
                        gap = True; runs += 1; x = y = z = None; esum = 0.0
                    else:                                             # a ring: leave it alone
                        kept += 1
                else:
                    gap = False
            out.append(ln); continue
        if re.match(r'G[0-3]\b', s):                                   # inside a stripped run
            mxy = re.search(r'X' + NUM + r'\s+Y' + NUM, s)
            if mxy: x, y = mxy.group(1), mxy.group(2)
            mz = re.search(r'\bZ' + NUM, s)
            if mz: z = mz.group(1); curz = float(mz.group(1))
            me = re.search(r'\bE' + NUM, s)
            if me:
                ev = float(me.group(1))
                # printing extrusion is discarded; retract / unretract / wipe must carry over
                if not (mxy and ev > 0): esum += ev
            dropped += 1
        # everything inside the run is dropped, comments included
    if gap: flush()
    p = re.sub(r'\.gcode$', '', path) + '_nogap.gcode'
    open(p, 'w').write('\n'.join(out))
    return p, dropped, runs, kept, zlo, zhi, lo, hi

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--keep-bottom', type=float, default=5.0,
                    help='mm above the first layer to leave gap fill alone (bottom cuff)')
    ap.add_argument('--keep-top', type=float, default=5.0,
                    help='mm below the last layer to leave gap fill alone (top rim)')
    ap.add_argument('--keep-below', type=float, help='absolute Z instead of --keep-bottom')
    ap.add_argument('--keep-above', type=float, help='absolute Z instead of --keep-top')
    a = ap.parse_args()
    for f in a.files:
        p, d, r, k, zlo, zhi, lo, hi = strip(f, keep_bottom=a.keep_bottom, keep_top=a.keep_top,
                                             keep_below=a.keep_below, keep_above=a.keep_above)
        print(f'{os.path.basename(f)}: print Z {zlo:.2f}..{zhi:.2f}, stripping between '
              f'Z {lo:.2f} and {hi:.2f}')
        print(f'  dropped {d} lines across {r} gap-fill runs, kept {k} runs in the rings '
              f'-> {os.path.basename(p)}')
