#!/usr/bin/env python3
"""
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

HomePod (1st gen) replacement mesh - printable stretch-fit lattice sock.

The body is a surface of revolution, so each rib is a helical arc swept along it: theta
just keeps increasing and wraps on its own, which means the sock has no seam anywhere.
Two counter-rotating rib families make the diamond knit, and a solid band at each end
acts as the cuff (Apple glues the bottom and cinches the top with a drawstring).

Prints unsupported like a vase: on a barrel every rib leans inward or nearly straight
up, and the per-layer sideways creep of a rib is layer_height / tan(angle), far less
than the rib is wide.

    python homepod_shell.py --profile profile.json -o out/
    python homepod_shell.py --pitch 3.5 --rib 1.0 --thick 1.6
    python homepod_shell.py --variants

deps: pip install numpy scipy shapely trimesh manifold3d
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.interpolate import PchipInterpolator   # shape-preserving: no overshoot at the ends
import trimesh

# The body profile is homepod_card.py's own card silhouette scaled up: same formula, same
# proportions. Scaled so the widest point is DMAX it reproduces Nic's caliper anchors on the
# real chassis - flat run 28.7..134.2 mm (measured 28 and 133), overall 164.3 mm (measured
# 164) - so the cards and the mesh share one silhouette. Truncated where the body reaches the
# rubber base at the bottom and the cap slot at the top.
CARD_W, CARD_H, CARD_R, BULGE, RISE = 54.0, 54.0 * 1.14, 0.22 * 54.0, 0.6, 0.8

def clamp_flare(pts, deg):
    """The card outline turns almost horizontal where it meets the base, which is real geometry
    but unprintable: a rib there would step sideways further than the wall is thick. Cap how fast
    the radius may grow off the bed, which straightens the last few mm of the bottom roll into a
    cone and leaves the rest - including the whole top shoulder - untouched."""
    z = np.array([a for a, b in pts]); r = np.array([b / 2 for a, b in pts])
    lim = r[0] + np.tan(np.radians(deg)) * (z - z[0])
    return [(float(a), float(2 * min(b, l))) for a, b, l in zip(z, r, lim)]

def card_profile(dmax=136.6, height=163.0, d_bottom=86.0, d_top=88.5, n=400):
    """(z, diameter) up the body, from the card outline scaled to a real HomePod"""
    from shapely.geometry import Polygon
    W, H, R = CARD_W, CARD_H, CARD_R
    ys = np.linspace(-H / 2, H / 2, n); xs = np.linspace(-W / 2, W / 2, n)
    ring = np.vstack([np.c_[W / 2 - BULGE * (2 * ys / H) ** 2, ys],
                      np.c_[xs[::-1], H / 2 + RISE * (1 - (2 * xs[::-1] / W) ** 2)],
                      np.c_[-(W / 2 - BULGE * (2 * ys[::-1] / H) ** 2), ys[::-1]],
                      np.c_[xs, -H / 2 - RISE * (1 - (2 * xs / W) ** 2)]])
    body = Polygon(ring).buffer(-R, join_style=1, resolution=48).buffer(R, join_style=1, resolution=48)
    c = np.array(body.exterior.coords)
    yy = np.linspace(c[:, 1].min(), c[:, 1].max(), 1600)
    rr = np.array([c[np.abs(c[:, 1] - y) < 0.2][:, 0].max() if (np.abs(c[:, 1] - y) < 0.2).any() else np.nan for y in yy])
    ok = ~np.isnan(rr); yy, rr = yy[ok], rr[ok]
    k = dmax / (2 * rr.max()); z = (yy - yy.min()) * k; d = 2 * rr * k
    top = np.argmax(d)
    zlo = float(np.interp(d_bottom, d[:top], z[:top]))                  # exact, not nearest sample
    zhi = float(np.interp(d_top, d[top:][::-1], z[top:][::-1]))
    k = (z > zlo) & (z < zhi)
    z = np.r_[zlo, z[k], zhi] - zlo; d = np.r_[d_bottom, d[k], d_top]
    if height: z = z * (height / z[-1])          # the card is a shape, not a scale: set width and
                                                 # height from the calipers independently
    keep = np.unique(np.linspace(0, len(z) - 1, 90).astype(int))
    return [(round(float(z[i]), 2), round(float(d[i]), 2)) for i in keep]

def arc_profile(dmax=136.6, height=163.0, d_bottom=86.0, d_top=88.5,
                z_bottom=28.0, z_top=133.0, n=400):
    """(z, diameter) as two exact circular arcs joined by a straight barrel.

    card_profile() has the right shape but it reads r(z) off a polygon by taking max(x) of
    whatever vertices land within +-0.2 mm of each height. Where the polygon's vertices are
    sparse that max jumps, so r(z) comes out stair-stepped and the spline reproduces every
    step as a visible flat - worst around z 152..156, which is the wobble you can see.

    Here each end curve is a true arc, tangent to the barrel (so the join is smooth, no
    crease) and passing through a measured end diameter. Both radii fall straight out of the
    calipers - nothing is fitted:
        R  = dmax/2                       barrel
        Rb = (z_bottom^2 + A^2) / (-2A)   A = d_bottom/2 - R
        Rt = (L^2 + B^2) / (-2B)          B = d_top/2 - R,  L = height - z_top
    Each arc is sampled uniformly in angle, so the steep end is not under-sampled."""
    R, A, B = dmax / 2, d_bottom / 2 - dmax / 2, d_top / 2 - dmax / 2
    L = height - z_top
    Rb = (z_bottom ** 2 + A * A) / (-2 * A)
    Rt = (L * L + B * B) / (-2 * B)
    nb = max(2, int(n * 0.3)); nm = max(2, int(n * 0.4)); nt = max(2, int(n * 0.3))
    fb = np.linspace(math.asin(min(1.0, z_bottom / Rb)), 0.0, nb)     # bottom roll, bed upward
    zb, rb = z_bottom - Rb * np.sin(fb), (R - Rb) + Rb * np.cos(fb)
    zm, rm = np.linspace(z_bottom, z_top, nm), np.full(nm, R)         # straight barrel
    ft = np.linspace(0.0, math.asin(min(1.0, L / Rt)), nt)            # top shoulder
    zt, rt = z_top + Rt * np.sin(ft), (R - Rt) + Rt * np.cos(ft)
    z = np.r_[zb[:-1], zm, zt[1:]]; r = np.r_[rb[:-1], rm, rt[1:]]
    arc_profile.radii = (Rb, Rt)
    return [(round(float(a), 3), round(float(2 * b), 3)) for a, b in zip(z, r)]

class Profile:
    """the body outline as a curve parameterised by arc length, so it can turn under at
    the bottom edge without the radius having to be a function of height"""
    def __init__(self, pts, fit=1.0, n=4000, max_flare=None, gap=0.0, slide=False, blend=1.0):
        p = np.asarray(pts, float)
        rz = np.c_[p[:, 1] / 2 * fit, p[:, 0]]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(rz, axis=0).T))]
        t = d / d[-1]
        tt = np.linspace(0, 1, n)
        self.r = PchipInterpolator(t, rz[:, 0])(tt); self.z = PchipInterpolator(t, rz[:, 1])(tt)
        if slide:
            # A rigid shell goes on from the top, so every section must clear the widest thing it
            # passes over on the way down. Take the running maximum of the radius looking upward:
            # the bore then never narrows going down and the shell cannot jam on the belly.
            self.r = np.maximum.accumulate(self.r[::-1])[::-1]
        if max_flare is not None:                    # clamp on the resampled curve, where it bites
            lim = self.r[0] + np.tan(np.radians(max_flare)) * (self.z - self.z[0])
            over = self.r - lim
            hard = np.minimum(self.r, lim)
            k = np.where(over > 0)[0]
            if blend > 0 and len(k) and k[-1] < len(hard) - 2:
                # The clamp line CROSSES the body instead of touching it, so the join is a corner -
                # the slope drops in a single step and you can see the crease. Fillet just that
                # crossing: a gaussian-weighted blur of the clamped curve, windowed to +-blend mm
                # around it, so the cone, the arc and the base anchor are all left untouched.
                ds = float(np.mean(np.hypot(np.diff(self.r), np.diff(self.z))))
                sig = max(1.0, blend / max(ds, 1e-6))
                t = np.arange(-int(3 * sig), int(3 * sig) + 1)
                ker = np.exp(-0.5 * (t / sig) ** 2); ker /= ker.sum()
                pad = len(t) // 2
                sm = np.convolve(np.r_[np.full(pad, hard[0]), hard, np.full(pad, hard[-1])], ker, 'valid')
                w = np.exp(-((self.z - self.z[k[-1]]) / blend) ** 2)      # 1 at the crossing only
                self.r = hard * (1 - w) + sm * w
            else:
                self.r = hard
        def normals():
            dr, dz = np.gradient(self.r), np.gradient(self.z)
            L = np.hypot(dr, dz); nr, nz = dz / L, -dr / L            # outward normal
            return (-nr, -nz) if np.mean(nr) < 0 else (nr, nz)
        if gap:
            # clearance must follow the NORMAL, not the radius. The wall is built +-thick/2
            # along the normal, so a radial offset stops cancelling wherever the surface is
            # not vertical - near the cap the normal is 19 deg off vertical and the bore ends
            # up inside the pod by millimetres.
            nr, nz = normals(); self.r = self.r + gap * nr; self.z = self.z + gap * nz
        self.s = np.r_[0, np.cumsum(np.hypot(np.diff(self.r), np.diff(self.z)))]
        self.S = float(self.s[-1])
        self.nr, self.nz = normals()
        self.Rref = float(np.mean(self.r))

    def trim_to(self, r_inner, half=0.0):
        """cut the top off where the shell's INNER face reaches r_inner, so it ends hugging
        the plastic cap instead of being sliced off while still wide. The inner face is half a
        wall in along the normal, which up here is nowhere near radial - hence the r - half*nr."""
        g = self.r - half * self.nr
        k = np.where(g >= r_inner)[0]
        if len(k) == 0 or k[-1] >= len(self.r) - 1: return self
        i = k[-1]
        f = (g[i] - r_inner) / max(g[i] - g[i + 1], 1e-9)
        end = [np.interp(f, [0, 1], [a[i], a[i + 1]]) for a in (self.r, self.z, self.nr, self.nz)]
        self.r = np.r_[self.r[:i + 1], end[0]]; self.z = np.r_[self.z[:i + 1], end[1]]
        self.nr = np.r_[self.nr[:i + 1], end[2]]; self.nz = np.r_[self.nz[:i + 1], end[3]]
        self.s = np.r_[0, np.cumsum(np.hypot(np.diff(self.r), np.diff(self.z)))]
        self.S = float(self.s[-1]); self.Rref = float(np.mean(self.r))
        return self

    def slice_z(self, z0, z1, n=400):
        """a sub-profile between two heights, for printing a short test band"""
        k = (self.z >= z0) & (self.z <= z1)
        zz = np.linspace(self.z[k].min(), self.z[k].max(), n)
        return [(float(a), float(2 * np.interp(a, self.z, self.r))) for a, b in zip(zz, zz)]

    def at(self, s):
        f = lambda a: np.interp(s, self.s, a)
        return f(self.r), f(self.z), f(self.nr), f(self.nz)

def tube(P, N, B, w, t, close=True):
    """sweep a w x t rectangle (w across the surface, t through it) along a centreline"""
    c = [P + B * (w / 2) + N * (t / 2), P + B * (w / 2) - N * (t / 2),
         P - B * (w / 2) - N * (t / 2), P - B * (w / 2) + N * (t / 2)]
    V = np.concatenate([x[:, None, :] for x in c], axis=1).reshape(-1, 3)   # n x 4
    n = len(P); F = []
    for k in range(4):
        a, b = k, (k + 1) % 4
        i = np.arange(n - 1) * 4
        F.append(np.c_[i + a, i + b, i + 4 + b]); F.append(np.c_[i + a, i + 4 + b, i + 4 + a])
    if close:
        F.append(np.array([[0, 2, 1], [0, 3, 2]]))
        e = (n - 1) * 4; F.append(np.array([[e, e + 1, e + 2], [e, e + 2, e + 3]]))
    return trimesh.Trimesh(V, np.vstack(F), process=False)

def ribs(prof, pitch, rib, thick, ang, step_mm=2.0, rib_range=None):
    """both rib families as swept solids. The angle is nudged so a rib shifted by one
    full turn lands on its neighbour - that is what makes the pattern close on itself."""
    U = 2 * math.pi * prof.Rref
    m = max(1, round(U * math.sin(math.radians(ang)) / pitch))
    ang = math.degrees(math.asin(min(0.999, m * pitch / U)))
    t = math.tan(math.radians(ang)); step = U / m                 # spacing along the circumference
    a0, a1 = rib_range if rib_range else (0.0, prof.S)
    # coarse through the straight belly, fine over the end curves: a 2 mm step chords badly
    # across the shoulder, which pushes the bore out by millimetres right at the cap
    dv = step_mm * math.sin(math.radians(ang)); ends = min(20.0, (a1 - a0) / 3)
    v = np.linspace(a0, a1, max(8, int((a1 - a0) / dv)))       # NB: not `step` - that is the
    v = np.union1d(v, np.linspace(a0, a0 + ends, max(2, int(ends / 0.5))))   # circumferential rib
    v = np.union1d(v, np.linspace(a1 - ends, a1, max(2, int(ends / 0.5))))   # spacing, used below
    v = v[np.r_[True, np.diff(v) > 0.05]]        # union1d can leave near-duplicates -> zero-length
                                                 # steps -> NaN normals -> junk near the axis
    r, z, nr, nz = prof.at(v)
    out = []
    for sgn in (1, -1):
        for k in range(m):
            th = sgn * v / (t * prof.Rref) + (k + (0.5 if sgn < 0 else 0)) * step / prof.Rref
            ct, st = np.cos(th), np.sin(th)
            P = np.c_[r * ct, r * st, z]
            N = np.c_[nr * ct, nr * st, nz]
            T = np.gradient(P, v, axis=0)                       # v, not unit spacing: the samples
            T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]   # are deliberately non-uniform
            B = np.cross(T, N); B /= np.maximum(np.linalg.norm(B, axis=1), 1e-9)[:, None]
            out.append(tube(P, N, B, rib, thick))
    return out, m, ang

def band(prof, s0, s1, thick, nθ=480):
    """solid ring of revolution between two arc-length stations: the cuff"""
    v = np.linspace(s0, s1, max(2, int((s1 - s0) / 0.8) + 1))
    r, z, nr, nz = prof.at(v)
    return band_at(r, z, nr, nz, thick, nθ)

def band_at(r, z, nr, nz, thick, nθ=480):
    th = np.linspace(0, 2 * math.pi, nθ, endpoint=False)
    ring = []
    for w in (thick / 2, -thick / 2):
        rr, zz = r + w * nr, z + w * nz
        ring.append(np.stack([np.outer(rr, np.cos(th)), np.outer(rr, np.sin(th)),
                              np.repeat(zz[:, None], nθ, 1)], -1))
    grid = np.concatenate([ring[0], ring[1][::-1]], 0)            # closed loop of the cross-section
    nv, nu = grid.shape[:2]
    V = grid.reshape(-1, 3); F = []
    for i in range(nv):
        j = (i + 1) % nv
        a = np.arange(nu); b = (a + 1) % nu
        F.append(np.c_[i * nu + a, i * nu + b, j * nu + b])
        F.append(np.c_[i * nu + a, j * nu + b, j * nu + a])
    return trimesh.Trimesh(V, np.vstack(F), process=False)

def cord_cut(mesh, dia, z0, slot=False, border=0.0, prof=None, thick=2.0, wall_s=None):
    """Notch for the power cord: a hole of diameter dia through the wall at height z0, on the
    +X side only. slot=True opens it straight down past the bottom of the part so the shell can
    be slid on over a plugged-in cord. border>0 frames the opening with a solid band of that
    width, which ties off the cut rib ends and turns the top of the hole into an arch instead
    of a row of stubs hanging in mid air."""
    b = mesh.bounds
    x1 = float(max(np.hypot(b[1][0], b[1][1]), np.hypot(b[0][0], b[0][1]))) + 10.0
    zb = float(b[0][2]) - 5.0

    def cutter(d):
        c = trimesh.creation.cylinder(radius=d / 2, height=x1)
        c.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        c.apply_translation([x1 / 2, 0, z0])          # axis radial, starts on the centreline
        if not slot: return c
        box = trimesh.creation.box(extents=[x1, d, z0 - zb])
        box.apply_translation([x1 / 2, 0, (z0 + zb) / 2])
        return trimesh.boolean.union([c, box], engine='manifold')

    small = cutter(dia)
    out = trimesh.boolean.difference([mesh, small], engine='manifold')
    if border > 0 and prof is not None:
        ring = trimesh.boolean.difference([cutter(dia + 2 * border), small], engine='manifold')
        # The frame is a full-wall band, so it must be confined to the stretch of shell that is
        # actually full wall. On a split half the joint is a HALF-wall tongue or socket, and a
        # frame run through it fills the rebate solid - the halves then will not close. wall_s
        # is the lattice's own arc range, which is exactly the full-wall part.
        s0, s1 = wall_s if wall_s else (0.0, prof.S)
        wall = band(prof, max(0.0, s0), min(prof.S, s1), thick)
        frame = trimesh.boolean.intersection([ring, wall], engine='manifold')   # on the surface
        out = trimesh.boolean.union([out, frame], engine='manifold')
    return out

def gauge(prof, path, t=3.0, pad=14.0):
    """flat template: hold it against the bare chassis and look for light under the edge"""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    zs = np.linspace(prof.z.min(), prof.z.max(), 400)
    rs = np.interp(zs, prof.z, prof.r)
    body = Polygon([(0, zs[0])] + list(zip(rs, zs)) + [(0, zs[-1])])
    plate = box(0, zs[0], rs.max() + pad, zs[-1] + 6).difference(body.buffer(0.15))
    marks = unary_union([box(rs.max() + pad - 4, z - .8, rs.max() + pad + 1, z + .8) for z in (28.0, 133.0)])
    g = trimesh.creation.extrude_polygon(plate.difference(marks), t)
    g.export(path); return g

def straight_bottom(pts, z0):
    """drop the bottom roll: hold the diameter at z0 straight down to the bench"""
    z = np.array([a for a, b in pts]); d = np.array([b for a, b in pts])
    d0 = float(np.interp(z0, z, d)); k = z > z0
    return [(0.0, d0), (z0 * 0.5, d0)] + [(float(a), float(b)) for a, b in zip(z[k], d[k])]

def stretch(pts, mm, at=80.0):
    """make the shell taller by inserting height in the straight midsection, so the bottom
    roll and the top shoulder keep their exact curvature - unlike scaling, which bends both"""
    return [(float(z + mm) if z > at else float(z), float(d)) for z, d in pts]

def collar(pts, h):
    """straight ring above the shell top, to meet the underside of the cap"""
    z = np.array([a for a, b in pts])
    return list(pts) + [(float(z[-1] + h), pts[-1][1])]

def joint(prof, z_cut, L, thick, gap, outer):
    """half-wall rebate: the lower piece carries the inner half up past the cut, the upper
    piece carries the outer half down over it, so the two slide together and locate."""
    s0 = float(np.interp(z_cut, prof.z, prof.s)); s1 = float(np.interp(z_cut + L, prof.z, prof.s))
    shift = thick / 4 if outer else -thick / 4
    t = thick / 2 - (0 if outer else 2 * gap)
    v = np.linspace(s0, s1, max(2, int((s1 - s0) / 0.6) + 1))
    r, z, nr, nz = prof.at(v)
    return band_at(r + shift * nr, z + shift * nz, nr, nz, t)

def seam_trim(mesh, keep, z0, z1, keep_below):
    """Square the rebate off on true planes.

    Every band is swept along the surface NORMAL, so its end faces are slanted and overhang
    the split plane by a couple of tenths - a full-wall ring sitting proud of a half-wall
    tongue. That is interference: the halves bottom out on it before they close. Outside the
    joint the piece is untouched; inside it, whatever survives is clipped to the tongue (or
    socket) envelope, so the cord hole and anything else already cut stays cut."""
    b = mesh.bounds; pad = 10.0
    def slab(a, c):
        return trimesh.creation.box(bounds=[[b[0][0] - pad, b[0][1] - pad, a],
                                            [b[1][0] + pad, b[1][1] + pad, c]])
    outside = slab(b[0][2] - pad, z0) if keep_below else slab(z1, b[1][2] + pad)
    body = trimesh.boolean.intersection([mesh, outside], engine='manifold')
    j = trimesh.boolean.intersection([mesh, keep, slab(z0, z1)], engine='manifold')
    return trimesh.boolean.union([body, j], engine='manifold')

def report(prof, m, ang, args, vol, faces, wt):
    open_f = max(0.0, (1 - args.rib / args.pitch)) ** 2
    dz, dr = np.diff(prof.z), np.diff(prof.r)
    lean = np.degrees(np.arctan2(np.abs(dr), np.abs(dz)))
    up = lean[dr > 0].max() if (dr > 0).any() else 0.0
    dn = lean[dr < 0].max() if (dr < 0).any() else 0.0
    creep = args.layer / math.tan(math.radians(ang))
    print(f'  {2*m} ribs at {ang:.1f} deg, pitch {args.pitch}, rib {args.rib} x {args.thick} mm wall')
    print(f'  {2*prof.r.max():.1f} mm across x {prof.z.max()-prof.z.min():.1f} mm tall, '
          f'surface {prof.S:.1f} mm, circumference {2*math.pi*prof.Rref:.1f} mm')
    print(f'  ~{open_f*100:.0f}% open, {vol/1000:.1f} cm3 of filament, {faces//1000}k faces, watertight {wt}')
    print(f'  {2*m} islands per layer; rib creeps {creep:.3f} mm per {args.layer} mm layer '
          f'({"self-supporting" if creep < args.rib * 0.75 else "TOO SHALLOW - raise --angle"})')
    flare = dr / np.where(dz == 0, 1e-9, dz)
    creep = np.abs(flare[flare > 0]).max() * args.layer if (flare > 0).any() else 0.0
    print(f'  printed base-down: worst outward creep {creep:.3f} mm/layer against a {args.thick} mm wall '
          f'({"self-supporting" if creep < args.thick * 0.5 else "NEEDS SUPPORT"}); '
          f'shoulder leans in at {dn:.0f} deg, free')

def build(args, prof, tag, extra=None, rib_range=None, seam=None):
    parts, m, ang = ribs(prof, args.pitch, args.rib, args.thick, args.angle, rib_range=rib_range)
    ht = args.hem_thick or args.thick
    hb = args.hem_bottom if args.hem_bottom is not None else args.hem
    if hb > 0: parts.append(band(prof, 0, hb, ht))          # bottom cuff: also covers the steep roll
    z_flat = None
    if args.bottom_border > 0:
        # A solid border at the base: the first --bottom-border mm of arc made solid at full
        # wall thickness, taken OUT of the lattice rather than added below it. The cord slot is
        # cut afterwards and goes straight through, so the shell still slides on over a
        # plugged-in cord.
        bb = band(prof, 0, args.bottom_border, args.thick)
        parts.append(bb)
        # The swept rib tubes are capped square to the helix, not to the rim, so their ends
        # hang a few tenths BELOW the band - a ragged fringe under an otherwise solid border,
        # and a first layer of hundreds of unsupported islands. Cut the whole part off at the
        # band's own underside so the base is one flat annulus.
        z_flat = float(bb.bounds[0][2])
    if args.hem > 0:
        tt = args.hem_thick_top or ht
        s0 = prof.S - args.hem
        v = np.linspace(s0, prof.S, max(2, int((prof.S - s0) / 0.4) + 1))
        r, z, nr, nz = prof.at(v)
        sh = -(args.thick - tt) / 2
        parts.append(band_at(r + sh * nr, z + sh * nz, nr, nz, tt))
    if extra: parts += extra
    mesh = trimesh.util.concatenate(parts)
    wt = 'multi-body'
    if not args.no_union:
        try:
            u = trimesh.boolean.union(parts, engine='manifold'); mesh = u; wt = str(u.is_watertight)
        except Exception as e:
            print(f'  (union skipped: {type(e).__name__}: {e})')
    if z_flat is not None and not args.no_bottom_flat:
        try:
            lost = z_flat - float(mesh.bounds[0][2])
            b = mesh.bounds
            box = trimesh.creation.box(bounds=[[b[0][0] - 10, b[0][1] - 10, z_flat],
                                               [b[1][0] + 10, b[1][1] + 10, b[1][2] + 10]])
            mesh = trimesh.boolean.intersection([mesh, box], engine='manifold')
            print(f'  base cut flat on the border underside: {lost:.3f} mm of ragged rib ends removed')
        except Exception as e:
            print(f'  (base flatten skipped: {type(e).__name__}: {e})')
    if args.cord:
        try:
            mesh = cord_cut(mesh, args.cord, args.cord_z,
                            args.cord_slot and mesh.bounds[0][2] < args.cord_z - args.cord / 2,
                            border=args.cord_border, prof=prof, thick=args.thick,
                            wall_s=rib_range)
        except Exception as e:
            print(f'  (cord cut skipped: {type(e).__name__}: {e})')
    if seam is not None:
        try:
            keep, z0, z1, keep_below = seam
            mesh = seam_trim(mesh, keep, z0, z1, keep_below)
        except Exception as e:
            print(f'  (seam trim skipped: {type(e).__name__}: {e})')
    if args.top_z is not None:
        # the wall is built along the normal, which near the cap points almost straight up, so
        # the shell would stand proud of the pod's top surface. Cut it off flush there.
        # Done as a boolean against a box rather than slice_plane: capping a slice needs
        # mapbox_earcut, whereas manifold3d is already required for the union.
        b = mesh.bounds; pad = 10.0
        box = trimesh.creation.box(bounds=[[b[0][0] - pad, b[0][1] - pad, b[0][2] - pad],
                                           [b[1][0] + pad, b[1][1] + pad, args.top_z]])
        try:
            mesh = trimesh.boolean.intersection([mesh, box], engine='manifold')
        except Exception as e:
            try: mesh = mesh.slice_plane([0, 0, args.top_z], [0, 0, -1], cap=True)
            except Exception as e2: print(f'  (top cut skipped: {type(e).__name__} / {type(e2).__name__})')
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f'homepod_{tag}.stl'); mesh.export(p)
    try:
        sc = trimesh.Scene(); sc.add_geometry(mesh, geom_name=tag)
        sc.export(p.replace('.stl', '.3mf'))                       # a third the size, and Orca prefers it
    except Exception as e: print(f'  (3mf skipped: {type(e).__name__})')
    print(f'{tag}:'); report(prof, m, ang, args, mesh.volume, len(mesh.faces), wt); print(f'  -> {p}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', help='json list of [z, diameter] up the body (default: scaled card outline)')
    ap.add_argument('--dmax', type=float, default=136.6, help='max diameter of the bare chassis (measured)')
    ap.add_argument('--arcs', action='store_true', help='build the profile as two exact circular arcs on a straight barrel instead of tracing the card silhouette (removes the sampling wobble)')
    ap.add_argument('--z-bottom', type=float, default=28.0, help='--arcs: height where the bottom roll meets the barrel (measured)')
    ap.add_argument('--z-top', type=float, default=133.0, help='--arcs: height where the barrel starts curving in (measured)')
    ap.add_argument('--height', type=float, default=163.0, help='rubber base to the shell top surface')
    ap.add_argument('--d-bottom', type=float, default=86.0, help='diameter at the rubber base')
    ap.add_argument('--d-top', type=float, default=88.5, help='diameter at the cap slot')
    ap.add_argument('-o', '--out', default='out')
    ap.add_argument('--pitch', type=float, default=2.1, help='rib spacing, mm')
    ap.add_argument('--rib', type=float, default=0.65, help='rib width, mm')
    ap.add_argument('--thick', type=float, default=2.0, help='wall thickness, mm')
    ap.add_argument('--angle', type=float, default=52.0, help='rib angle from horizontal')
    ap.add_argument('--hem', type=float, default=4.0, help='solid cuff at each end, mm (0 = none)')
    ap.add_argument('--hem-bottom', type=float, help='bottom cuff, mm of arc (default: --hem). Raise it to make the steep bottom roll solid instead of lattice')
    ap.add_argument('--hem-thick', type=float, default=3.0, help='cuff wall, mm - the top one seats in the 4 mm slot')
    ap.add_argument('--band', type=float, nargs=2, metavar=('Z0','Z1'), help='only this height range, for a test print')
    ap.add_argument('--gauge', action='store_true', help='also write a flat profile template to check against the real part')
    ap.add_argument('--flare-blend', type=float, default=1.0, help='mm of fillet where the flare clamp meets the body (0 = the old hard corner)')
    ap.add_argument('--max-flare', type=float, help='cap the bottom roll at this angle off vertical (default: from wall/layer)')
    ap.add_argument('--fit', type=float, default=0.95, help='radial undersize for the stretch fit')
    ap.add_argument('--layer', type=float, default=0.2, help='layer height, for the printability check')
    ap.add_argument('--no-union', action='store_true', help='leave the ribs as separate bodies')
    ap.add_argument('--variants', action='store_true', help='write fine / medium / coarse in one go (pitch and rib only; --thick applies to all)')
    ap.add_argument('--gap', type=float, default=0.0, help='radial clearance on the bore, mm - for a rigid shell that slides on. Half the wall thickness is always added on top, so 0 puts the inner face exactly on the measured profile; go negative for an interference fit')
    ap.add_argument('--straight-bottom', type=float, metavar='Z', help='drop the bottom roll: hold the diameter from this height straight down to the bench')
    ap.add_argument('--collar', type=float, default=0.0, help='straight ring above the shell top, to meet the underside of the cap')
    ap.add_argument('--split', type=float, metavar='Z', help='cut into two pieces at this height with an interlocking rebate')
    ap.add_argument('--joint', type=float, default=4.0, help='rebate overlap length, mm')
    ap.add_argument('--slide', action='store_true', help='rigid shell: widen the bore to the running max so it can slide on from the top')
    ap.add_argument('--hem-thick-top', type=float, help='top band wall, mm (default: --hem-thick)')
    ap.add_argument('--cap-dia', type=float, default=88.5, help='plastic top cap diameter - the top opening')
    ap.add_argument('--cap-gap', type=float, default=0.4, help='clearance on the top opening, mm')
    ap.add_argument('--no-bottom-flat', action='store_true', help='leave the ragged rib ends hanging below the bottom border instead of cutting the base flat')
    ap.add_argument('--bottom-border', type=float, default=0.0, help='solid border at the base, mm of arc, full wall thickness. Taken out of the existing lattice so the overall height is unchanged; the cord slot still cuts through it. One-piece builds only')
    ap.add_argument('--cord', type=float, default=0.0, help='diameter of a power-cord hole through the wall, mm (0 = none)')
    ap.add_argument('--cord-z', type=float, default=28.5, help='height of the cord hole centre, mm')
    ap.add_argument('--cord-border', type=float, default=2.0, help='width of the solid frame around the cord opening, mm (0 = bare cut rib ends)')
    ap.add_argument('--cord-slot', action='store_true', help='open the cord hole downward to the bottom edge so the shell slides on without unplugging')
    ap.add_argument('--cap-trim', action='store_true', help='trim the top so the bare lattice edge ends on the plastic cap')
    ap.add_argument('--no-cap-trim', action='store_true', help='do not trim the top to the cap diameter')
    ap.add_argument('--top-z', type=float, help='trim the shell flat at this height, flush with the pod top surface')
    ap.add_argument('--stretch', type=float, default=0.0, help='add this much height in the straight midsection, curvature untouched')
    ap.add_argument('--joint-gap', type=float, default=0.15, help='clearance in the rebate, mm')
    a = ap.parse_args()

    jobs = [(f'p{a.pitch:g}', a.pitch, a.rib)] if not a.variants else \
           [('fine', 2.1, 0.65), ('mid-fine', 2.4, 0.72), ('medium', 3.5, 1.0), ('coarse', 5.0, 1.4)]
    for i, (tag, pitch, rib) in enumerate(jobs):
        a.pitch, a.rib = pitch, rib
        if a.profile:   pts = json.load(open(a.profile))
        elif a.arcs:    pts = arc_profile(a.dmax, a.height, a.d_bottom, a.d_top, a.z_bottom, a.z_top)
        else:           pts = card_profile(a.dmax, a.height, a.d_bottom, a.d_top)
        if a.arcs and not a.profile and i == 0:
            Rb, Rt = arc_profile.radii
            ex = (a.height - a.z_top) / math.sqrt(max(Rt ** 2 - (a.height - a.z_top) ** 2, 1e-9))
            print(f'arc profile: bottom roll R{Rb:.2f} (0..{a.z_bottom:g}), barrel {a.dmax:g} dia '
                  f'({a.z_bottom:g}..{a.z_top:g}), top shoulder R{Rt:.2f} ({a.z_top:g}..{a.height:g})')
            print(f'  top shoulder leaves at dr/dz {-ex:.2f} = {ex*a.layer:.2f} mm of bore per '
                  f'{a.layer} mm layer{"  <- lower --z-top to soften it" if ex*a.layer > 0.5 else ""}\n')
        if a.straight_bottom: pts = straight_bottom(pts, a.straight_bottom)
        if a.stretch: pts = stretch(pts, a.stretch)
        if a.collar: pts = collar(pts, a.collar)
        flare = a.max_flare if a.max_flare else np.degrees(np.arctan(0.25 * a.thick / a.layer))
        if i == 0:
            print(f'bottom roll clamped to {flare:.0f} deg off vertical '
                  f'(0.25 x {a.thick} mm wall per {a.layer} mm layer)\n')
            if a.gauge:
                os.makedirs(a.out, exist_ok=True)
                gauge(Profile(pts, 1.0, max_flare=flare, blend=a.flare_blend), os.path.join(a.out, 'homepod_profile_gauge.stl'))
                print(f'wrote {a.out}/homepod_profile_gauge.stl\n')
        if a.band: pts = Profile(pts, 1.0, max_flare=flare, blend=a.flare_blend).slice_z(*a.band)
        # --gap is clearance on the BORE. Half the wall is geometry, not clearance, so it is
        # ALWAYS added: --gap 0 puts the inner face exactly on the measured profile, and a
        # negative gap gives an interference fit (what a stretch-on TPU sock wants).
        off = a.gap + a.thick / 2
        prof = Profile(pts, a.fit, max_flare=flare, gap=off, slide=a.slide, blend=a.flare_blend)
        if a.cap_trim and not a.no_cap_trim:
            prof.trim_to((a.cap_dia + a.cap_gap) / 2, a.thick / 2)

        if a.split is None:
            build(a, prof, ('testband_' if a.band else '') + tag)
        else:
            zc, L = a.split, a.joint
            sc = float(np.interp(zc, prof.z, prof.s)); hem0, hem = a.hem, a.hem_bottom
            # the lower piece goes on from below, where the pod only widens, so it stays
            # conformal - but its top must swell to meet the upper's widened bore. Short taper.
            conf = Profile(pts, a.fit, max_flare=flare, gap=off, slide=False, blend=a.flare_blend)
            lp = np.array(conf.slice_z(conf.z.min(), zc + L))
            rj = float(np.interp(zc, prof.z, prof.r))
            ramp = np.where(lp[:, 0] > zc - 4, np.minimum(rj - 0.4 * (zc - lp[:, 0]), rj), 0.0)
            lp[:, 1] = 2 * np.maximum(lp[:, 1] / 2, ramp)      # only the top 4 mm, nothing below
            lo = Profile([(float(x), float(y)) for x, y in lp], 1.0)            # lower: roll + tongue
            a.hem, a.hem_bottom = 0.0, (a.hem_bottom if a.hem_bottom is not None else 4.0)
            tz, bb = a.top_z, a.bottom_border
            a.top_z, a.bottom_border = None, 0.0   # the border is a one-piece feature
            tongue = joint(prof, zc, L, a.thick, a.joint_gap, outer=False)
            build(a, lo, f'{tag}_lower', extra=[tongue, band(prof, max(0, sc - 3), sc, a.thick)],
                  rib_range=(0.0, float(np.interp(zc, lo.z, lo.s))),
                  seam=(tongue, zc, zc + L, True))
            a.top_z = tz
            up = Profile(prof.slice_z(zc, prof.z.max()), 1.0, slide=a.slide)
            if a.cap_trim and not a.no_cap_trim:
                up.trim_to((a.cap_dia + a.cap_gap) / 2, a.thick / 2)
            a.hem, a.hem_bottom = hem0, 0.0
            socket = joint(prof, zc, L, a.thick, a.joint_gap, outer=True)
            build(a, up, f'{tag}_upper', extra=[socket],
                  rib_range=(float(np.interp(zc + L, up.z, up.s)), up.S),
                  seam=(socket, zc, zc + L, False))
            a.hem, a.hem_bottom, a.bottom_border = hem0, hem, bb
        print()

if __name__ == '__main__':
    main()
