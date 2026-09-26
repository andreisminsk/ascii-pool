#!/usr/bin/env python3
"""ASCII Pool — 8-ball billiards in the terminal. Pure stdlib, zero deps.

Sister project to ascii-pong: same ethos — single file, continuous float
physics rendered to terminal cells, tuning constants at the top, fits any
POSIX terminal (and Termux on phones).

You vs a deterministic CPU (v1). The LLM opponent (v2) is sketched in
POOL-AI-ARCH.md.

Controls
  <-/-> or h/l or a/d  rotate aim        ^/v or j/k or w/s  power up/down
  , / .                fine aim          Space   shoot / place cue ball
  q / Esc              quit              r       rematch (at game end)

Balls: cue = O - solids = 1-7 (yellow) - 8 = 8 (magenta) -
stripes = A-G (A=9 ... G=15, cyan). Pocket your group, then the 8.

Run:  python3 ascii_pool.py [--error DEG] [--seed N] [--selftest]
"""
import argparse
import curses
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave

# ---------------------------------------------------------------- tuning
FRAME           = 0.03    # sim seconds per rendered frame
SUBSTEPS        = 8       # physics substeps per frame (anti-tunneling)
FRICTION_K      = 1.05    # v *= exp(-K*dt); a ball rolls ~ v0/K cells
STOP_V          = 1.5     # below this speed a ball halts
BALL_R          = 0.45    # ball radius (cells)
CUSHION         = 0.75    # cushion restitution
RAIL_FRICTION   = 0.97    # tangential damping on a rail bounce
CORNER_GAP      = 1.4     # no-bounce zone next to each corner pocket
SIDE_GAP        = 1.0     # half-width of the side pocket mouth
CORNER_POCKET_R = 1.15    # capture radius at the four corners
SIDE_POCKET_R   = 0.95    # capture radius at the two side pockets
WIDE_POCKET_W   = 100     # table width where pockets widen to 3 cells
POCKET_CELLS    = 1       # pocket mouth in cells (set_v_scale: 1 or 3)
SHOT_V_MIN      = 18.0    # launch speed at power 0 (at DESIGN_W)
SHOT_V_MAX      = 115.0   # launch speed at power 1 (at DESIGN_W)
DESIGN_W        = 76.0    # table width the constants are tuned for
V_SCALE         = 1.0     # live scale for wider tables (set_v_scale)
AI_ERROR_DEG    = 1.5     # CPU aim jitter (gauss sigma, degrees)
AI_POWER_MARGIN = 1.35    # CPU power headroom over the minimum
GUIDE_MAX       = 40       # aim guide length (cells)
COARSE          = 0.07    # aim step, radians (~4 degrees)
FINE            = 0.015    # fine aim step
AIM_COARSE_CELLS = 2.0    # coarse tap: contact-point slide, in cells
AIM_FINE_CELLS   = 0.4    # fine tap: contact-point slide, in cells
BAR_W           = 18      # power bar width
PRIZE_FILES     = ('diana.txt',)   # first found wins

DEFAULT_PRIZE = [
    "     ___________    ",
    "    '._==_==_=_'   ",
    "    .-\\:      /-.  ",
    "   | (|:.     |) |  ",
    "    '-|:.     |-'   ",
    "      \\::.    /     ",
    "       '::. .'      ",
    "         ) (        ",
    "        _.' '._     ",
    "       '-------'   ",
]

def set_v_scale(w):
    """Rebase speed/guide/substep/aim/pocket constants on DESIGN_W. A
    wider table needs proportionally faster shots (same pace, longer
    roll), a longer aim guide, finer substeps (anti-tunneling), finer
    aim steps (same precision at longer distances), and - once it is
    WIDE_POCKET_W wide - 3-cell pocket mouths. Game.__init__ calls it;
    constants recompute from literals, so repeated calls are safe."""
    global V_SCALE, SHOT_V_MIN, SHOT_V_MAX, GUIDE_MAX, SUBSTEPS
    global COARSE, FINE
    global CORNER_GAP, SIDE_GAP, CORNER_POCKET_R, SIDE_POCKET_R
    global POCKET_CELLS
    s = max(1.0, w / DESIGN_W)
    V_SCALE = s
    SHOT_V_MIN = 18.0 * s
    SHOT_V_MAX = 115.0 * s
    GUIDE_MAX = int(40 * s)
    SUBSTEPS = max(8, int(8 * s))
    COARSE = 0.07 / s
    FINE = 0.015 / s
    # pockets: a 1-cell mouth is too small a target on a large table -
    # widen to 3 cells (gaps and capture radii grow to match)
    if w >= WIDE_POCKET_W:
        POCKET_CELLS = 3
        CORNER_GAP = 1.5       # 3-cell corner mouth along each rail
        SIDE_GAP = 1.5         # 3-cell side mouth (half-width)
        CORNER_POCKET_R = 1.5
        SIDE_POCKET_R = 1.3
    else:
        POCKET_CELLS = 1
        CORNER_GAP = 1.4
        SIDE_GAP = 1.0
        CORNER_POCKET_R = 1.15
        SIDE_POCKET_R = 0.95


# ---------------------------------------------------------------- sounds
_TONES = {
    'cue':    [(900, 45)],
    'click':  [(1400, 25)],
    'rail':   [(300, 45)],
    'pocket': [(520, 70), (330, 80), (190, 110)],
}
_PLAYER_CANDIDATES = (
    ('termux-media-player', ('termux-media-player', 'play')),
    ('play', ('play',)),
    ('mpv', ('mpv',)),
    ('paplay', ('paplay',)),
    ('ffplay', ('ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet')),
    ('afplay', ('afplay',)),
)
_player = None
_tone_cache = {}


def _write_tone(path, seq):
    sr = 22050
    frames = bytearray()
    for hz, ms in seq:
        n = int(sr * ms / 1000)
        for i in range(n):
            v = int(9000 * math.sin(2 * math.pi * hz * i / sr) * (1 - i / n))
            frames += struct.pack('<h', v)
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def play_sound(kind):
    global _player
    if _player is None:
        _player = []
        for name, cmd in _PLAYER_CANDIDATES:
            if shutil.which(name):
                _player = list(cmd)
                break
    if not _player:
        return
    try:
        f = _tone_cache.get(kind)
        if f is None:
            f = os.path.join(tempfile.gettempdir(), 'asciipool_%s.wav' % kind)
            _write_tone(f, _TONES[kind])
            _tone_cache[kind] = f
        subprocess.Popen(_player + [f], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ---------------------------------------------------------------- model
def ball_char(n):
    if n == 0:
        return 'O'
    if n <= 8:
        return str(n)
    return 'ABCDEFGHI'[n - 9]


def group_of(n):
    return 'solid' if n < 8 else 'stripe'


def stick_char(sx, sy):
    if abs(sx) > 2.2 * abs(sy):
        return '-'
    if abs(sy) > 2.2 * abs(sx):
        return '|'
    return '/' if sx * sy < 0 else '\\'


def seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / l2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


class Ball:
    __slots__ = ('n', 'x', 'y', 'vx', 'vy', 'on')

    def __init__(self, n, x, y):
        self.n, self.x, self.y = n, x, y
        self.vx = self.vy = 0.0
        self.on = True


class Table:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.balls = []
        self.events = []
        self.pockets = [
            (0, 0, CORNER_POCKET_R), (w, 0, CORNER_POCKET_R),
            (0, h, CORNER_POCKET_R), (w, h, CORNER_POCKET_R),
            (w / 2, -0.35, SIDE_POCKET_R), (w / 2, h + 0.35, SIDE_POCKET_R),
        ]

    def cue(self):
        return self.balls[0]

    def moving(self):
        return any(b.on and (b.vx or b.vy) for b in self.balls)

    def rack(self):
        w, h = self.W, self.H
        self.balls = [Ball(0, w * 0.25, h / 2)]
        others = [n for n in range(1, 16) if n not in (1, 8)]
        random.shuffle(others)
        flat = [1, others[0], others[1], others[2], 8] + others[3:]
        # back corners of the rack must be one solid + one stripe
        def grp(n):
            return 0 if n < 8 else 1
        if grp(flat[10]) == grp(flat[14]):
            want = 1 - grp(flat[10])
            for i in range(15):
                if i in (0, 4, 10, 14):
                    continue
                if grp(flat[i]) == want:
                    flat[10], flat[i] = flat[i], flat[10]
                    break
        ax, idx = w * 0.72, 0
        for row in range(5):
            for j in range(row + 1):
                x = ax + row * 0.80
                y = h / 2 + (j - row / 2) * 0.92
                self.balls.append(Ball(flat[idx], x, y))
                idx += 1

    def step(self, dt):
        ev = self.events
        w, h, r = self.W, self.H, BALL_R
        decay = math.exp(-FRICTION_K * dt)
        for b in self.balls:
            if not b.on:
                continue
            b.x += b.vx * dt
            b.y += b.vy * dt
            b.vx *= decay
            b.vy *= decay
            if b.vx * b.vx + b.vy * b.vy < STOP_V * STOP_V:
                b.vx = b.vy = 0.0
        # ball-ball (equal mass, elastic: exchange normal components)
        R = 2 * r
        balls = self.balls
        for i in range(len(balls)):
            a = balls[i]
            if not a.on:
                continue
            for j in range(i + 1, len(balls)):
                c = balls[j]
                if not c.on:
                    continue
                dx, dy = c.x - a.x, c.y - a.y
                d2 = dx * dx + dy * dy
                if d2 >= R * R or d2 < 1e-9:
                    continue
                d = math.sqrt(d2)
                nx, ny = dx / d, dy / d
                ov = (R - d) / 2
                a.x -= nx * ov; a.y -= ny * ov
                c.x += nx * ov; c.y += ny * ov
                rvn = (a.vx - c.vx) * nx + (a.vy - c.vy) * ny
                if rvn > 0:
                    a.vx -= rvn * nx; a.vy -= rvn * ny
                    c.vx += rvn * nx; c.vy += rvn * ny
                    ev.append(('hit', a.n, c.n))
        # cushions (with gaps at the pockets) + pockets
        for b in self.balls:
            if not b.on:
                continue
            if b.x < r and b.vx < 0 and CORNER_GAP < b.y < h - CORNER_GAP:
                b.x = r; b.vx = -b.vx * CUSHION; b.vy *= RAIL_FRICTION
                ev.append(('rail',))
            if b.x > w - r and b.vx > 0 and CORNER_GAP < b.y < h - CORNER_GAP:
                b.x = w - r; b.vx = -b.vx * CUSHION; b.vy *= RAIL_FRICTION
                ev.append(('rail',))
            near_side = abs(b.x - w / 2) < SIDE_GAP
            in_corner = b.x < CORNER_GAP or b.x > w - CORNER_GAP
            if b.y < r and b.vy < 0 and not near_side and not in_corner:
                b.y = r; b.vy = -b.vy * CUSHION; b.vx *= RAIL_FRICTION
                ev.append(('rail',))
            if b.y > h - r and b.vy > 0 and not near_side and not in_corner:
                b.y = h - r; b.vy = -b.vy * CUSHION; b.vx *= RAIL_FRICTION
                ev.append(('rail',))
            for (px, py, pr) in self.pockets:
                if (b.x - px) ** 2 + (b.y - py) ** 2 < pr * pr:
                    b.on = False; b.vx = b.vy = 0.0
                    ev.append(('pocket', b.n))
                    break
            if b.on and (b.x < -0.9 or b.x > w + 0.9 or
                         b.y < -0.9 or b.y > h + 0.9):
                b.on = False; b.vx = b.vy = 0.0
                ev.append(('pocket', b.n))

    def ray_hit(self, px, py, dx, dy):
        """First ball hit by the cue travelling along (dx,dy), or cushion."""
        best_t, best = 1e9, None
        R = 2 * BALL_R
        for b in self.balls:
            if not b.on or b.n == 0:
                continue
            rx, ry = b.x - px, b.y - py
            proj = rx * dx + ry * dy
            if proj <= 0:
                continue
            d2 = rx * rx + ry * ry - proj * proj
            if d2 >= R * R:
                continue
            t = proj - math.sqrt(R * R - d2)
            if 0 < t < best_t:
                best_t, best = t, b
        tc = 1e9
        if dx > 1e-9:
            tc = min(tc, (self.W - BALL_R - px) / dx)
        elif dx < -1e-9:
            tc = min(tc, (BALL_R - px) / dx)
        if dy > 1e-9:
            tc = min(tc, (self.H - BALL_R - py) / dy)
        elif dy < -1e-9:
            tc = min(tc, (BALL_R - py) / dy)
        if tc < best_t:
            return tc, None
        return best_t, best

    def seg_blocked(self, x1, y1, x2, y2, exclude):
        clear = 2 * BALL_R * 0.97
        for b in self.balls:
            if not b.on or b.n in exclude:
                continue
            if seg_dist(b.x, b.y, x1, y1, x2, y2) < clear:
                return True
        return False


def best_shot(tbl, cx, cy, targets):
    """Best (ball, pocket) pair reachable from (cx,cy).
    Returns (score, angle, launch_v, ball) or None."""
    best = None
    R = 2 * BALL_R
    for b in targets:
        for (px, py, pr) in tbl.pockets:
            ddx, ddy = px - b.x, py - b.y
            dl = math.hypot(ddx, ddy)
            if dl < 1e-6:
                continue
            ox, oy = ddx / dl, ddy / dl
            gx, gy = b.x - ox * R, b.y - oy * R
            if not (BALL_R - 0.1 <= gx <= tbl.W - BALL_R + 0.1 and
                    BALL_R - 0.1 <= gy <= tbl.H - BALL_R + 0.1):
                continue
            ax, ay = gx - cx, gy - cy
            al = math.hypot(ax, ay)
            if al < 1e-6:
                continue
            ax, ay = ax / al, ay / al
            cut = math.degrees(math.acos(max(-1.0, min(1.0, ax * ox + ay * oy))))
            if cut > 80:
                continue
            if tbl.seg_blocked(cx, cy, gx, gy, {0}):
                continue
            if tbl.seg_blocked(b.x, b.y, px, py, {b.n}):
                continue
            score = 100.0 - 0.55 * (al + dl) - 0.8 * cut
            if best is None or score > best[0]:
                v = (al + dl) * FRICTION_K * AI_POWER_MARGIN + 12.0
                v = max(SHOT_V_MIN * 0.8, min(SHOT_V_MAX, v))
                best = (score, math.atan2(gy - cy, gx - cx), v, b)
    return best


def aim_delta(game, fine=False):
    """Adaptive aim step for one tap. Each tap slides the predicted
    contact point by a fixed number of cells (AIM_*_CELLS), so aiming
    precision at the target is constant however far the cue ball is.
    Sweeping with no ball in the ray, and very close targets, keep
    the full fixed step (COARSE/FINE)."""
    t = game.table
    cue = t.cue()
    th, ob = t.ray_hit(cue.x, cue.y, math.cos(game.angle),
                       math.sin(game.angle))
    fixed = FINE if fine else COARSE
    if ob is None:
        return fixed
    lat = AIM_FINE_CELLS if fine else AIM_COARSE_CELLS
    return min(fixed, lat / max(th, 1.0))


# ---------------------------------------------------------------- game
class Game:
    def __init__(self, w, h, ai_error):
        set_v_scale(w)
        self.table = Table(w, h)
        self.table.rack()
        self.ai_error = ai_error
        self.turn = random.randint(0, 1)
        self.next_break = 1 - self.turn
        self.state = 'aim' if self.turn == 0 else 'cpu'
        self.groups = [None, None]
        self.open = True
        self.break_shot = True
        self.angle = 0.0
        self.power = 0.5
        self.msg = 'You break - Space to shoot' if self.turn == 0 \
            else 'CPU breaks'
        self.winner = None
        self.prize_shown = False
        self.sim_start = 0
        self.own_before = 99
        self.cpu_phase = None
        self.cpu_shot = None
        self.cpu_show_t = 0.0
        self.cpu_place_needed = False
        self.place_x, self.place_y = w * 0.25, h / 2

    # -- helpers
    def own_left(self, p):
        g = self.groups[p]
        if g is None:
            return 99
        return sum(1 for b in self.table.balls
                   if b.on and b.n != 8 and group_of(b.n) == g)

    def legal_targets(self, p):
        t = self.table
        g = self.groups[p]
        if self.open or g is None:
            return [b for b in t.balls if b.on and b.n not in (0, 8)]
        if self.own_left(p) > 0:
            return [b for b in t.balls
                    if b.on and b.n != 8 and group_of(b.n) == g]
        return [b for b in t.balls if b.on and b.n == 8]

    def valid_place(self, x, y):
        t = self.table
        if not (BALL_R + 0.1 <= x <= t.W - BALL_R - 0.1 and
                BALL_R + 0.1 <= y <= t.H - BALL_R - 0.1):
            return False
        for b in t.balls:
            if not b.on or b.n == 0:
                continue
            if math.hypot(b.x - x, b.y - y) < 2 * BALL_R + 0.05:
                return False
        return True

    # -- actions
    def shoot(self, shooter):
        t = self.table
        cue = t.cue()
        if shooter == 0:
            ang, v = self.angle, SHOT_V_MIN + self.power * (SHOT_V_MAX - SHOT_V_MIN)
        else:
            ang, v = self.cpu_shot[0], self.cpu_shot[1]
        cue.vx = math.cos(ang) * v
        cue.vy = math.sin(ang) * v
        self.own_before = self.own_left(shooter) if self.groups[shooter] else 99
        self.sim_start = len(t.events)
        self.state = 'sim'
        play_sound('cue')

    def cpu_act(self):
        t = self.table
        cue = t.cue()
        if self.cpu_place_needed:
            self.cpu_place()
            self.cpu_place_needed = False
            self.cpu_phase = None
        if self.cpu_phase is None:
            if self.break_shot:
                apex = t.balls[1]
                ang = math.atan2(apex.y - cue.y, apex.x - cue.x)
                v = SHOT_V_MAX * 0.85
                self.cpu_shot = (ang, v, apex.n)
                self.msg = 'CPU breaks'
            else:
                targets = self.legal_targets(1)
                best = best_shot(t, cue.x, cue.y, targets)
                if best:
                    _, ang, v, b = best
                    ang += random.gauss(0.0, math.radians(self.ai_error))
                    v *= 1.0 + random.gauss(0.0, 0.04)
                    v = max(SHOT_V_MIN * 0.8, min(SHOT_V_MAX, v))
                    self.cpu_shot = (ang, v, b.n)
                    self.msg = 'CPU lines up on %s' % ball_char(b.n)
                else:
                    near = min(targets,
                               key=lambda b: math.hypot(b.x - cue.x, b.y - cue.y))
                    ang = math.atan2(near.y - cue.y, near.x - cue.x)
                    self.cpu_shot = (ang, 30.0 * V_SCALE, near.n)
                    self.msg = 'CPU plays safe'
            self.cpu_phase = 'show'
            self.cpu_show_t = time.time() + 1.1
        elif time.time() >= self.cpu_show_t:
            self.cpu_phase = None
            self.shoot(1)

    def cpu_place(self):
        t = self.table
        targets = self.legal_targets(1)
        best = None
        x = BALL_R + 0.6
        while x < t.W - BALL_R - 0.6:
            y = BALL_R + 0.6
            while y < t.H - BALL_R - 0.6:
                ok = True
                for b in t.balls:
                    if b.on and b.n != 0 and \
                            math.hypot(b.x - x, b.y - y) < 2 * BALL_R + 0.1:
                        ok = False
                        break
                if ok:
                    s = best_shot(t, x, y, targets)
                    score = s[0] if s else -100.0
                    if best is None or score > best[0]:
                        best = (score, x, y)
                y += 2.5
            x += 2.5
        if best:
            t.cue().x, t.cue().y = best[1], best[2]

    def resolve(self):
        t = self.table
        evs = t.events[self.sim_start:]
        first, pocketed = None, []
        for e in evs:
            if e[0] == 'hit' and first is None and (e[1] == 0 or e[2] == 0):
                first = e[2] if e[1] == 0 else e[1]
            elif e[0] == 'pocket':
                pocketed.append(e[1])
        shooter = self.turn
        scratch = 0 in pocketed
        # 8-ball on the break: re-rack, same breaker
        if self.break_shot and 8 in pocketed:
            t.rack()
            self.msg = '8-ball on the break - re-racking'
            self.state = 'aim' if shooter == 0 else 'cpu'
            self.sim_start = len(t.events)
            return
        foul = None
        if scratch:
            foul = 'scratch'
        elif first is None:
            foul = 'no contact'
        elif self.open or self.groups[shooter] is None:
            if first == 8:
                foul = 'hit the 8 first'
        elif self.own_before > 0:
            if group_of(first) != self.groups[shooter]:
                foul = 'wrong ball first'
        elif first != 8:
            foul = 'must hit the 8'
        # 8-ball decides the game
        if 8 in pocketed:
            if self.own_before == 0 and foul is None:
                self.winner = shooter
                self.msg = 'The 8-ball drops - rack won!'
            else:
                self.winner = 1 - shooter
                self.msg = '8-ball down illegally - rack lost'
            self.state = 'over'
            return
        # group assignment (open table, after the break, clean shot)
        if self.open and not self.break_shot and foul is None:
            for n in pocketed:
                if n != 8:
                    g = group_of(n)
                    self.groups[shooter] = g
                    self.groups[1 - shooter] = 'stripe' if g == 'solid' else 'solid'
                    self.open = False
                    self.msg = 'You are %s' % (
                        'SOLIDS' if g == 'solid' else 'STRIPES') \
                        if shooter == 0 else 'CPU takes %s' % (
                        'SOLIDS' if g == 'solid' else 'STRIPES')
                    break
        cont = False
        if foul is None:
            if self.open:
                cont = any(n != 8 for n in pocketed)
            else:
                g = self.groups[shooter]
                cont = any(n != 8 and group_of(n) == g for n in pocketed)
        self.break_shot = False
        if foul:
            self.msg = 'Foul: %s - ball in hand' % foul
            self.turn = 1 - shooter
            cue = t.cue()
            cue.on = True
            cue.vx = cue.vy = 0.0
            if self.turn == 0:
                self.state = 'place'
                self.place_x, self.place_y = t.W * 0.25, t.H / 2
                while not self.valid_place(self.place_x, self.place_y) \
                        and self.place_x < t.W - 1:
                    self.place_x += 1.0
            else:
                self.state = 'cpu'
                self.cpu_place_needed = True
                self.cpu_phase = None
        elif cont:
            self.msg = 'Nice - shoot again'
            self.state = 'aim' if shooter == 0 else 'cpu'
        else:
            self.turn = 1 - shooter
            self.msg = 'Turn passes'
            self.state = 'aim' if self.turn == 0 else 'cpu'

    def rematch(self):
        w, h = self.table.W, self.table.H
        self.table = Table(w, h)
        self.table.rack()
        self.turn = self.next_break
        self.next_break = 1 - self.turn
        self.state = 'aim' if self.turn == 0 else 'cpu'
        self.groups = [None, None]
        self.open = True
        self.break_shot = True
        self.angle = 0.0
        self.power = 0.5
        self.winner = None
        self.prize_shown = False
        self.sim_start = 0
        self.own_before = 99
        self.cpu_phase = None
        self.cpu_shot = None
        self.cpu_place_needed = False
        self.msg = 'You break' if self.turn == 0 else 'CPU breaks'


# ---------------------------------------------------------------- view
def put(s, y, x, ch, attr=0):
    try:
        s.addch(y, x, ch, attr)
    except curses.error:
        pass


def puts(s, y, x, txt, attr=0):
    try:
        s.addstr(y, x, txt, attr)
    except curses.error:
        pass


def init_colors():
    if not curses.has_colors():
        return {}
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        return {}
    m = {}

    def P(i, fg, bg=-1):
        try:
            curses.init_pair(i, fg, bg)
            m[i] = curses.color_pair(i)
        except curses.error:
            pass
    P(1, curses.COLOR_BLUE)
    P(2, curses.COLOR_WHITE)
    P(3, curses.COLOR_YELLOW)
    P(4, curses.COLOR_CYAN)
    P(5, curses.COLOR_MAGENTA)
    P(7, curses.COLOR_GREEN)
    P(8, curses.COLOR_RED)
    return m


def ball_attr(n, colors):
    if n == 0:
        return colors.get(2, 0) | curses.A_BOLD
    if n == 8:
        return colors.get(5, 0) | curses.A_BOLD
    if n < 8:
        return colors.get(3, 0)
    return colors.get(4, 0) | curses.A_BOLD


def group_str(game, p):
    g = game.groups[p]
    if g is None:
        return 'open'
    s = 'SOLIDS' if g == 'solid' else 'STRIPES'
    if game.own_left(p) == 0:
        s += ' -> 8!'
    return s


def draw(stdscr, game, colors, footer=''):
    t = game.table
    w, h = t.W, t.H
    stdscr.erase()
    rail = colors.get(1, 0)
    pok = rail | curses.A_BOLD
    mid = 1 + w // 2
    if POCKET_CELLS >= 3:
        # wide mouths: 3 cells per pocket; a corner mouth is an L
        # (3 cells along each adjoining rail)
        top_at = (0, 1, 2, mid - 1, mid, mid + 1, w - 1, w, w + 1)
        side_at = (1, 2, h - 1, h)
    else:
        top_at = (0, mid, w + 1)
        side_at = ()
    for x in range(w + 2):
        ch = '@' if x in top_at else '#'
        put(stdscr, 0, x, ch, pok if ch == '@' else rail)
        put(stdscr, h + 1, x, ch, pok if ch == '@' else rail)
    for y in range(1, h + 1):
        ch = '@' if y in side_at else '#'
        put(stdscr, y, 0, ch, pok if ch == '@' else rail)
        put(stdscr, y, w + 1, ch, pok if ch == '@' else rail)
    cue = t.cue()
    show_aim = game.state == 'aim' or \
        (game.state == 'cpu' and game.cpu_phase == 'show')
    if show_aim and cue.on:
        ang = game.angle if game.state == 'aim' else game.cpu_shot[0]
        dx, dy = math.cos(ang), math.sin(ang)
        dim = curses.A_DIM
        th, ob = t.ray_hit(cue.x, cue.y, dx, dy)
        for k in range(1, min(int(th), GUIDE_MAX) + 1):
            put(stdscr, 1 + int(round(cue.y + dy * k)),
                1 + int(round(cue.x + dx * k)), '.', dim)
        if ob is not None:
            gx, gy = cue.x + dx * th, cue.y + dy * th
            ox, oy = ob.x - gx, ob.y - gy
            ol = math.hypot(ox, oy)
            if ol > 1e-6:
                ox, oy = ox / ol, oy / ol
                for k in range(1, 7):
                    put(stdscr, 1 + int(round(ob.y + oy * k)),
                        1 + int(round(ob.x + ox * k)), '.', dim)
                tx = dx - (dx * ox + dy * oy) * ox
                ty = dy - (dx * ox + dy * oy) * oy
                tl = math.hypot(tx, ty)
                if tl > 0.2:
                    tx, ty = tx / tl, ty / tl
                    for k in (1, 2, 3):
                        put(stdscr, 1 + int(round(gy + ty * k * 0.8)),
                            1 + int(round(gx + tx * k * 0.8)), '.', dim)
        for k in (1.5, 2.2, 2.9, 3.6, 4.3, 5.0, 5.7):
            put(stdscr, 1 + int(round(cue.y - dy * k)),
                1 + int(round(cue.x - dx * k)),
                stick_char(-dx, -dy), rail | curses.A_BOLD)
    # balls
    occ = set()
    for b in t.balls:
        if not b.on:
            continue
        if game.state == 'place' and b.n == 0:
            continue
        row, col = 1 + int(round(b.y)), 1 + int(round(b.x))
        if (row, col) in occ:
            continue
        occ.add((row, col))
        put(stdscr, row, col, ball_char(b.n), ball_attr(b.n, colors))
    if game.state == 'place':
        ok = game.valid_place(game.place_x, game.place_y)
        put(stdscr, 1 + int(round(game.place_y)), 1 + int(round(game.place_x)),
            'O' if ok else 'X',
            (colors.get(2, 0) | curses.A_BOLD) if ok else colors.get(8, 0))
    # status lines
    y = h + 2
    m0 = '>' if game.turn == 0 else ' '
    m1 = '>' if game.turn == 1 else ' '

    def gattr(p, active):
        """Segment in the target group's ball color: solids yellow,
        stripes cyan. Bold = that player's turn."""
        g = game.groups[p]
        base = colors.get(3, 0) if g == 'solid' else colors.get(4, 0)
        return base | (curses.A_BOLD if active else 0)

    def idattr(p, active):
        """Player identity color before groups are assigned:
        You = white (the cue ball), CPU = red. Bold = your turn."""
        base = colors.get(2, 0) if p == 0 else colors.get(8, 0)
        return base | (curses.A_BOLD if active else 0)

    if game.groups[0] is None:
        puts(stdscr, y, 1, '%s You: ' % m0, idattr(0, m0 == '>'))
        puts(stdscr, y, 8, group_str(game, 0), curses.A_DIM)
        puts(stdscr, y, 20, '  %s CPU: ' % m1, idattr(1, m1 == '>'))
        puts(stdscr, y, 29, group_str(game, 1), curses.A_DIM)
    else:
        puts(stdscr, y, 1, '%s You: ' % m0, gattr(0, m0 == '>'))
        puts(stdscr, y, 8, group_str(game, 0), gattr(0, m0 == '>'))
        puts(stdscr, y, 20, '  %s CPU: ' % m1, gattr(1, m1 == '>'))
        puts(stdscr, y, 29, group_str(game, 1), gattr(1, m1 == '>'))
    if game.open:
        puts(stdscr, y, 43, '[OPEN TABLE]', curses.A_BOLD)
    x = 1
    puts(stdscr, y + 1, x, 'Potted:')
    x += 8
    for b in t.balls:
        if b.on or b.n == 0:
            continue
        put(stdscr, y + 1, x, ball_char(b.n), ball_attr(b.n, colors))
        x += 2
    if game.state == 'over':
        win = game.winner == 0
        puts(stdscr, y + 2, 2, '*** YOU WIN THE RACK ***' if win
             else '*** CPU WINS THE RACK ***',
             (colors.get(7, 0) | curses.A_BOLD) if win
             else (colors.get(8, 0) | curses.A_BOLD))
        puts(stdscr, y + 3, 2, game.msg)
        puts(stdscr, y + 4, 1, '[r] rematch   [q] quit')
    else:
        if game.state == 'aim':
            deg = math.degrees(game.angle) % 360
            filled = int(game.power * BAR_W + 0.5)
            puts(stdscr, y + 2, 1, 'Aim %5.1f deg   Power [%s] %3d%%' %
                 (deg, '#' * filled + '-' * (BAR_W - filled),
                  int(game.power * 100)))
        elif game.state == 'place':
            puts(stdscr, y + 2, 1,
                 'Place the cue ball: arrows move, Space to drop')
        elif game.state == 'sim':
            puts(stdscr, y + 2, 1, '... rolling')
        elif game.state == 'cpu':
            if game.cpu_phase == 'show' and game.cpu_shot:
                v = game.cpu_shot[1]
                p = max(0.0, min(1.0, (v - SHOT_V_MIN) /
                                 (SHOT_V_MAX - SHOT_V_MIN)))
                filled = int(p * BAR_W + 0.5)
                puts(stdscr, y + 2, 1, 'CPU aiming   Power [%s] %3d%%' %
                     ('#' * filled + '-' * (BAR_W - filled), int(p * 100)))
            else:
                puts(stdscr, y + 2, 1, 'CPU is thinking...')
        attr = colors.get(8, 0) if game.msg.startswith('Foul') else 0
        puts(stdscr, y + 3, 1, game.msg, attr)
        puts(stdscr, y + 4, 1,
             'h/l aim  ,/. fine  j/k power  Space shoot  q quit')
    if footer:
        puts(stdscr, h + 7, 1, footer[:w])
    stdscr.refresh()


def run_sim(stdscr, game, colors, footer=''):
    t = game.table
    frames = 0
    while t.moving():
        for _ in range(SUBSTEPS):
            t.step(FRAME / SUBSTEPS)
        frames += 1
        kinds = set()
        for e in t.events[game.sim_start:]:
            if e[0] == 'hit':
                kinds.add('click')
            elif e[0] == 'rail':
                kinds.add('rail')
            elif e[0] == 'pocket':
                kinds.add('pocket')
        for k in kinds:
            play_sound(k)
        draw(stdscr, game, colors, footer)
        if stdscr.getch() == ord('q'):
            return 'quit'
        time.sleep(FRAME)
        if frames > 3000:
            break
    return None


def prize_lines():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in PRIZE_FILES:
        path = p if os.path.isabs(p) else os.path.join(here, p)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return [ln.rstrip() for ln in f]
            except OSError:
                pass
    return DEFAULT_PRIZE


def show_prize(stdscr):
    curses.endwin()
    cols = shutil.get_terminal_size().columns
    for ln in prize_lines():
        print(ln[:cols])
        sys.stdout.flush()
        time.sleep(0.15)
    print('You win the rack!')
    stdscr.refresh()


# ---------------------------------------------------------------- main
OPTS = {'error': AI_ERROR_DEG, 'seed': None, 'selftest': False}


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    colors = init_colors()
    scr_h, scr_w = stdscr.getmaxyx()
    h = max(10, min(scr_h - 8, 30))
    # cap width at 4x height: a real table's 2:1 proportions (a cell
    # is ~1:2) - a wide terminal gives a bigger table, not a distorted
    # one, and long shots stay aimable
    w = max(30, min(scr_w - 2, 4 * h))
    game = Game(w, h, OPTS['error'])
    while True:
        draw(stdscr, game, colors)
        k = stdscr.getch()
        if k in (ord('q'), 27):
            return
        st = game.state
        if st == 'aim':
            if k in (curses.KEY_LEFT, ord('h'), ord('a')):
                game.angle -= aim_delta(game)
            elif k in (curses.KEY_RIGHT, ord('l'), ord('d')):
                game.angle += aim_delta(game)
            elif k == ord(','):
                game.angle -= aim_delta(game, True)
            elif k == ord('.'):
                game.angle += aim_delta(game, True)
            elif k in (curses.KEY_UP, ord('k'), ord('w')):
                game.power = min(1.0, game.power + 0.05)
            elif k in (curses.KEY_DOWN, ord('j'), ord('s')):
                game.power = max(0.0, game.power - 0.05)
            elif k == ord(' '):
                game.shoot(0)
            time.sleep(0.02)
        elif st == 'place':
            step = 0.8
            if k in (curses.KEY_LEFT, ord('h'), ord('a')):
                game.place_x = max(BALL_R, game.place_x - step)
            elif k in (curses.KEY_RIGHT, ord('l'), ord('d')):
                game.place_x = min(game.table.W - BALL_R, game.place_x + step)
            elif k in (curses.KEY_UP, ord('k'), ord('w')):
                game.place_y = max(BALL_R, game.place_y - step)
            elif k in (curses.KEY_DOWN, ord('j'), ord('s')):
                game.place_y = min(game.table.H - BALL_R, game.place_y + step)
            elif k == ord(','):
                game.place_x = max(BALL_R, game.place_x - 0.2)
            elif k == ord('.'):
                game.place_x = min(game.table.W - BALL_R, game.place_x + 0.2)
            elif k == ord(' '):
                if game.valid_place(game.place_x, game.place_y):
                    cue = game.table.cue()
                    cue.x, cue.y = game.place_x, game.place_y
                    game.state = 'aim'
                    game.msg = 'Ball in hand placed - shoot'
            time.sleep(0.02)
        elif st == 'cpu':
            game.cpu_act()
            time.sleep(0.02)
        elif st == 'sim':
            if run_sim(stdscr, game, colors) == 'quit':
                return
            game.resolve()
        elif st == 'over':
            if not game.prize_shown and game.winner == 0:
                game.prize_shown = True
                show_prize(stdscr)
            if k == ord('r'):
                game.rematch()
            time.sleep(0.02)


def selftest():
    random.seed(OPTS['seed'])
    t = Table(60, 20)
    t.rack()
    t.cue().vx = 110.0
    n = 0
    while t.moving() and n < 20000:
        t.step(1 / 240.0)
        n += 1
    potted = [b.n for b in t.balls if not b.on]
    for b in t.balls:
        if b.on:
            assert BALL_R - 0.05 <= b.x <= t.W - BALL_R + 0.05, \
                'ball %d out of bounds x=%.2f' % (b.n, b.x)
            assert BALL_R - 0.05 <= b.y <= t.H - BALL_R + 0.05, \
                'ball %d out of bounds y=%.2f' % (b.n, b.y)
    print('selftest: %d steps, potted=%s' % (n, potted))
    print('OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='ASCII 8-ball pool vs a CPU')
    ap.add_argument('--error', type=float, default=AI_ERROR_DEG,
                    metavar='DEG', help='CPU aim jitter in degrees '
                    '(difficulty; default %(default)s)')
    ap.add_argument('--seed', type=int, default=None, metavar='N',
                    help='random seed')
    ap.add_argument('--selftest', action='store_true',
                    help='headless physics check, no curses')
    args = ap.parse_args()
    OPTS['error'] = args.error
    OPTS['seed'] = args.seed
    if args.selftest:
        selftest()
    else:
        if args.seed is not None:
            random.seed(args.seed)
        try:
            curses.wrapper(main)
        except KeyboardInterrupt:
            pass
