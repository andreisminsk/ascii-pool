#!/usr/bin/env python3
"""ASCII Pool v2 — the CPU's shots are chosen by an Ollama LLM.

Architecture (see POOL-AI-ARCH.md): hybrid intelligence.
- Physics enumerates every feasible shot: for each (legal ball, pocket)
  pair — ghost-ball contact point, path clearance, cut angle, distance
  → a scored candidate list (enumerate_shots, the guts of v1's
  best_shot).
- The LLM picks among the top-N candidates and tweaks power: "which
  ball, which pocket, soft or firm?" — a judgment call in the
  small-model sweet spot: few-shot, key=value state in, one tiny JSON
  object out.
- Calls run on daemon threads; the game loop never blocks. Silence,
  malformed output, or outages degrade to the v1 heuristic pick (the
  highest-scored candidate); a circuit breaker disables the LLM after
  repeated failures and retries it later. No feasible shot at all →
  v1's soft safety, as before.

Run:  python3 ascii_pool_llm.py [--model MODEL] [--timeout SECS]
      python3 ascii_pool_llm.py --sim 3          # headless smoke test
Env:  OLLAMA_URL   (default http://localhost:11434)
      POOL_MODEL   (default gemma3:270m; --model overrides)
      POOL_DEBUG=1 -> log prompts/responses to pool_llm_debug.jsonl

Before each session a pre-flight probe checks the model (server
reachable, model listed, answers parseable JSON within the timeout)
and asks how to proceed if something is wrong: play anyway, CPU
fallback, or pick another model from the server.
"""
import argparse
import curses
import difflib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ascii_pool as ap

# ---------------------------------------------------------------- knobs
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
LLM_MODEL = os.environ.get('POOL_MODEL', 'gemma3:270m')
LLM_TIMEOUT = 5.0        # s per shot — pool waits for the answer
LLM_TOPN = 5             # candidates offered to the LLM
LLM_BLEND = 1.0          # 0 = heuristic pick, 1 = full LLM
LLM_BREAKER = 3          # consecutive failures before fallback-only
LLM_REARM = 20.0         # s of fallback before the breaker retries
LLM_NUM_PREDICT = 48     # tiny answers = fast
LLM_TEMP = 0.4           # variety: the interesting axis is taste
DEBUG_LOG = 'pool_llm_debug.jsonl'

OPTS = {'error': ap.AI_ERROR_DEG}

POCKET_NAMES = ('TL', 'TR', 'BL', 'BR', 'TM', 'BM')  # Table.pockets order


def dbg(**rec):
    """Append a record to the debug log when POOL_DEBUG is set."""
    if not os.environ.get('POOL_DEBUG'):
        return
    rec['ts'] = round(time.time(), 3)
    try:
        with open(DEBUG_LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    except OSError:
        pass


# ------------------------------------------------- deterministic half
def enumerate_shots(tbl, cx, cy, targets):
    """All feasible (ball, pocket) shots from (cx, cy), best first.
    The same geometry as v1's best_shot, but the whole list."""
    out = []
    R = 2 * ap.BALL_R
    for b in targets:
        for pi, (px, py, pr) in enumerate(tbl.pockets):
            ddx, ddy = px - b.x, py - b.y
            dl = math.hypot(ddx, ddy)
            if dl < 1e-6:
                continue
            ox, oy = ddx / dl, ddy / dl
            gx, gy = b.x - ox * R, b.y - oy * R
            if not (ap.BALL_R - 0.1 <= gx <= tbl.W - ap.BALL_R + 0.1 and
                    ap.BALL_R - 0.1 <= gy <= tbl.H - ap.BALL_R + 0.1):
                continue
            ax, ay = gx - cx, gy - cy
            al = math.hypot(ax, ay)
            if al < 1e-6:
                continue
            ax, ay = ax / al, ay / al
            cut = math.degrees(math.acos(
                max(-1.0, min(1.0, ax * ox + ay * oy))))
            if cut > 80:
                continue
            if tbl.seg_blocked(cx, cy, gx, gy, {0}):
                continue
            if tbl.seg_blocked(b.x, b.y, px, py, {b.n}):
                continue
            score = 100.0 - 0.55 * (al + dl) - 0.8 * cut
            v = (al + dl) * ap.FRICTION_K * ap.AI_POWER_MARGIN + 12.0
            v = max(ap.SHOT_V_MIN * 0.8, min(ap.SHOT_V_MAX, v))
            out.append({'n': b.n, 'ch': ap.ball_char(b.n),
                        'pocket': POCKET_NAMES[pi], 'cut': cut,
                        'dist': al + dl, 'score': score,
                        'angle': math.atan2(gy - cy, gx - cx), 'v': v})
    out.sort(key=lambda c: c['score'], reverse=True)
    return out


# ---------------------------------------------------------- LLM half
SYSTEM = (
    "You are a pool shark choosing a shot in 8-ball. You get numbered "
    "candidate shots; score = difficulty (higher = easier). Reply with "
    'ONLY one JSON object, nothing else: {"pick": <int>, "power": <float>}\n'
    "pick: the candidate number you shoot (1..N).\n"
    "power: 0.0 soft .. 1.0 firm, 0.5 = normal.\n"
    "Tactics: usually take the highest score. Prefer short straight shots "
    "with soft power to hold position for the next ball; firm power for "
    "long distances or thin cuts. Sink the eight cleanly when it is your "
    "target."
)

FEWSHOT = """Example 1:
you=SOLIDS balls=2,5,7 opp=A,C eight=-
shots:
1) ball=5 pocket=TL cut=12 dist=18 score=87
2) ball=2 pocket=BR cut=41 dist=26 score=61
3) ball=7 pocket=TM cut=63 dist=31 score=34
{"pick": 1, "power": 0.3}

Example 2:
you=STRIPES balls=A,C,F opp=3,5 eight=-
shots:
1) ball=A pocket=TR cut=38 dist=34 score=71
2) ball=C pocket=BL cut=9 dist=15 score=66
3) ball=F pocket=BM cut=33 dist=29 score=52
{"pick": 2, "power": 0.3}

Example 3:
you=SOLIDS balls=- opp=D,G eight=8
shots:
1) ball=8 pocket=BL cut=24 dist=20 score=74
2) ball=8 pocket=TM cut=52 dist=33 score=38
{"pick": 1, "power": 0.8}

Now:"""


def game_state(game):
    """Compact key=value snapshot of the CPU's situation."""
    t = game.table
    g = game.groups[1]
    eight = '8' if game.own_left(1) == 0 else '-'
    if g is None:
        return 'open', 'any', '-', eight
    og = 'stripe' if g == 'solid' else 'solid'
    own = ','.join(ap.ball_char(b.n) for b in t.balls
                  if b.on and b.n not in (0, 8)
                  and ap.group_of(b.n) == g) or '-'
    opp = ','.join(ap.ball_char(b.n) for b in t.balls
                   if b.on and b.n not in (0, 8)
                   and ap.group_of(b.n) == og) or '-'
    return ('SOLIDS' if g == 'solid' else 'STRIPES'), own, opp, eight


def build_prompt(state, cands):
    you, own, opp, eight = state
    lines = ['you=%s balls=%s opp=%s eight=%s' % (you, own, opp, eight),
             'shots:']
    for i, c in enumerate(cands, 1):
        lines.append('%d) ball=%s pocket=%s cut=%d dist=%d score=%d'
                     % (i, c['ch'], c['pocket'], c['cut'], c['dist'],
                        c['score']))
    return FEWSHOT + '\n' + '\n'.join(lines)


def _num(x):
    """float(x) for numbers and numeric strings; None otherwise."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip().rstrip('.'))
        except ValueError:
            return None
    return None


def parse_llm(text, n):
    """Extract {pick, power} from a response; clamp to sane ranges.
    Returns (pick, power) or None if unusable."""
    s, e = text.find('{'), text.rfind('}')
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except (ValueError, TypeError):
        return None
    pick = _num(obj.get('pick'))
    power = _num(obj.get('power'))
    if pick is None:
        return None
    if power is None:
        power = 0.5
    return max(1, min(n, int(pick))), max(0.0, min(1.0, power))


class AsyncAdvisor:
    """One-shot LLM calls from daemon threads; the game loop never
    blocks. Latest result wins; a circuit breaker falls back after
    LLM_BREAKER consecutive failures."""

    def __init__(self):
        self.url = OLLAMA_URL.rstrip('/') + '/api/generate'
        self.enabled = True
        self.user_disabled = False   # probe menu choice - stays off
        self.tripped_at = None       # when the breaker last tripped
        self.failures = 0
        self.seq = 0
        self.result = None        # (seq, pick, power, latency_ms)
        self._delivered = 0
        self._lock = threading.Lock()

    def submit(self, prompt, n):
        """Fire a request; returns its sequence number, or None if
        disabled."""
        with self._lock:
            if not self.enabled:
                return None
            self.seq += 1
            seq = self.seq
        threading.Thread(target=self._worker, args=(seq, prompt, n),
                         daemon=True).start()
        return seq

    def _worker(self, seq, prompt, n):
        payload = json.dumps({
            'model': LLM_MODEL,
            'prompt': prompt,
            'system': SYSTEM,
            'stream': False,
            'keep_alive': '5m',
            'options': {'temperature': LLM_TEMP,
                        'num_predict': LLM_NUM_PREDICT},
        }).encode()
        t0 = time.time()
        try:
            req = urllib.request.Request(
                self.url, data=payload,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
                out = json.loads(r.read()).get('response', '')
            parsed = parse_llm(out, n)
            if parsed is None:
                raise ValueError('unparseable: %r' % out[:80])
            ms = (time.time() - t0) * 1000
            with self._lock:
                self.failures = 0
                self.result = (seq, parsed[0], parsed[1], ms)
            dbg(seq=seq, ms=round(ms), ok=True, prompt=prompt, response=out)
        except Exception as e:
            with self._lock:
                self.failures += 1
                if self.failures >= LLM_BREAKER and self.enabled:
                    self.enabled = False
                    self.tripped_at = time.time()
            dbg(seq=seq, ok=False, error=str(e))

    def collect(self):
        """Return the newest undelivered result, if any."""
        with self._lock:
            if self.result and self.result[0] > self._delivered:
                self._delivered = self.result[0]
                return self.result
        return None

    def rearm(self):
        """Fresh chance for the LLM (new rack). Honors user_disabled."""
        with self._lock:
            self.failures = 0
            self.tripped_at = None
            if not self.user_disabled:
                self.enabled = True

    def maybe_rearm(self):
        """Mid-rack: retry the LLM after LLM_REARM seconds of fallback."""
        with self._lock:
            if (not self.enabled and not self.user_disabled
                    and self.tripped_at is not None
                    and time.time() - self.tripped_at >= LLM_REARM):
                self.enabled = True
                self.failures = 0
                self.tripped_at = None


ADVISOR = AsyncAdvisor()


# ---------------------------------------------------------------- game
class LLMGame(ap.Game):
    """v1 Game with the CPU's choice delegated to the LLM.
    Fallback ladder: LLM pick -> v1 heuristic (top candidate) -> safety."""

    def __init__(self, w, h, ai_error):
        super().__init__(w, h, ai_error)
        self.cands = None       # candidates offered this turn
        self.llm_seq = None     # pending advisor request
        self.think_t0 = 0.0
        self.last_ms = None     # latency of the last applied answer
        self.llm_used = None    # True/False once a shot was chosen

    def cpu_act(self):
        t = self.table
        cue = t.cue()
        ADVISOR.maybe_rearm()
        if self.cpu_place_needed:
            self.cpu_place()
            self.cpu_place_needed = False
            self.cpu_phase = None
        if self.cpu_phase is None:
            self.llm_used = None
            if self.break_shot:
                apex = t.balls[1]
                ang = math.atan2(apex.y - cue.y, apex.x - cue.x)
                self.cpu_shot = (ang, ap.SHOT_V_MAX * 0.85, apex.n)
                self.msg = 'CPU breaks'
                self.cpu_phase = 'show'
                self.cpu_show_t = time.time() + 1.1
                return
            targets = self.legal_targets(1)
            self.cands = enumerate_shots(t, cue.x, cue.y, targets)
            if not self.cands:
                near = min(targets,
                           key=lambda b: math.hypot(b.x - cue.x,
                                                    b.y - cue.y))
                self.cpu_shot = (math.atan2(near.y - cue.y,
                                            near.x - cue.x),
                                 30.0 * ap.V_SCALE, near.n)
                self.msg = 'CPU plays safe'
                self.cpu_phase = 'show'
                self.cpu_show_t = time.time() + 1.1
            elif LLM_BLEND <= 0 or not ADVISOR.enabled:
                self._heuristic_shot()
            else:
                self.cands = self.cands[:LLM_TOPN]
                self.cpu_phase = 'think'
                self.think_t0 = time.time()
                self.msg = 'LLM is thinking...'
                self.llm_seq = ADVISOR.submit(
                    build_prompt(game_state(self), self.cands),
                    len(self.cands))
                if self.llm_seq is None:
                    self._heuristic_shot()
        elif self.cpu_phase == 'think':
            res = ADVISOR.collect()
            if res is not None and res[0] == self.llm_seq:
                self._llm_shot(*res[1:])
            elif time.time() - self.think_t0 > LLM_TIMEOUT + 0.5:
                self._heuristic_shot()   # silence: worker counted the fail
        elif time.time() >= self.cpu_show_t:
            self.cpu_phase = None
            self.shoot(1)

    def _llm_shot(self, pick, power, ms):
        c = self.cands[pick - 1]
        ang = c['angle'] + random.gauss(0.0, math.radians(self.ai_error))
        v = c['v'] * (0.7 + 0.6 * power) * (1.0 + random.gauss(0.0, 0.04))
        v = max(ap.SHOT_V_MIN * 0.8, min(ap.SHOT_V_MAX, v))
        self.cpu_shot = (ang, v, c['n'])
        self.last_ms = ms
        self.llm_used = True
        self.msg = 'LLM: %s -> %s, power %.2f (%d ms)' % (
            c['ch'], c['pocket'], power, ms)
        self.cpu_phase = 'show'
        self.cpu_show_t = time.time() + 1.1

    def _heuristic_shot(self):
        c = self.cands[0]
        ang = c['angle'] + random.gauss(0.0, math.radians(self.ai_error))
        v = c['v'] * (1.0 + random.gauss(0.0, 0.04))
        v = max(ap.SHOT_V_MIN * 0.8, min(ap.SHOT_V_MAX, v))
        self.cpu_shot = (ang, v, c['n'])
        self.llm_used = False
        self.msg = 'CPU lines up on %s' % c['ch']
        self.cpu_phase = 'show'
        self.cpu_show_t = time.time() + 1.1

    def rematch(self):
        ADVISOR.rearm()          # new rack: fresh chance for the LLM
        super().rematch()


# ------------------------------------------------------ pre-flight
SAMPLE_STATE = ('SOLIDS', '2,5,7', 'A,C', '-')
SAMPLE_CANDS = [
    {'n': 5, 'ch': '5', 'pocket': 'TL', 'cut': 12.0, 'dist': 18.0,
     'score': 87.0, 'angle': 0.3, 'v': 40.0},
    {'n': 2, 'ch': '2', 'pocket': 'BR', 'cut': 41.0, 'dist': 26.0,
     'score': 61.0, 'angle': 0.8, 'v': 52.0},
    {'n': 7, 'ch': '7', 'pocket': 'TM', 'cut': 63.0, 'dist': 31.0,
     'score': 34.0, 'angle': 1.2, 'v': 60.0},
]


def sample_prompt():
    """A representative shot-choice prompt, used by the pre-flight
    probe."""
    return build_prompt(SAMPLE_STATE, SAMPLE_CANDS)


def probe_model():
    """Pre-flight diagnostic of LLM_MODEL. Returns a report dict:
      reachable, listed, names, ping_ms,
      calls: [{ms, ok, done, raw}] - two live calls with the game prompt,
      problems: human-readable issues (empty = model is game-ready),
      ok: True when problems is empty."""
    rep = {'reachable': False, 'listed': False, 'names': [],
           'ping_ms': None, 'calls': [], 'problems': []}
    try:
        t0 = time.time()
        with urllib.request.urlopen(OLLAMA_URL.rstrip('/') + '/api/tags',
                                    timeout=3) as r:
            rep['names'] = [m.get('name', '')
                            for m in json.loads(r.read()).get('models', [])]
        rep['ping_ms'] = (time.time() - t0) * 1000
        rep['reachable'] = True
    except Exception:
        rep['problems'].append('server unreachable at ' + OLLAMA_URL)
        return rep
    rep['listed'] = any(n == LLM_MODEL or n.split(':')[0] ==
                        LLM_MODEL.split(':')[0] for n in rep['names'])
    if not rep['listed']:
        near = difflib.get_close_matches(LLM_MODEL, rep['names'],
                                         n=3, cutoff=0.3)
        msg = 'model not on server'
        if near:
            msg += ' - did you mean: ' + ', '.join(near)
        rep['problems'].append(msg)
        return rep
    # Two live calls with the real game prompt (the first may be cold).
    url = OLLAMA_URL.rstrip('/') + '/api/generate'
    for _ in range(2):
        call = {'ms': None, 'ok': False, 'done': '', 'raw': ''}
        try:
            payload = json.dumps({
                'model': LLM_MODEL, 'prompt': sample_prompt(),
                'system': SYSTEM, 'stream': False, 'keep_alive': '5m',
                'options': {'temperature': LLM_TEMP,
                            'num_predict': LLM_NUM_PREDICT},
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={'Content-Type': 'application/json'})
            t0 = time.time()
            with urllib.request.urlopen(
                    req, timeout=max(LLM_TIMEOUT * 4, 10)) as r:
                body = json.loads(r.read())
            call['ms'] = (time.time() - t0) * 1000
            call['raw'] = body.get('response', '')
            call['done'] = str(body.get('done_reason', ''))
            call['ok'] = parse_llm(call['raw'], len(SAMPLE_CANDS)) is not None
        except Exception as e:
            call['done'] = type(e).__name__
        rep['calls'].append(call)
    good = [c for c in rep['calls'] if c['ok']]
    if not good:
        if any(c['done'] == 'length' and not c['raw']
               for c in rep['calls']):
            rep['problems'].append(
                'reasoning model: thinks past its token budget and never '
                'answers (done_reason=length, empty response) - pick a '
                'non-reasoning model')
        elif any(c['done'] in ('TimeoutError', 'HTTPError', 'URLError')
                 for c in rep['calls']):
            rep['problems'].append(
                'calls failed (timeout/server error) - model or server '
                'too slow for the probe budget')
        else:
            last = rep['calls'][-1]
            rep['problems'].append(
                'output not parseable as {"pick", "power"}: '
                + repr((last['raw'] or last['done'])[:60]))
    else:
        ms = good[-1]['ms']
        if ms > LLM_TIMEOUT * 1000:
            rep['problems'].append(
                f'answers take ~{ms:.0f} ms, longer than --timeout '
                f'{LLM_TIMEOUT:.1f}s - most shots will fall back to the '
                'heuristic CPU; raise --timeout')
    rep['ok'] = not rep['problems']
    return rep


def pick_model(stdscr, names):
    """Interactive model picker (numbered, paginated, terminal-aware).
    Returns the chosen model name, or None if cancelled."""
    rows, cols = stdscr.getmaxyx()
    body = max(1, rows - 4)          # rows available for the list
    ncols = max(1, cols // 26)       # model names per row
    per_page = body * ncols
    page, sel = 0, ''
    pages = max(1, math.ceil(len(names) / per_page))
    while True:
        stdscr.erase()
        try:
            stdscr.addstr(0, 2, f"pick a model - {len(names)} on server, "
                                f"page {page + 1}/{pages}  "
                                "(number+Enter, n/p pages, Esc cancels)"
                            [:cols - 1])
        except curses.error:
            pass
        for i, n in enumerate(names[page * per_page:
                                    (page + 1) * per_page]):
            num = page * per_page + i + 1
            try:
                stdscr.addstr(2 + i % body, 2 + (i // body) * 26,
                              f"{num:>2} {n[:21]}")
            except curses.error:
                pass
        try:
            stdscr.addstr(rows - 2, 2, f"> {sel}")
        except curses.error:
            pass
        stdscr.refresh()
        k = stdscr.getch()
        if ord('0') <= k <= ord('9') and len(sel) < 4:
            sel += chr(k)
        elif k in (8, 127, curses.KEY_BACKSPACE):
            sel = sel[:-1]
        elif k in (10, 13, curses.KEY_ENTER) and sel:
            i = int(sel) - 1
            if 0 <= i < len(names):
                return names[i]
            sel = ''
        elif k in (ord('n'), ord('N'), curses.KEY_NPAGE):
            page = (page + 1) % pages
        elif k in (ord('p'), ord('P'), curses.KEY_PPAGE):
            page = (page - 1) % pages
        elif k in (27, ord('q'), ord('Q')):
            return None


def confirm_probe(stdscr, rep):
    """Show the probe report; get the user's decision.
    Returns 'play' | 'fallback' | 'model:<name>' | 'quit'.
    Auto-continues when the model is game-ready."""
    stdscr.nodelay(False)          # blocking input for the menus
    stdscr.timeout(-1)

    def draw():
        stdscr.erase()
        y = 1

        def line(s='', attr=0):
            nonlocal y
            try:
                stdscr.addstr(y, 2, s, attr)
            except curses.error:
                pass
            y += 1

        line(f"model probe: {LLM_MODEL}  ({OLLAMA_URL})")
        if rep['reachable']:
            line(f"server: reachable, ping {rep['ping_ms']:.0f} ms, "
                 f"{len(rep['names'])} models")
        for c in rep['calls']:
            mark = 'ok ' if c['ok'] else 'BAD'
            ms = f"{c['ms']:.0f} ms" if c['ms'] is not None else 'failed'
            line(f"test call: {mark}  {ms}  done={c['done'] or '?'}")
        if rep['ok']:
            line()
            line("model ready - starting in 3 s (any key to start now) ...")
        else:
            line()
            line("problems found:", curses.A_BOLD)
            for p in rep['problems']:
                line("  - " + p)
            line()
            line("[Enter] play anyway   [f] CPU fallback   "
                 + ("[m] pick model   " if rep['names'] else "")
                 + "[q] quit")
        stdscr.refresh()

    draw()
    if rep['ok']:
        # guaranteed readable pause before the table appears;
        # any keypress starts immediately
        stdscr.timeout(3000)          # ms
        stdscr.getch()                # waits up to 3 s
        stdscr.timeout(-1)
        return 'play'
    while True:
        k = stdscr.getch()
        if k in (10, 13, curses.KEY_ENTER):
            return 'play'
        if k in (ord('f'), ord('F')):
            return 'fallback'
        if k in (ord('m'), ord('M')) and rep['names']:
            name = pick_model(stdscr, rep['names'])
            if name:
                return 'model:' + name
            draw()                 # cancelled: back to the report
        if k in (ord('q'), ord('Q'), 27):
            return 'quit'


# ------------------------------------------------------ headless sim
def auto_place(game):
    t = game.table
    x = ap.BALL_R + 0.6
    while x < t.W - ap.BALL_R - 0.6:
        y = ap.BALL_R + 0.6
        while y < t.H - ap.BALL_R - 0.6:
            if game.valid_place(x, y):
                t.cue().x, t.cue().y = x, y
                game.state = 'aim'
                return
            y += 1.5
        x += 1.5
    raise RuntimeError('no valid cue placement found')


def auto_shoot(game):
    """The human side plays the v1 heuristic in headless runs."""
    t = game.table
    cue = t.cue()
    if game.break_shot:
        apex = t.balls[1]
        game.angle = math.atan2(apex.y - cue.y, apex.x - cue.x)
        game.power = 1.0
        game.shoot(0)
        return
    targets = game.legal_targets(0)
    best = ap.best_shot(t, cue.x, cue.y, targets)
    if best:
        _, ang, v, b = best
        game.angle = ang
        game.power = max(0.0, min(1.0, (v - ap.SHOT_V_MIN) /
                                  (ap.SHOT_V_MAX - ap.SHOT_V_MIN)))
    else:
        near = min(targets, key=lambda b: math.hypot(b.x - cue.x,
                                                     b.y - cue.y))
        game.angle = math.atan2(near.y - cue.y, near.x - cue.x)
        game.power = 0.35
    game.shoot(0)


def run_headless(racks, seed0, w=60, h=20):
    """Full racks without curses: the state machine end-to-end, with
    the LLM in the loop when the probe passes."""
    ap.play_sound = lambda kind: None      # no audio in headless runs
    rep = probe_model()
    if rep['ok']:
        print('probe: %s ready (%.0f ms)' %
              (LLM_MODEL, rep['calls'][-1]['ms']))
    else:
        ADVISOR.enabled = False
        ADVISOR.user_disabled = True
        print('probe: %s -> heuristic CPU' % '; '.join(rep['problems']))
    wins = [0, 0]
    llm = heur = 0
    for i in range(racks):
        random.seed(seed0 + i)
        game = LLMGame(w, h, OPTS['error'])
        shots = 0
        done = False
        for _ in range(600):
            st = game.state
            if st == 'over':
                done = True
                break
            if st == 'aim':
                auto_shoot(game)
                shots += 1
            elif st == 'place':
                auto_place(game)
            elif st == 'cpu':
                guard = 0
                while game.state == 'cpu':
                    game.cpu_act()
                    if game.cpu_phase == 'show':
                        game.cpu_show_t = 0.0    # skip the aiming pause
                    elif game.cpu_phase == 'think':
                        time.sleep(0.05)         # let the worker run
                    guard += 1
                    if guard > 400:
                        break
                shots += 1
                if game.llm_used:
                    llm += 1
                elif game.llm_used is False:
                    heur += 1
            elif st == 'sim':
                n = 0
                while game.table.moving() and n < 60000:
                    game.table.step(1 / 240.0)
                    n += 1
                game.resolve()
        if not done:
            raise RuntimeError('rack %d did not finish (state=%s)' %
                               (i + 1, game.state))
        wins[game.winner] += 1
        left = ''.join(ap.ball_char(b.n) for b in game.table.balls
                       if b.on and b.n)
        print('rack %d: winner=%-3s shots=%d left=%s' %
              (i + 1, 'you' if game.winner == 0 else 'cpu', shots,
               left or '-'))
    print('totals: you=%d cpu=%d  cpu shots: llm=%d heuristic=%d' %
          (wins[0], wins[1], llm, heur))
    print('OK')


# ---------------------------------------------------------------- main
def main(stdscr):
    global LLM_MODEL
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    colors = ap.init_colors()
    # pre-flight probe (pong v2 flow)
    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        try:
            stdscr.addstr(max(0, rows // 2 - 1), 2,
                          ('probing %s at %s ...'
                           % (LLM_MODEL, OLLAMA_URL))[:cols - 1])
            stdscr.addstr(max(0, rows // 2), 2,
                          '(two test calls - may take a while for slow '
                          'models)'[:cols - 1])
        except curses.error:
            pass
        stdscr.refresh()
        rep = probe_model()
        action = confirm_probe(stdscr, rep)
        if action == 'play':
            ADVISOR.enabled = True
            break
        if action == 'fallback':
            ADVISOR.enabled = False
            ADVISOR.user_disabled = True   # stays off, even across racks
            break
        if action == 'quit':
            return
        if action.startswith('model:'):
            LLM_MODEL = action[6:]     # re-probe the new choice
    stdscr.nodelay(True)     # confirm_probe left blocking input
    scr_h, scr_w = stdscr.getmaxyx()
    h = max(10, min(scr_h - 8, 30))
    # cap width at 4x height: a real table's 2:1 proportions (a cell
    # is ~1:2) - a wide terminal gives a bigger table, not a distorted
    # one, and long shots stay aimable
    w = max(30, min(scr_w - 2, 4 * h))
    game = LLMGame(w, h, OPTS['error'])
    while True:
        tag = 'model: %s' % LLM_MODEL
        if not ADVISOR.enabled:
            tag += ' (cpu fallback)'
        if game.last_ms:
            tag += ' - %d ms' % game.last_ms
        ap.draw(stdscr, game, colors, tag)
        k = stdscr.getch()
        if k in (ord('q'), 27):
            return
        st = game.state
        if st == 'aim':
            if k in (curses.KEY_LEFT, ord('h'), ord('a')):
                game.angle -= ap.aim_delta(game)
            elif k in (curses.KEY_RIGHT, ord('l'), ord('d')):
                game.angle += ap.aim_delta(game)
            elif k == ord(','):
                game.angle -= ap.aim_delta(game, True)
            elif k == ord('.'):
                game.angle += ap.aim_delta(game, True)
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
                game.place_x = max(ap.BALL_R, game.place_x - step)
            elif k in (curses.KEY_RIGHT, ord('l'), ord('d')):
                game.place_x = min(game.table.W - ap.BALL_R,
                                   game.place_x + step)
            elif k in (curses.KEY_UP, ord('k'), ord('w')):
                game.place_y = max(ap.BALL_R, game.place_y - step)
            elif k in (curses.KEY_DOWN, ord('j'), ord('s')):
                game.place_y = min(game.table.H - ap.BALL_R,
                                   game.place_y + step)
            elif k == ord(','):
                game.place_x = max(ap.BALL_R, game.place_x - 0.2)
            elif k == ord('.'):
                game.place_x = min(game.table.W - ap.BALL_R,
                                   game.place_x + 0.2)
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
            if ap.run_sim(stdscr, game, colors, tag) == 'quit':
                return
            game.resolve()
        elif st == 'over':
            if not game.prize_shown and game.winner == 0:
                game.prize_shown = True
                ap.show_prize(stdscr)
            if k == ord('r'):
                game.rematch()
            time.sleep(0.02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ASCII 8-ball pool vs an Ollama LLM')
    parser.add_argument('--model', default=LLM_MODEL, metavar='MODEL',
                        help='Ollama model choosing the CPU shots '
                             '(default: %(default)s)')
    parser.add_argument('--timeout', type=float, default=LLM_TIMEOUT,
                        metavar='SECS',
                        help='per-shot LLM timeout in seconds '
                             '(default: %(default)s)')
    parser.add_argument('--error', type=float, default=ap.AI_ERROR_DEG,
                        metavar='DEG',
                        help='aim jitter in degrees, the difficulty knob '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed')
    parser.add_argument('--sim', type=int, default=None, metavar='RACKS',
                        help='headless smoke test: RACKS automated racks')
    parser.add_argument('--width', type=int, default=60, metavar='COLS',
                        help='table width for --sim (default: %(default)s)')
    parser.add_argument('--height', type=int, default=20, metavar='ROWS',
                        help='table height for --sim (default: %(default)s)')
    args = parser.parse_args()
    LLM_MODEL = args.model          # rebind globals; workers read them live
    LLM_TIMEOUT = args.timeout
    OPTS['error'] = args.error
    if args.sim:
        run_headless(args.sim, args.seed or 1, args.width, args.height)
    else:
        if args.seed is not None:
            random.seed(args.seed)
        try:
            curses.wrapper(main)
        except KeyboardInterrupt:
            pass
