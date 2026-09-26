# ASCII Pool

Single-file 8-ball billiards for your terminal — pure Python,
standard library only, zero dependencies. Sister project to
ascii-pong: same ethos (continuous float physics rendered to
terminal cells, tuning constants at the top, fits any POSIX
terminal, including Termux on phones).

## Run

```sh
python3 ascii_pool.py                 # you vs the CPU
python3 ascii_pool.py --error 2.5     # clumsier CPU
python3 ascii_pool.py --selftest      # headless physics check
```

## Controls

| Key | Action |
|---|---|
| `←` `→` or `h/l` or `a/d` | rotate aim |
| `,` / `.` | fine aim |
| `↑` `↓` or `j/k` or `w/s` | power up / down |
| `Space` | shoot (or drop the cue ball) |
| `Q` / `Esc` | quit |
| `R` | rematch (at game end) |

Aim steps are distance-adaptive: one tap slides the predicted
contact point by a fixed number of cells (`AIM_COARSE_CELLS` = 2,
`AIM_FINE_CELLS` = 0.4), so a far target is as easy to zero in on
as a near one — the angular step shrinks with distance. Sweeping
with no ball in the ray keeps the full step.

## Display

- Cue ball `O`, solids `1`–`7` (yellow), 8-ball `8`,
  stripes `A`–`G` (= 9–15, cyan).
- The dotted guide shows where the cue ball first makes contact;
  the short line from the struck ball shows where it will go;
  the tiny third line is the cue ball's deflection.
- `#` is the cushion, `@` are the six pockets — on large tables
(≥ `WIDE_POCKET_W` cells wide) each mouth spans 3 cells.
- The table grows with the terminal but is capped at 4x its height
(a real table's 2:1 proportions, since a cell is ~1:2) — stretching
across the full width made long shots nearly impossible to aim.
Shot speeds, the aim guide, aim steps, and physics substeps scale
with width, so pace and aiming precision stay constant at any size.

## Rules (8-ball, simplified)

**Objective.** You are assigned a group — solids (1–7) or
stripes (9–15). Pocket all seven of your group, then pocket the
**8** to win the rack.

**The break.** The opening breaker is chosen at random. The
rack is broken from the cue ball's starting spot. If the 8 goes
down on the break, the rack is re-racked and the same player
breaks again.

**Open table.** Until a group is assigned the table is **open**:
either player may hit any ball except the 8 first. The first
player to legally pocket a ball *after* the break claims that
ball's group; the opponent gets the other. (Pocketing on the
break does not assign groups.)

**Turns.** You keep shooting as long as you legally pocket a
ball from your own group. A miss, a foul, or pocketing only the
opponent's balls ends your turn.

**Fouls** — the opponent gets **ball in hand** (place the cue
ball anywhere on the table):
- **Scratch** — the cue ball is pocketed.
- **Wrong ball first** — the cue ball's first contact is not
  one of your own balls (or, on an open table, the 8).
- **No contact** — the cue ball hits nothing.

**Winning and losing.**
- Win: pocket the 8 after clearing your group, with no foul on
  that shot.
- Lose: pocket the 8 early, or pocket the 8 together with a
  scratch — the opponent wins the rack immediately.

**Simplified away** (vs. official rules): call-shot, the
rail-after-contact requirement, and behind-the-line placement
after fouls on the break.

## Physics

- Continuous float simulation, sub-stepped 8× per frame so fast
  balls can't tunnel through each other or the cushions.
- Exponential friction (a ball rolls ≈ v0/K cells), elastic
  equal-mass ball collisions, restitution on the cushions.
- Pockets are **gaps in the cushion** plus a capture radius —
  near-misses rattle in the jaws like the real thing.

## CPU

Ghost-ball geometry: for every (own ball, pocket) pair it
computes the exact contact point, checks both path segments are
clear, scores the shot (distance + cut angle), picks the best,
then adds gaussian aim jitter (`--error`, default 1.5°). No
feasible shot? It plays a soft safety at the nearest legal
ball. On ball in hand it scans candidate positions for the best
leave.

## Tuning

| Constant | Meaning |
|---|---|
| `FRICTION_K` | cloth speed (total roll ≈ v0/K) |
| `CUSHION` | cushion restitution |
| `SHOT_V_MIN/MAX` | power range |
| `CORNER/SIDE_POCKET_R` | pocket capture radii |
| `CORNER/SIDE_GAP` | pocket mouth width |
| `WIDE_POCKET_W` | table width where pockets widen to 3 cells |
| `AI_ERROR_DEG` | CPU aim jitter — the difficulty knob |
| `AIM_COARSE/FINE_CELLS` | contact-point slide per aim tap, cells |
| `BALL_R` | ball radius in cells |

## Sounds

Same trick as ascii-pong: detects a player at runtime
(`termux-media-player`, sox `play`, `mpv`, `paplay`, `ffplay`,
`afplay`) and synthesizes click/rail/pocket tones with the
stdlib `wave` module. No player → silent.

## Prize

Win a rack and ASCII art lands in your scrollback (`diana.txt` if
present, else a built-in trophy).

## v2 — LLM opponent

`ascii_pool_llm.py` — the CPU's shot choice goes to an Ollama LLM:
physics enumerates the feasible shots, the model picks one and sets
the power. Circuit breaker + pre-flight probe; falls back to the v1
heuristic on silence, bad output, or outage. See `POOL-AI-ARCH.md`.
