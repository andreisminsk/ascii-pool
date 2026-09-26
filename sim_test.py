#!/usr/bin/env python3
"""Headless full-rack simulation: both players automated.

Verifies the state machine end-to-end without curses: break, group
assignment, fouls, ball-in-hand placement, 8-ball resolution.

Run: python3 sim_test.py [RACKS] [SEED]
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ascii_pool as ap

ap.play_sound = lambda kind: None      # no audio in headless runs


def run_sim_headless(game, cap=60000):
    t = game.table
    n = 0
    while t.moving():
        t.step(1 / 240.0)
        n += 1
        if n > cap:
            return False
    return True


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


def auto_shoot(game, player):
    t = game.table
    cue = t.cue()
    if game.break_shot:
        apex = t.balls[1]
        game.angle = math.atan2(apex.y - cue.y, apex.x - cue.x)
        game.power = 1.0
        game.shoot(player)
        return
    targets = game.legal_targets(player)
    best = ap.best_shot(t, cue.x, cue.y, targets)
    if best:
        _, ang, v, b = best
        game.angle = ang
        game.power = max(0.0, min(1.0, (v - ap.SHOT_V_MIN) /
                                  (ap.SHOT_V_MAX - ap.SHOT_V_MIN)))
    else:
        near = min(targets, key=lambda b: math.hypot(b.x - cue.x, b.y - cue.y))
        game.angle = math.atan2(near.y - cue.y, near.x - cue.x)
        game.power = 0.35
    game.shoot(player)


def play_rack(seed):
    random.seed(seed)
    game = ap.Game(60, 20, 1.5)
    shots = 0
    for _ in range(600):
        st = game.state
        if st == 'over':
            break
        if st == 'aim':
            auto_shoot(game, 0)
            shots += 1
        elif st == 'place':
            auto_place(game)
        elif st == 'cpu':
            guard = 0
            while game.state == 'cpu':
                game.cpu_act()
                if game.cpu_phase == 'show':
                    game.cpu_show_t = 0.0    # skip the aiming pause
                guard += 1
                if guard > 50:
                    break
            shots += 1
        elif st == 'sim':
            assert run_sim_headless(game), 'simulation did not settle'
            game.resolve()
    else:
        raise RuntimeError('rack did not finish (state=%s)' % game.state)
    assert game.state == 'over', 'no winner, state=%s' % game.state
    assert game.winner in (0, 1)
    left = ''.join(ap.ball_char(b.n) for b in game.table.balls if b.on and b.n)
    return game.winner, shots, left


if __name__ == '__main__':
    racks = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    wins = [0, 0]
    for i in range(racks):
        w, shots, left = play_rack(seed0 + i)
        wins[w] += 1
        print('rack %d: winner=%-3s shots=%d left=%s' %
              (i + 1, 'you' if w == 0 else 'cpu', shots, left or '-'))
    print('totals: you=%d cpu=%d' % (wins[0], wins[1]))
    print('OK')
