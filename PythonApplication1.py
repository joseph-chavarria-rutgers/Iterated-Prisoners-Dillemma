#!/usr/bin/env python3
"""
Prisoner's Dilemma Tournament (Round-Robin with optional Evolution)

Strategies included:
- TitForTat
- AlwaysCooperate
- AlwaysDefect
- RandomPlayer (with adjustable cooperation probability)

NEW: Evolution across generations
- You can run multiple generations (via --generations N).
- After each generation's tournament, a new population is created by
  selecting parents with probability proportional to their scores.
- Offspring inherit only the parent's strategy (no mutation here).
"""

from __future__ import annotations
import argparse
import random
from typing import Dict, List, Tuple
import os

# Track which strategy labels existed in generation 0 so we can show extinct strategies
INITIAL_LABELS: set[str] = set()


# ==========================
# 1) Payoff Matrix (standard PD)
# ==========================
# Moves are 'C' (cooperate) or 'D' (defect)
T = 5  # Temptation to defect (player defects, opponent cooperates)
R = 3  # Reward for mutual cooperation
P = 1  # Punishment for mutual defection
S = 0  # Sucker's payoff (player cooperates, opponent defects)

# (player_move, opponent_move) -> payoff awarded to the player
PAYOFF: Dict[Tuple[str, str], int] = {
    ('C', 'C'): R,
    ('C', 'D'): S,
    ('D', 'C'): T,
    ('D', 'D'): P,
}


# ==========================
# 2) Base Player class
# ==========================
class Player:
    """
    Base class for all strategies.
    - label: strategy name (e.g., 'TitForTat', 'AlwaysDefect')
    - score: cumulative score within a generation
    Subclasses implement make_move(my_history, opp_history) -> 'C' or 'D',
    where histories are lists of moves within the current match only.
    """

    def __init__(self, label: str):
        self.label = label
        self.score = 0

    def make_move(self, my_history: List[str], opp_history: List[str]) -> str:
        """Return 'C' or 'D'. Must be overridden by each strategy subclass."""
        raise NotImplementedError

    def __repr__(self) -> str:
        # Small numeric suffix helps distinguish instances when printing
        return f"{self.label}({id(self) % 10000})"


# ==========================
# 3) Strategy subclasses
# ==========================
class AlwaysDefect(Player):
    def __init__(self):
        super().__init__('AlwaysDefect')

    def make_move(self, my_history: List[str], opp_history: List[str]) -> str:
        return 'D'


class AlwaysCooperate(Player):
    def __init__(self):
        super().__init__('AlwaysCooperate')

    def make_move(self, my_history: List[str], opp_history: List[str]) -> str:
        return 'C'


class RandomPlayer(Player):
    def __init__(self, p_cooperate: float = 0.5):
        super().__init__('Random')
        self._p = float(p_cooperate)

    def make_move(self, my_history: List[str], opp_history: List[str]) -> str:
        return 'C' if random.random() < self._p else 'D'


class TitForTat(Player):
    def __init__(self):
        super().__init__('TitForTat')

    def make_move(self, my_history: List[str], opp_history: List[str]) -> str:
        # Cooperate on the first move, then copy the opponent's previous move
        if len(opp_history) == 0:
            return 'C'
        return opp_history[-1]
    
class Grudger(Player):
    def __init__(self):
        super().__init__('Grudger')

    def make_move(self, my_history: List[str], opp_history: List[str]) -> str:
        # Stateless: defect if the opponent has defected earlier in THIS match
        return 'D' if 'D' in opp_history else 'C'


# ==========================
# 4) Round & Match mechanics
# ==========================
def play_round(p1: Player, p2: Player, h1: List[str], h2: List[str]) -> None:
    """
    Play one round between p1 and p2.
    h1/h2 are the pair-specific histories (moves made so far in this match).
    Updates scores and appends current moves to the histories.
    """
    m1 = p1.make_move(h1, h2)
    m2 = p2.make_move(h2, h1)

    # Update scores using the payoff matrix
    p1.score += PAYOFF[(m1, m2)]
    p2.score += PAYOFF[(m2, m1)]

    # Record moves for future rounds in this match
    h1.append(m1)
    h2.append(m2)


def play_match(p1: Player, p2: Player, rounds: int) -> None:
    """
    Play a repeated PD match of 'rounds' rounds between the same two players.
    Histories are per match: reset to empty lists when a new opponent pair begins.
    """
    h1: List[str] = []
    h2: List[str] = []
    for _ in range(rounds):
        play_round(p1, p2, h1, h2)


# ==========================
# 5) Build population
# ==========================
def build_population(
    n_tft: int,
    n_coop: int,
    n_defect: int,
    n_rand: int,
    n_grudge: int,
    p_cooperate_random: float = 0.5,
) -> List[Player]:
    """
    Create a list of Player instances according to requested counts.
    RandomPlayer's cooperation probability can be adjusted via p_cooperate_random.
    """
    players: List[Player] = []
    for _ in range(n_tft):
        players.append(TitForTat())
    for _ in range(n_coop):
        players.append(AlwaysCooperate())
    for _ in range(n_defect):
        players.append(AlwaysDefect())
    for _ in range(n_rand):
        players.append(RandomPlayer(p_cooperate_random))
    for _ in range(n_grudge):
        players.append(Grudger())
    return players


def make_player_from_label(label: str, p_rand: float) -> Player:
    """
    Helper: create a new Player instance given a label.
    Used when constructing the next generation.
    """
    if label == 'TitForTat':
        return TitForTat()
    elif label == 'AlwaysCooperate':
        return AlwaysCooperate()
    elif label == 'AlwaysDefect':
        return AlwaysDefect()
    elif label == 'Random':
        return RandomPlayer(p_rand)
    elif label == 'Grudger':
        return Grudger()
    else:
        raise ValueError(f"Unknown strategy label: {label}")


# ==========================
# 6) Tournament (round-robin)
# ==========================
def run_tournament(players: List[Player], rounds_per_pair: int) -> None:
    """
    Every distinct pair of players plays a 'rounds_per_pair' round match.
    """
    n = len(players)
    for i in range(n):
        for j in range(i + 1, n):
            play_match(players[i], players[j], rounds_per_pair)


# ==========================
# 7) Reporting
# ==========================
def summarize_by_strategy(players: List[Player]) -> Dict[str, Dict[str, float]]:
    """
    Group results by strategy label and compute count, total score, and average per player.
    """
    by: Dict[str, Dict[str, float]] = {}
    for p in players:
        s = p.label
        if s not in by:
            by[s] = {"count": 0.0, "total_score": 0.0}
        by[s]["count"] += 1.0
        by[s]["total_score"] += float(p.score)

    for s, d in by.items():
        d["avg_per_player"] = d["total_score"] / max(1.0, d["count"])
    return by


def print_results(players: List[Player], rounds_per_pair: int, generation: int | None = None) -> None:
    """
    Pretty-print tournament settings and per-strategy outcomes.
    If generation is provided, it is included in the header.

    Extinct strategies that were present in generation 0 will be shown with count=0.
    """
    n = len(players)
    total_pairs = n * (n - 1) // 2
    total_rounds = total_pairs * rounds_per_pair

    header = "=== Prisoner's Dilemma Tournament Results ==="
    if generation is not None:
        header = f"=== Generation {generation} Results ==="
    print("\n" + header, flush=True)
    print(f"Players: {n}", flush=True)
    print(f"Rounds per pair: {rounds_per_pair}", flush=True)
    print(f"Total rounds played: {total_rounds}", flush=True)

    by = summarize_by_strategy(players)

    # Include strategies that existed in generation 0 even if their current count is zero
    for label in INITIAL_LABELS:
        if label not in by:
            by[label] = {"count": 0.0, "total_score": 0.0, "avg_per_player": 0.0}

    print("\nStrategy summary (sorted by avg/player):", flush=True)
    for strat, d in sorted(by.items(), key=lambda kv: kv[1]["avg_per_player"], reverse=True):
        print(
            f"- {strat:16s} | count={int(d['count']):3d} | "
            f"total={int(d['total_score']):8d} | avg/player={d['avg_per_player']:.2f}"
        , flush=True
        )


# ==========================
# 8) NEW: Evolution across generations
# ==========================
def evolve_population(players: List[Player], p_rand: float) -> List[Player]:
    """
    Create the next generation by selecting parents with probability
    proportional to their scores (fitness-proportionate selection).
    Offspring inherit only the parent's strategy label (no mutation).
    """
    # Avoid zero probability: treat extremely bad players as having tiny fitness
    scores = [max(p.score, 1.0) for p in players]
    labels = [p.label for p in players]
    total_score = sum(scores)

    # If somehow everyone scored 0 (shouldn't happen with our payoff setup), fallback to uniform
    if total_score == 0:
        weights = None
    else:
        weights = scores

    population_size = len(players)
    new_labels = random.choices(labels, weights=weights, k=population_size)

    new_players = [make_player_from_label(label, p_rand) for label in new_labels]
    return new_players


def run_evolution(
    n_generations: int,
    initial_players: List[Player],
    rounds_per_pair: int,
    p_rand: float,
) -> None:
    """
    Run multiple generations.
    Each generation:
      1. Reset scores.
      2. Run tournament.
      3. Print results.
      4. Build next generation (except after last gen).
    """
    players = initial_players

    for g in range(n_generations):
        # Reset scores at the start of each generation
        for p in players:
            p.score = 0

        run_tournament(players, rounds_per_pair)
        print_results(players, rounds_per_pair, generation=g)

        # Skip evolution after the final generation
        if g == n_generations - 1:
            break

        players = evolve_population(players, p_rand)


# ==========================
# 9) Main (CLI)
# ==========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma Tournament (Round-Robin, optional evolution).")
    parser.add_argument("--tft", type=int, default=10, help="Number of TitForTat players.")
    parser.add_argument("--coop", type=int, default=10, help="Number of AlwaysCooperate players.")
    parser.add_argument("--defect", type=int, default=10, help="Number of AlwaysDefect players.")
    parser.add_argument("--rand", type=int, default=20, help="Number of Random players.")
    parser.add_argument("--p-rand", type=float, default=0.5, help="Random player's P(cooperate).")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per pair in each match.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations to simulate.")
    parser.add_argument("--grudge", type=int, default=10, help="Number of Grudger players.")
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    print(f"For help on how to use this program, run: python {os.path.basename(__file__)} --help", flush=True)

    # Ensure we always seed the RNG and print the actual seed used so runs are reproducible
    if args.seed is not None:
        seed_used = int(args.seed)
    else:
        # Use system entropy to pick a seed and then seed the RNG with it so we can print it
        seed_used = random.SystemRandom().randint(0, 2**32 - 1)

    random.seed(seed_used)
    print(f"Seed used: {seed_used}", flush=True)

    initial_players = build_population(
        n_tft=args.tft,
        n_coop=args.coop,
        n_defect=args.defect,
        n_rand=args.rand,
        p_cooperate_random=args.p_rand,
        n_grudge=args.grudge,
    )

    # record the set of strategy labels present at generation 0 so we can report extinct ones
    INITIAL_LABELS.clear()
    INITIAL_LABELS.update(p.label for p in initial_players)

    # If generations == 1, this behaves like the original single-tournament version.
    run_evolution(
        n_generations=args.generations,
        initial_players=initial_players,
        rounds_per_pair=args.rounds,
        p_rand=args.p_rand,
    )


if __name__ == "__main__":
    main()
