# Answer 1 (baseline) vs Answer 2 (re-ranked)

Models: `gemini-3.5-flash` (8), `gemini-3.1-flash-lite` (2) · temperature: provider default (see NOTES.md 1.2) · thinking: minimal · identical prompt template for both answers. Both answers of any one query always come from the same model (asserted in §5).

## Q1 — keyword_trap

**Q:** How many cards is a player allowed to keep in their hand between rounds?

*Model:* `gemini-3.5-flash` · *gold rules:* `2.4` · *gold rank:* `#1.0` → `#1.0` · *outcome:* `changed_context_same_verdict` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `6;137;91` · *Re-ranked Top 3:* `6;57;70`

**Answer 1**

According to Rule 2.4, a player's hand can have a maximum of seven action cards, though a game effect can increase or decrease this limit. If a player ever has more than seven action cards, they must choose seven cards to keep and discard the rest (Rule 2.4).

**Rules cited:** Rule 2.4

**Answer 2**

According to Rule 2.4, a player’s hand can have a maximum of seven action cards, and if a player ever has more than seven, they must choose seven to keep and discard the rest. However, game effects can increase or decrease this hand limit. 

[Rule 2.4]

---

## Q2 — synonym_gap

**Q:** Can a warship fly into a star that has exploded?

*Model:* `gemini-3.5-flash` · *gold rules:* `75.0;75.1` · *gold rank:* `#1.0` → `#1.0` · *outcome:* `changed_context_same_verdict` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `140;115;145` · *Re-ranked Top 3:* `140;24;115`

**Answer 1**

No, a ship cannot move into a supernova (a star that has exploded). Rule 65.0 classifies carriers, cruisers, dreadnoughts, destroyers, fighters, war suns, and flagships as ships. Rule 75.1 explicitly states that a ship cannot move through or into a supernova.

[Rule 65.0, Rule 75.1]

**Answer 2**

The provided rules state that a ship cannot move through or into a supernova. Because warships fall under the definition of ships, they are subject to this restriction. Therefore, a warship cannot fly into a supernova. 

Relying on Rules: Rule 11 (11.0, 11.1), Rule 65 (65.0), and Rule 75 (75.0, 75.1).

---

## Q3 — exception_corner

**Q:** May a player build new warships at a shipyard while an enemy fleet sits in the same system?

*Model:* `gemini-3.5-flash` · *gold rules:* `58.6;14.0;14.1` · *gold rank:* `#4.0` → `#6.0` · *outcome:* `changed_context_same_verdict` · *answers identical:* `True`

*Baseline Top 3 (chunk ids):* `115;129;45` · *Re-ranked Top 3:* `115;61;84`

**Answer 1**

The provided rules do not cover this.

**Answer 2**

The provided rules do not cover this.

---

## Q4 — negation_constraint

**Q:** Can I take a tactical action on a system I have already claimed this round?

*Model:* `gemini-3.5-flash` · *gold rules:* `5.2` · *gold rank:* `#6.0` → `#1.0` · *outcome:* `promotion` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `146;144;145` · *Re-ranked Top 3:* `11;121;144`

**Answer 1**

No, you cannot perform a tactical action on a system that already contains one of your command tokens. During Step 1 (Activation) of a tactical action, you must activate a system that does not contain one of your command tokens by placing a token from your tactic pool into that system (Rule 78.1). Additionally, during Step 2 (Movement), you can only move ships out of systems that do not contain one of your command tokens (Rule 78.2).

Cited Rules: 78.1, 78.2

**Answer 2**

Yes, you can take a tactical action in a system you have already claimed, provided it does not currently contain one of your command tokens. The rules specify that a player cannot activate a system that already contains one of their own command tokens, but command tokens belonging to other players do not prevent activation. 

(Rules 5.1, 5.2, 5.3, 78.1)

---

## Q5 — role_limit

**Q:** How many planetary defence systems can one player have on the same world?

*Model:* `gemini-3.5-flash` · *gold rules:* `54.3;74.5` · *gold rank:* `#1.0` → `#1.0` · *outcome:* `changed_context_same_verdict` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `95;77;55` · *Re-ranked Top 3:* `95;44;77`

**Answer 1**

Based on Rule 54.3, each planet can have a maximum of two PDS units. 

*(Cited: Rule 54.3)*

**Answer 2**

Based on the provided rules, each planet can have a maximum of two PDS units. 

(Rule 54.3)

---

## Q6 — exception_corner

**Q:** Who may remove the token blocking the galaxy's central planet, and what does it cost?

*Model:* `gemini-3.5-flash` · *gold rules:* `46.1` · *gold rank:* `#nan` → `#nan` · *outcome:* `changed_context_same_verdict` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `34;47;49` · *Re-ranked Top 3:* `49;47;152`

**Answer 1**

According to Rule 26.2, the active player may remove the custodians token from Mecatol Rex before the "Commit Ground Forces" step of an invasion. To do so, that player must spend six influence and commit at least one ground force to land on the planet.

(Rule 26.2)

**Answer 2**

The active player may remove the custodians token from Mecatol Rex. To do so, the player must spend six influence and commit at least one ground force to land on the planet.

(Rule 26.2)

---

## Q7 — negation_constraint

**Q:** When is a player forbidden from passing their turn?

*Model:* `gemini-3.5-flash` · *gold rules:* `3.2;3.4` · *gold rank:* `#1.0` → `#1.0` · *outcome:* `no_op_identical` · *answers identical:* `True`

*Baseline Top 3 (chunk ids):* `9;8;10` · *Re-ranked Top 3:* `9;8;10`

**Answer 1**

A player is forbidden from passing their turn until they have performed a strategic action. Additionally, during a three-player or four-player game, a player cannot pass until they have exhausted both of their strategy cards. 

(Rule 3.4)

**Answer 2**

A player is forbidden from passing their turn until they have performed a strategic action. Additionally, during a three-player or four-player game, a player cannot pass until they have exhausted both of their strategy cards. 

(Rule 3.4)

---

## Q8 — buried_vital_chunk

**Q:** If a unit takes damage, does it fight less effectively?

*Model:* `gemini-3.5-flash` · *gold rules:* `76.2` · *gold rank:* `#2.0` → `#2.0` · *outcome:* `changed_context_same_verdict` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `53;141;142` · *Re-ranked Top 3:* `142;141;23`

**Answer 1**

No, a unit does not fight less effectively when it takes damage. According to rule 76.2, a damaged unit does not have reduced capabilities and is functionally the same as an undamaged unit, except that it cannot use the "Sustain Damage" ability again until it is repaired.

**Source Rule:** 76.2

**Answer 2**

No, a unit does not fight less effectively when it takes damage. According to Rule 76.2, a damaged unit does not have reduced capabilities and is functionally the same as an undamaged unit, except that it cannot use the "Sustain Damage" ability again.

(Relying on Rule 76.2)

---

## Q9 — role_limit

**Q:** How many swaps am I allowed with the player beside me during one turn?

*Model:* `gemini-3.1-flash-lite` · *gold rules:* `83.1` · *gold rank:* `#8.0` → `#1.0` · *outcome:* `promotion` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `9;8;137` · *Re-ranked Top 3:* `160;9;159`

**Answer 1**

The provided rules do not cover this.

**Answer 2**

During the active player’s turn, they may resolve up to one transaction with each of their neighbors (Rule 83.1). A transaction allows the exchange of commodities, trade goods, and promissory notes (Rule 83.0). The provided rules do not specify a limit on the number of individual items or "swaps" exchanged within that single transaction, as long as the terms are agreed upon (Rule 83.4).

---

## Q10 — negation_constraint

**Q:** What stops my factory from turning out new warships?

*Model:* `gemini-3.1-flash-lite` · *gold rules:* `14.0;14.1` · *gold rank:* `#7.0` → `#1.0` · *outcome:* `promotion` · *answers identical:* `False`

*Baseline Top 3 (chunk ids):* `115;105;61` · *Re-ranked Top 3:* `27;83;84`

**Answer 1**

You are limited by the number of units in your reinforcements (58.5). Additionally, you cannot produce ships in a system that contains other players’ ships (58.6). Finally, the number of non-fighter ships you can have in a system is limited by the command tokens in your fleet pool (65.2, 34.1, 34.3).

**Answer 2**

Your factory (a unit with "Production") cannot produce ships if it is blockaded (Rule 14.1). A unit is blockaded if it is in a system that contains other players' ships but does not contain any of your own ships (Rule 14.0).

---
