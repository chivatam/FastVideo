# WRITING_AUDIT — reverse outline + quality passes (v2 rewrite)

Per `ml-paper-writing` (claim ledger, narrative principle), ARIS
`paper-write` (5 audit passes, reverse outline), and Slazee reverse
outlining. Draft audited: `PAPER.md` (v2); previous draft preserved as
`PAPER.v1-backup.md`.

## Reverse outline (topic sentences, in order)

1. §1: problem scale (quadratic attention, 75k tokens)
2. §1: "The deployed system does not deliver it." (gap, C1)
3. §1: "The loss occurs at the boundary between two independently designed
   components." (insight, C2-C3)
4. §1: "Our fix changes the selector rather than the mask or the kernel."
   (method, C4)
5. §1: contributions (C1-C10 mapped)
6. §1: general lesson (implication)
7. §2: three methodological categories, each ending in positioning
8. §3: preserved family -> exact mapping -> boundary correctness -> what is
   unchanged
9. §5.1-5.3: each opens with "This experiment tests claim C#"
10. §6: decomposition -> Amdahl envelope -> precision independence ->
    scaling (H1 marked untested)

Verdict: topic sentences alone reproduce the Problem -> Gap -> Insight ->
Method -> Evidence -> Implication story. No orphan paragraphs.

## Claim coverage

C1-C11 all appear in the sections assigned by `PAPER_PLAN.md`; H1-H3 appear
only as marked hypotheses (§6.3, §7). No `[CLAIM NEEDS EVIDENCE]` markers
remain. No claim exceeds its ledger wording (checked: no
"statistically indistinguishable", no non-inferiority, 1.40x always BF16,
8.93x never presented as E2E).

## Quality passes

- Pass 1 clutter / AI-isms: watch-word scan clean (delve, pivotal,
  landscape, underscore, notably, importantly, groundbreaking, remarkable,
  sobering, etc. — zero hits).
- Pass 2-3 voice/architecture: experiments lead with the claim tested;
  subjects and verbs kept adjacent in headline sentences.
- Pass 4 keyword consistency (Banana Rule): "selector tile",
  "kernel sparse granularity", "retention", "deployed VSA (P2)",
  "aligned configuration (P4G)" used consistently; no mid-paper renames.
- Pass 5 numerical/citation integrity: all numbers re-checked against
  `data/*.csv` and canonical tables; citations verified 2026-08-17
  (`SOTA_RECOVERY_LIT_REVIEW.md`); FA4 characterization explicitly
  second-hand.

## Abstract check (Farquhar 5-part)

1. what: geometry re-alignment converts 1.13x -> 1.40x. 2. why hard:
selector and kernel designed separately; both routes across the boundary
lose. 3. how: one-constant selector change, 1:1 mapping. 4. evidence:
kernel sweep, E2E, 326-prompt Holm-corrected VBench. 5. strongest result:
1.40x E2E / alignment beats precision. Self-contained; one concrete
quantitative result per slot.
