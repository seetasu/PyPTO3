---
name: ship
description: Use when the user explicitly asks to publish, merge, and clean the current change as one completed GitHub delivery.
---

# Ship a Completed Change

Use this workflow only when the user wants the active change fully delivered,
including merge and local branch cleanup. For a pull request that should remain
open for review, use `github-pr` or `auto-pr` instead.

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full suite.
If CI cannot run it, report that limitation instead of substituting a local
full-suite run.

## Establish the delivery boundary

Resolve the [repository scope gate](../../lib/repository/scope.md) before any
mutation. A `ship` request authorizes publishing, merging, and cleaning up only
the one pull request this invocation creates or updates; it never authorizes
cleanup of pre-existing branches or unrelated pull requests. A later explicit
`ship cleanup` request has the narrower authority defined below.

Read and use [automatic pull request](../auto-pr/SKILL.md) to publish the
current change and obtain its exact PR identity. It owns commit, push,
verification, and review-repair mechanics. Keep the reported host, repository,
number, URL, base, head branch, and head OID as immutable delivery identity.

## Merge only a ready pull request

Before merging, refresh the exact PR identity and require all of the following:

- The PR is still open, targets the recorded base, and its head branch and OID
  still match the delivery identity.
- The final `auto-pr` inventory is green, with no deferred judgement or
  objective finding left open.
- The repository permits the merge under its required reviews, checks, and
  branch protection. Never override a protection rule or force a merge.

An explicit `ship` invocation authorizes merging this verified PR. If it is
not mergeable or needs a decision, leave the PR and its branch intact, report
the blocker, and stop. After the merge request succeeds, re-read the PR and
require its server state to be merged before any cleanup.

## Reconcile a browser-merged pull request

GitHub's browser merge and branch deletion do not modify a local checkout. On
an explicit later `ship cleanup` request, do not publish, merge, or inspect a
set of old branches. Instead, use the current local branch as the only cleanup
candidate:

1. Require a clean worktree, then refresh the configured head remote and base
   remote with pruning.
2. Resolve exactly one merged PR for that branch on the configured host. Its
   recorded head OID must equal the live local branch OID; an absent, ambiguous,
   open, or mismatched PR preserves the branch and stops the workflow.
3. Treat that verified merged PR and branch as the delivery identity, then run
   the local return-and-cleanup sequence below. Do not infer candidates from
   names such as `ship/*`.

## Return the local checkout to its default branch

Clean up only after confirmed merge. Resolve the [branch cleanup](../clean-branches/SKILL.md)
contract and apply these additional bounds:

1. Require a clean working tree and confirm that the recorded head branch is
   not checked out by another linked worktree. If either check fails, preserve
   the branch and report the required manual action.
2. Switch to `DEFAULT_BRANCH`, then fast-forward it from the recorded base
   remote. Do not reset, discard local changes, or rewrite history.
3. Re-read the local head branch. Delete it only if its live OID still equals
   the recorded PR head OID and it classifies as merged into the refreshed
   default branch. Use the cleanup helper's exact-OID local deletion path.
4. Do not enumerate or remove older `ship/*` branches, remote branches, or any
   other refs as a side effect. Remote head-branch deletion remains the
   repository's GitHub policy unless the user explicitly asks for it.

If the switch, fast-forward, or deletion is refused, stop at that point and
report the current branch plus the preserved ref. A completed result must state
that the local checkout is on `DEFAULT_BRANCH`, whether the active local head
branch was removed, and the merged PR URL.
