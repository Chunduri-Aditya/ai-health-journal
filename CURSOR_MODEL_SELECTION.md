# Cursor Model Selection Guide

Choose the right class of model for the job to balance speed, reasoning depth, and token cost. Use this when improving or working on any project under `~/Desktop/Projects/`.

| Task Complexity / Type | Recommended Model Class | Specific Choice | Why |
| :--- | :--- | :--- | :--- |
| **Complex Architecture / Deep Reasoning** | Frontier / Thinking | `Claude Opus` / `GPT-5.x` | Infers intent, plans multi-file changes, handles ambiguous requirements. |
| **Day-to-Day Coding / Refactoring** | Balanced / Flagship | `Claude Sonnet` / `Grok` | Strong instruction following, large context, concise code output. |
| **Fast Iteration / UI & Vibe Coding** | Speed-Optimized | `Cursor Composer` | Extremely fast inline/multiline generation for layout and routine edits. |
| **Boilerplate / Tests / Commit Messages** | Economical / Fast | `Gemini Flash` / `Claude Haiku` | Cost-efficient and rapid for repetitive, low-complexity tasks. |
| **Default / Unsure Tasks** | Native Router | `Auto` (`Balance` or `Intelligence`) | Cursor routes by task type and complexity; default when unsure. |

## Quick Rules of Thumb

- **Plan Mode:** Use frontier reasoning (`Claude Opus` or equivalent) during planning (`Shift + Tab` in Agent mode) before executing code.
- **Execution:** Switch to daily drivers (`Composer` or `Sonnet`) once the implementation steps and file pattern are clear.
- **Unsure:** Stay on `Auto` → `Balance` for ordinary work; use `Auto` → `Intelligence` for harder multi-step portfolio improvements.
- **Do not** leave a frontier model as the default for every request — most project improvements do not need it.

## Project Work Mapping

| Work in this folder | Prefer |
| :--- | :--- |
| Architecture, deploy plans, security/auth, multi-repo strategy | Frontier / Plan Mode |
| Feature implementation, refactors, tests, docs, SHOWCASE updates | Sonnet / Composer / Auto Balance |
| Small copy edits, renames, commit messages, boilerplate | Flash / Haiku / Composer |

## References

- [Cursor Model Routing](https://cursor.com/guides/model-routing)
- [Cursor Router docs](https://cursor.com/docs/cursor-router)
