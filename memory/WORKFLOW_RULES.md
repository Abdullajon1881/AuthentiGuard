# Workflow Rules for AI Sessions

## Before Any Coding

1. Read ALL memory files in `memory/` directory
2. Verify your planned changes align with `01_architecture_rules.md`
3. Check `12_anti_patterns.md` for things to avoid
4. Check `07_protected_files.md` before modifying load-bearing files
5. Check `11_known_issues.md` for relevant blockers

## During Work

- Follow detector contract in `06_detector_contract.md`
- Follow pipeline flow in `04_pipeline_map.md`
- Log significant decisions in `10_decision_log.md`
- If you discover a new issue, add it to `11_known_issues.md`
- If you discover a new anti-pattern, add it to `12_anti_patterns.md`

## After Any Task

1. Update `20_current_focus.md` with current state
2. Update `21_current_tasks.md` — mark completed, add new
3. Update `22_next_tasks.md` if priorities changed
4. Add session entry to `99_session_summary.md`
5. Update `10_decision_log.md` with any decisions made

## Memory File Rules

- All files MUST be under 200 lines
- Use bullet points, not prose
- No marketing language — actionable facts only
- Memory files override assumptions from training data
- When in doubt, read the actual source code to verify memory accuracy

## File Naming Convention

- `0X_` = Tier 0 (project identity) — read every session
- `1X_` = Tier 1 (decision history) — read when planning changes
- `2X_` = Tier 2 (current state) — read when starting work
- `3X_` = Tier 3 (long-term) — read when planning roadmap
- `99_` = Session log
- `WORKFLOW_RULES.md` = This file (meta)
