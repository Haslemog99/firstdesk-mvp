#!/usr/bin/env python3
"""Every career in app.CAREER_CONFIGS needs a scenario-engine tool palette.

Without one, ScenarioExtractor.extract() returns None before it even calls the
LLM, so /api/scenario/from-jd answers 422 for every JD classified into that
career. Both files are read via AST, so this runs with no deps, env or DB.
"""
import ast
import sys


def literal(path: str, name: str):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    return None


careers = literal("app.py", "CAREER_CONFIGS")
palette = literal("scenario_engine.py", "CAREER_TOOL_PALETTE")
channels = literal("scenario_engine.py", "DEFAULT_SLACK_CHANNELS")
valid = literal("scenario_engine.py", "VALID_CAREERS")

if not careers or palette is None or channels is None or valid is None:
    print("PREREQ — could not read CAREER_CONFIGS / palette tables")
    sys.exit(2)

problems = []
for c in sorted(careers):
    if not palette.get(c):
        problems.append(f"{c}: no CAREER_TOOL_PALETTE entry")
    if not channels.get(c):
        problems.append(f"{c}: no DEFAULT_SLACK_CHANNELS entry")
    if c not in valid:
        problems.append(f"{c}: missing from VALID_CAREERS")

if problems:
    for p in problems:
        print("FAIL —", p)
    sys.exit(1)

print(f"OK  {len(careers)} careers have a palette, channels and a VALID_CAREERS entry")
