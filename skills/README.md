# A Claude Code skill for SpatialEconometrics.jl

`spatial-econometrics/` is a [skill](https://docs.claude.com/en/docs/claude-code/skills)
for Claude Code: instructions and reference files that Claude loads when a
conversation turns to spatial econometrics, so that it uses the package the
way its documentation intends. It contains:

- `SKILL.md`: installation, the analysis workflow, and the conventions that
  fail silently (panel stacking, row order of `W`, intercepts);
- `references/choosing-a-model.md`: from the data and the question to an
  entry point;
- `references/weights.md`: building and checking `W`, and its pitfalls;
- `references/warnings-and-errors.md`: what each warning or error about the
  data or model means, and what may still be reported;
- `references/api/`: the docstring of every exported name, one file per
  submodule, generated from the source.

It needs Julia 1.10 or later on the machine where Claude Code runs. The
skill tells Claude how to install the package if it is missing.

## Installing it

Copy or link the folder into a skills directory: `~/.claude/skills/` for
every project, or `.claude/skills/` inside one project.

```bash
# macOS / Linux, from a clone of this repository
ln -s "$PWD/skills/spatial-econometrics" ~/.claude/skills/spatial-econometrics
```

```bat
:: Windows (cmd), from a clone of this repository
mklink /J "%USERPROFILE%\.claude\skills\spatial-econometrics" "%CD%\skills\spatial-econometrics"
```

A link keeps the skill current when the clone is pulled; a copy does not.
Claude Code picks it up in the next session. Ask, for example, "fit a
spatial lag model to this county data and tell me what the spillovers are".

## Keeping the function reference in sync

`references/api/` is generated; do not edit it by hand. After changing a
docstring, regenerate it:

```bash
julia --project=. skills/build_reference.jl
```

Testset 45 of `test/runtests.jl` fails when the files differ from the
docstrings (`julia --project=. skills/build_reference.jl --check` runs the
same check alone). The hand-written files are not checked against the
source: when a fit's behaviour or a message changes, update them too.
