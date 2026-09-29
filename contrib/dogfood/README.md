# Opt-in integration branch

`nsiccha/Reactant.jl:dogfood` combines selected proposed changes with upstream
`main` for RK development and representative CPU workloads. Normal consumers
continue using upstream releases. The source set is recorded in `stack.json`;
the compiler dependency pins in `deps/ReactantExtra/WORKSPACE` and the companion
Enzyme-JAX fork identify the matching Enzyme-JAX and Enzyme sources.

## Use a tested revision

Use an isolated Julia project and pin an exact Reactant commit. This branch does
not change the registered Reactant_jll artifact: installing the Julia source
alone does not load the proposed compiler changes. Build or obtain the matching
`libReactantExtra.so`, record its SHA256 with the four source commits, and set
`[Reactant_jll] libReactantExtra_path` in that project's `LocalPreferences.toml`.
Reactant_jll must be a direct dependency of that project for the preference to
apply. Follow the repository's `docs/src/tutorials/local-build.md` for a local
build. Verify the actual loaded path with `Libdl.dllist()` before interpreting
any results. Do not copy the preference into a global Julia environment.

Keep the previous passing project, exact commit, library and receipt together.
Rollback selects that project and library. These local builds do not satisfy
RK's separate released-toolchain gates for enabling product defaults.

## Refresh and promote

1. Fetch each upstream `main` and the selected proposal branches. Record exact
   commits. Check closed or merged proposals before carrying them forward.
2. Create candidate branches from the previous dogfood tips, then merge
   upstream and selected inputs without rewriting their original history.
   Update dependency pins from Enzyme through Enzyme-JAX to Reactant and refresh
   `stack.json`. Keep incompatible alternatives in separate candidates.
3. Build the compiler tools and matching library from those exact sources.
   Run the full Enzyme-JAX lit suite, custom-rule MLIR/HLO regressions, Julia
   frontend and independent gradient checks, plus representative linear-algebra,
   GP and PK/PD workloads. Record failures and timings as well as successes.
4. Promote only a passing candidate with ordinary fast-forward pushes, in
   dependency order. Record the exact four commits and library SHA256 in the
   acceptance receipt. Preserve the previous passing receipt and artifacts.
   A moving branch name is not a reproducible dependency pin.

Refreshes are currently explicit maintenance runs. No unattended merge, build,
publish schedule or standing approval is installed. New source revisions need
new validation before promotion. The original proposal branches stay separate
so upstream PRs remain independently reviewable.

## Native Enzyme.jl

The opt-in project also pins Enzyme.jl and its bundled EnzymeCore to the exact
Enzyme.jl source commit in `stack.json`. Its tuple-field repair is part of the
candidate. It was tested separately on Julia 1.10 and 1.12, but the current
integration with upstream main has not been retested. Its known quadratic
dominance calculation must be replaced and revalidated before promotion.

Enzyme.jl uses the native library from Enzyme_jll; it does not use Reactant's
`libReactantExtra.so`. Start with the repair's tested Enzyme_jll 0.0.298+0 and
record `Enzyme_jll.libEnzyme_path`, the library hash, Julia version and LLVM
version separately. The shared source repository named Enzyme contains both
native and MLIR code, but a Reactant MLIR build is not a native Julia LLVM
plugin. Any later native-library override must be built against Julia's LLVM
and verified independently, following Enzyme.jl's `docs/src/dev_docs.md`.
