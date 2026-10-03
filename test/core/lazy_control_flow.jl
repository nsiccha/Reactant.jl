using Reactant, Enzyme, Test

@testset "Nested textual pipelines retain the low-level runner contract" begin
    IR = Reactant.MLIR.IR
    Reactant.MLIR.IR.@with_context Reactant.ReactantContext() begin
        mod = parse(
            IR.Module,
            """
module {
    func.func @identity(%arg0: tensor<f64>) -> tensor<f64> {
        return %arg0 : tensor<f64>
    }
}
""",
        )
        try
            for pipeline in (
                "canonicalize,enzyme-batch,canonicalize",
                "builtin.module(canonicalize,enzyme-batch,canonicalize)",
            )
                @test Reactant.Compiler.run_pass_pipeline!(mod, pipeline) === mod
                @test IR.verifyall(IR.Operation(mod))
            end
        finally
            IR.dispose(mod)
        end
    end
    identity_array(x) = x
    input = Reactant.to_rarray([1.0, 2.0])
    compiled = Reactant.@compile optimize = "builtin.module(inline,canonicalize,enzyme-batch,canonicalize)" identity_array(
        input
    )
    @test Array(compiled(input)) == [1.0, 2.0]
end

@testset "Shared integer expressions have bounded evaluation cost" begin
    IR = Reactant.MLIR.IR
    Reactant.MLIR.IR.@with_context Reactant.ReactantContext() begin
        expressions = ["%v0 = stablehlo.constant dense<0> : tensor<i64>"]
        for i in 1:40
            push!(expressions, "%v$i = stablehlo.add %v$(i - 1), %v$(i - 1) : tensor<i64>")
        end
        source = """
            module {
                func.func @expressions(%arg0: tensor<i64>) -> tensor<i64> {
                    $(join(expressions, '\n'))
                    %max = stablehlo.constant dense<9223372036854775807> : tensor<i64>
                    %one = stablehlo.constant dense<1> : tensor<i64>
                    %overflow = stablehlo.add %max, %one : tensor<i64>
                    %unknown = stablehlo.add %arg0, %arg0 : tensor<i64>
                    %undefined = stablehlo.divide %one, %v0 : tensor<i64>
                    return %v40 : tensor<i64>
                }
            }
            """
        mod = parse(IR.Module, source)
        try
            @test IR.verifyall(IR.Operation(mod))
            function_op = first(first(first(IR.Operation(mod))))
            block = first(IR.region(function_op, 1))
            operations = collect(block)
            constants = Dict{IR.Value,Any}()
            @test Reactant.Compiler.constant_loop_limit(
                IR.result(operations[41], 1), constants
            ) == 0
            @test length(constants) == 41
            limit = IR.result(operations[19], 1)
            Reactant.Compiler.constant_loop_limit(limit)
            @test (@allocated Reactant.Compiler.constant_loop_limit(limit)) < 8 * 1024^2
            for index in (44, 45, 46)
                value = IR.result(operations[index], 1)
                @test Reactant.Compiler.constant_loop_limit(value, constants) === nothing
                @test haskey(constants, value)
                @test Reactant.Compiler.constant_loop_limit(value, constants) === nothing
            end
        finally
            IR.dispose(mod)
        end
    end
end

function lazy_guard(x, scale)
    result = zero(x)
    Reactant.@trace if scale > 0
        result = log(scale) + x / scale
    else
        result = -2 * one(x)
    end
    return result
end

function nested_lazy_guard(x, scale)
    result = zero(x)
    Reactant.@trace if scale > 0
        if scale > 1
            result = log(scale) + x / scale
        else
            result = sqrt(scale) + x / scale
        end
    else
        result = -2 * one(x)
    end
    return result
end

lazy_gradient(f, x, s) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x, s)
lazy_slope(x, s) = last(lazy_gradient(lazy_guard, x, s))
lazy_curvature(x, s) = lazy_gradient(lazy_slope, x, s)

function retained_lazy_loop(x, s)
    total = zero(s)
    Reactant.@trace for i in 1:length(x)
        value = Reactant.@allowscalar x[i]
        total += lazy_guard(value, s)
    end
    return total
end

function lane_guard(x)
    value = Reactant.@allowscalar x[]
    result = zero(value)
    Reactant.@trace if value > 0
        result = log(value)
    end
    return result
end
lane_batch(x) = sum(only(Reactant.Ops.batch(lane_guard, [x], Int64[length(x)])))
lane_gradient(x) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(lane_batch), x))
traced(x::Number) = Reactant.to_rarray(x; track_numbers=true)
traced(x) = Reactant.to_rarray(x)
host(x::Tuple) = map(host, x)
host(x::Reactant.AbstractConcreteArray) = Array(x)
host(x) = Reactant.to_number(x)

function complete_inventory(hlo)
    assignments = collect(eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = ", hlo))
    matches = collect(
        eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo)
    )
    @test length(assignments) == length(matches)
    counts = Dict{String,Int}()
    for m in matches
        name = m.captures[1]
        counts[name] = get(counts, name, 0) + 1
    end
    return counts
end

# Follow calls and fusions from the executable entry, carrying the branch
# context. Counting conditionals alone would miss speculated arithmetic.
function unguarded_lazy_arithmetic(hlo)
    computations = Dict{String,Vector{Tuple{String,Vector{Tuple{String,String}}}}}()
    entry = nothing
    current = nothing
    for line in split(hlo, '\n')
        header = match(r"^(ENTRY )?%?([\w.-]+) \(.*\) -> .* \{$", line)
        if header !== nothing
            current = String(header.captures[2])
            computations[current] = Tuple{String,Vector{Tuple{String,String}}}[]
            header.captures[1] === nothing || (entry = current)
        elseif occursin(r"^\s*(?:ROOT\s+)?%?[\w.-]+ = ", line)
            instruction = match(
                r"^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", line
            )
            @assert instruction !== nothing && current !== nothing
            callees = [
                (String(m.captures[1]), String(m.captures[2])) for m in eachmatch(
                    r"\b(calls|to_apply|true_computation|false_computation|condition|body)=%?([\w.-]+)",
                    line,
                )
            ]
            push!(computations[current], (instruction.captures[1], callees))
        elseif line == "}"
            current = nothing
        end
    end
    @assert entry !== nothing
    pending = [(entry, false)]
    visited = Set{Tuple{String,Bool}}()
    unguarded = Set{String}()
    while !isempty(pending)
        name, guarded = pop!(pending)
        (name, guarded) in visited && continue
        push!(visited, (name, guarded))
        for (operation, callees) in computations[name]
            if !guarded && operation in ("log", "divide", "sqrt", "rsqrt")
                push!(unguarded, name * ":" * operation)
            end
            for (attribute, callee) in callees
                branch =
                    operation == "conditional" &&
                    attribute in ("true_computation", "false_computation")
                push!(pending, (callee, guarded || branch))
            end
        end
    end
    return unguarded
end

function inspect(compiled, label)
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    if haskey(ENV, "LAZY_CONTROL_FLOW_OUTPUT")
        mkpath(ENV["LAZY_CONTROL_FLOW_OUTPUT"])
        write(joinpath(ENV["LAZY_CONTROL_FLOW_OUTPUT"], label * ".executable.hlo"), hlo)
        write(
            joinpath(ENV["LAZY_CONTROL_FLOW_OUTPUT"], label * ".serialized.mlir"),
            compiled.module_string,
        )
    end
    counts = complete_inventory(hlo)
    @test isempty(unguarded_lazy_arithmetic(hlo))
    println(label, " complete inventory: ", counts)
    return counts
end

@testset "Default scalar and nested lazy control flow" begin
    for T in (Float32, Float64), f in (lazy_guard, nested_lazy_guard)
        x, s = traced.((T(0.5), T(2)))
        reverse(a, b) = lazy_gradient(f, a, b)
        for (direction, fn) in (("primal", f), ("reverse", reverse))
            compiled = Reactant.@compile serializable = true fn(x, s)
            counts = inspect(compiled, "$(T)-$(f)-$direction")
            @test get(counts, "conditional", 0) >= (f === lazy_guard ? 1 : 2)
            @test occursin("stablehlo.optimization_barrier", compiled.module_string)
            for a in (T(0.5), T(1.2)), b in T.((2, 0.7, -1, 0, NaN))
                actual = host(compiled(traced(a), traced(b)))
                expected = fn(a, b)
                if actual isa Tuple
                    @test all(
                        isapprox.(actual, expected; rtol=T === Float32 ? 2e-6 : 1e-12)
                    )
                else
                    @test actual ≈ expected rtol = T === Float32 ? 2e-6 : 1e-12
                end
            end
        end
    end
end

@testset "Ordinary second derivatives through lazy guards" begin
    for T in (Float32, Float64)
        x, s = traced.((T(0.5), T(2)))
        compiled = Reactant.@compile serializable = true lazy_curvature(x, s)
        counts = inspect(compiled, "$(T)-second-derivative")
        @test get(counts, "conditional", 0) > 0
        for a in (T(0.5), T(1.2)), b in T.((2, 0.7, -1, 0, NaN))
            expected =
                b > 0 ? (-inv(b * b), -inv(b * b) + 2a / (b * b * b)) : (zero(T), zero(T))
            @test all(
                isapprox.(
                    host(compiled(traced(a), traced(b))),
                    expected;
                    rtol=T === Float32 ? 3e-6 : 1e-12,
                ),
            )
        end
    end
end

@testset "Default retained scalar loops and lane-dependent guards" begin
    previous_loop = nothing
    previous_batch = nothing
    for n in (3, 9, 15)
        values = [iseven(i) ? -1.0 : 2.0 for i in 1:n]
        x = traced(values)
        s = traced(2.0)
        reverse(a, b) = lazy_gradient(retained_lazy_loop, a, b)
        pair = []
        for (direction, fn) in (("primal", retained_lazy_loop), ("reverse", reverse))
            compiled = Reactant.@compile serializable = true fn(x, s)
            counts = inspect(compiled, "loop-$n-$direction")
            @test get(counts, "while", 0) > 0
            @test get(counts, "conditional", 0) > 0
            for scale in (2.0, 0.7, -1.0, 0.0, NaN)
                actual = host(compiled(x, traced(scale)))
                if direction == "primal"
                    expected = scale > 0 ? n * log(scale) + sum(values) / scale : -2n
                    @test actual ≈ expected
                else
                    dx, ds = actual
                    @test dx ≈ (scale > 0 ? fill(inv(scale), n) : zeros(n))
                    @test ds ≈ (scale > 0 ? n / scale - sum(values) / (scale * scale) : 0.0)
                end
            end
            push!(pair, counts)
        end
        previous_loop === nothing || @test pair == previous_loop
        previous_loop = pair
        batchpair = []
        for (direction, fn) in (("primal", lane_batch), ("reverse", lane_gradient))
            compiled = Reactant.@compile serializable = true fn(x)
            counts = inspect(compiled, "batch-$n-$direction")
            @test get(counts, "while", 0) > 0
            @test get(counts, "conditional", 0) > 0
            expected = if direction == "primal"
                count(>(0), values) * log(2.0)
            else
                [v > 0 ? inv(v) : 0.0 for v in values]
            end
            @test host(compiled(x)) ≈ expected
            changed = [iseven(i) ? NaN : 0.7 for i in 1:n]
            expected_changed = if direction == "primal"
                count(isodd, 1:n) * log(0.7)
            else
                [isodd(i) ? inv(0.7) : 0.0 for i in 1:n]
            end
            @test host(compiled(traced(changed))) ≈ expected_changed
            push!(batchpair, counts)
        end
        previous_batch === nothing || @test batchpair == previous_batch
        previous_batch = batchpair
    end
end
