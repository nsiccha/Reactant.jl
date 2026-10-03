using Reactant, Enzyme, Test

function retained_diagonal_sum(eta, diagonal, K)
    total = zero(eta)
    @trace track_numbers=false for i in 2:K
        total += (K - i + 2eta - 2) * log(@allowscalar diagonal[i])
    end
    return total
end

function retained_diagonal_loss(u)
    eta = sum(view(u, 1:1))
    K = length(u) - 1
    diagonal = u[2:(K + 1)]
    total = zero(eta)
    @trace if isfinite(eta) && eta > 0
        total = retained_diagonal_sum(eta, diagonal, K)
    end
    return total
end

retained_diagonal_gradient(u) = only(
    Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(retained_diagonal_loss), u)
)

function retained_loop_inventory(hlo)
    counts = Dict{String,Int}()
    for match in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", repr(hlo))
        counts[match.match] = get(counts, match.match, 0) + 1
    end
    return counts
end

function retained_work_inventory(ops)
    simplified = ("stablehlo.constant", "stablehlo.reshape", "stablehlo.add",
        "stablehlo.multiply", "stablehlo.subtract", "stablehlo.negate")
    return Dict(name => count for (name, count) in ops if name ∉ simplified)
end

function retained_executable_inventory(compiled)
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    counts = Dict{String,Int}()
    # Tuple type comments contain '='; match past the complete result type.
    for match in eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo)
        name = match.captures[1]
        counts[name] = get(counts, name, 0) + 1
    end
    return counts
end

function retained_work_outside_loops(compiled)
    module_ = only(Reactant.XLA.get_hlo_modules(compiled.exec))
    counts = Dict{String,Int}()
    visited = Set{Ptr{Cvoid}}()
    function visit(computation)
        computation.ptr in visited && return
        push!(visited, computation.ptr)
        for instruction in computation.instructions
            name = repr(instruction.opcode)
            counts[name] = get(counts, name, 0) + 1
            name == "while" && continue
            if Reactant.XLA.is_fusion_instruction(instruction)
                visit(instruction.fused_instructions_computation)
            elseif Reactant.XLA.has_to_apply(instruction)
                visit(instruction.to_apply)
            end
        end
    end
    GC.@preserve module_ visit(module_.entry_computation)
    return counts
end

function retained_invariant_loss(u)
    x = sum(view(u, 1:1))
    total = zero(x)
    @trace track_numbers=false for i in 1:(length(u) - 1)
        total += log(x)
    end
    return total
end

@testset "Invariant work remains inside retained loops, including zero trips" begin
    for n in (0, 1, 3)
        ru = Reactant.to_rarray([0.8; zeros(n)])
        compiled = @compile retained_invariant_loss(ru)
        ops = retained_executable_inventory(compiled)
        @test get(ops, "while", 0) == 1
        @test get(ops, "log", 0) == 1
        @test get(retained_work_outside_loops(compiled), "log", 0) == 0
        for x in (0.8, 1.1)
            w = [x; zeros(n)]
            rw = Reactant.to_rarray(w)
            @test Float64(compiled(rw)) ≈ n * log(x) rtol=1e-12
            @test Array(rw) == w
        end
        if n == 0
            # The inactive loop body must not evaluate this invalid log.
            rw = Reactant.to_rarray([-0.2])
            @test Float64(compiled(rw)) == 0.0
            @test Array(rw) == [-0.2]
        end
    end
end

@testset "Shape-specialized loops retain default primal and reverse structure" begin
    inventories = []
    for K in (2, 4, 8, 16)
        u = [1.5; fill(0.8, K)]
        ru = Reactant.to_rarray(u)
        primal = @compile retained_diagonal_loss(ru)
        reverse = @compile retained_diagonal_gradient(ru)
        ops = (
            retained_loop_inventory(@code_hlo retained_diagonal_loss(ru)),
            retained_loop_inventory(@code_hlo retained_diagonal_gradient(ru)),
        )
        @test get(ops[1], "stablehlo.while", 0) == 1
        @test get(ops[2], "stablehlo.while", 0) == 2
        println("retained diagonal complete inventory, K=", K, ": ", ops)
        push!(inventories, ops)

        # Inspect the executable after ordinary XLA optimization as well as
        # MLIR: singleton loops and lazy branches can disappear at this stage.
        xla_ops = map(retained_executable_inventory, (primal, reverse))
        println("retained diagonal executable inventory, K=", K, ": ", xla_ops)
        @test get(xla_ops[1], "while", 0) == 1
        @test get(xla_ops[2], "while", 0) == 2
        @test get(xla_ops[1], "conditional", 0) == 1
        @test get(xla_ops[2], "conditional", 0) == 2
        @test all(ops -> get(ops, "log", 0) == 1, xla_ops)

        for w in (u, u .+ 0.03)
            rw = Reactant.to_rarray(w)
            expected = [
                2 * sum(log, w[3:end]);
                0.0;
                [(K - i + 2w[1] - 2) / w[i + 1] for i in 2:K]
            ]
            @test Float64(primal(rw)) ≈ retained_diagonal_loss(w) rtol=1e-12
            @test Array(reverse(rw)) ≈ expected rtol=1e-12
            @test Array(rw) == w
        end
        for eta in (-1.0, 0.0, Inf, NaN)
            w = [eta; fill(-0.2, K)]
            rw = Reactant.to_rarray(w)
            @test Float64(primal(rw)) == 0.0
            @test Array(reverse(rw)) == zeros(K + 1)
            @test isequal(Array(rw), w)
        end
    end
    # Complete inventories stop growing; small dimensions may share constants
    # or simplify scalar identities without replicating the loop's work.
    @test inventories[3] == inventories[4]
    @test allequal([map(retained_work_inventory, pair) for pair in inventories])
end
