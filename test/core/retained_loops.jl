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
