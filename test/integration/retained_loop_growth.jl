using Reactant, Enzyme, Test
const IR = Reactant.MLIR.IR

function nested_runtime_product(q, counts)
    @allowscalar b = q[1]
    result = 1.0
    @trace for i in eachindex(counts)
        @allowscalar remaining = copy(counts[i])
        @trace while remaining > 0
            result = result * b
            remaining = remaining >> 1
        end
    end
    return result
end

nested_runtime_gradient(q, counts) =
    only(Enzyme.gradient(Enzyme.Reverse, x -> nested_runtime_product(x, counts), q))

function native_nested_product(q, counts)
    result = 1.0
    for count in counts
        while count > 0
            result *= q[1]
            count = count >> 1
        end
    end
    return result
end

function complete_operation_inventory!(counts, op)
    name = IR.name(op)
    counts[name] = get(counts, name, 0) + 1
    for region in op, block in region, child in block
        complete_operation_inventory!(counts, child)
    end
    return counts
end

@testset "Growing data axes retain nested primal and reverse loops" begin
    primal_inventories = Dict{String,Int}[]
    reverse_inventories = Dict{String,Int}[]
    for n in (3, 6, 13)
        q = Reactant.to_rarray([1.01])
        counts = Reactant.to_rarray(fill(Int64(5), n))
        value_exe = @compile nested_runtime_product(q, counts)
        gradient_exe = @compile nested_runtime_gradient(q, counts)
        for host_q in ([1.01], [0.97]), host_counts in
            (fill(Int64(5), n), Int64[mod(i, 5) == 0 ? 0 : 2i + 1 for i in 1:n])
            runtime_q = Reactant.to_rarray(host_q)
            runtime_counts = Reactant.to_rarray(host_counts)
            @test Float64(value_exe(runtime_q, runtime_counts)) ≈
                native_nested_product(host_q, host_counts)
            native_gradient = only(Enzyme.gradient(
                Enzyme.Reverse, x -> native_nested_product(x, host_counts), host_q))
            @test Array(gradient_exe(runtime_q, runtime_counts)) ≈ native_gradient
            @test Array(runtime_q) == host_q
            @test Array(runtime_counts) == host_counts
        end
        for (f, inventories, label) in
            ((nested_runtime_product, primal_inventories, "primal"),
             (nested_runtime_gradient, reverse_inventories, "reverse"))
            inventory = IR.@dispose ctx = Reactant.ReactantContext() begin
                mod = Reactant.Compiler.code_hlo(ctx, f, (q, counts))
                try
                    complete_operation_inventory!(Dict{String,Int}(), IR.Operation(mod))
                finally
                    IR.dispose(mod)
                end
            end
            println("Complete default ", label, " inventory n=", n, ": ",
                sort!(collect(inventory)))
            push!(inventories, inventory)
        end
    end
    for inventories in (primal_inventories, reverse_inventories)
        # Shape bounds remain constants. CSE can share (n - 1) with an existing
        # structural literal, e.g. 2 at n=3. Report the complete inventories
        # above, then require every body/control operation count to stay fixed.
        body_inventories = map(inventories) do inventory
            Dict(k => v for (k, v) in inventory if k != "stablehlo.constant")
        end
        @test all(==(first(body_inventories)), body_inventories)
        @test get(first(inventories), "stablehlo.while", 0) > 0
    end
end
