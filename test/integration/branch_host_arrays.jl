using Reactant, Enzyme, Test

function choose_host_array(pred, yes, no, args)
    Reactant.@trace result = if pred
        yes(args...)
    else
        no(args...)
    end
    return result
end

function captured_host_array_cost(q, ::Type{T}) where {T}
    coefficients = T[2, 3, 4]
    Reactant.@allowscalar begin
        x = q[1]
        pred = q[2] > 0
    end
    return choose_host_array(
        pred, () -> x * coefficients[1], () -> x * coefficients[2], ()
    )
end

function reference_host_array_cost(q)
    if q[2] > 0
        return 2q[1]
    end
    return 3q[1]
end

@testset "Read-only captured numeric host array branch paths" begin
    for T in (Float32, Float64, Int64)
        cost = q -> captured_host_array_cost(q, T)
        gradient = q -> only(Enzyme.gradient(Enzyme.Reverse, cost, q))
        q = Reactant.to_rarray([1.2, 1.0])
        value = Reactant.@compile cost(q)
        reverse = Reactant.@compile gradient(q)
        for x in (0.5, 1.2), pred in (-1.0, 1.0)
            host = [x, pred]
            arg = Reactant.to_rarray(host)
            @test Float64(value(arg)) ≈ reference_host_array_cost(host)
            @test Array(reverse(arg)) ≈ only(Enzyme.gradient(Enzyme.Reverse, reference_host_array_cost, host))
            @test Array(arg) == host
        end
    end
end
