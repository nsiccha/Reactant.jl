using Reactant, StaticArrays, Enzyme, LinearAlgebra, Test

function retained_static_binary_power(B, count)
    remaining = copy(count)
    result = one(B)
    @trace while remaining > 0
        @trace result = if isodd(remaining)
            result * B
        else
            result
        end
        remaining = div(remaining, 2)
        @trace B = if remaining > 0
            B * B
        else
            B
        end
    end
    return result
end

function static_power_and_input(q, count, ::Val{N}) where {N}
    @allowscalar A = SMatrix{N,N}(q)
    result = retained_static_binary_power(A, count)
    return result, A
end
static_power_cost(q, count, dimension) = sum(first(static_power_and_input(q, count, dimension)))
static_power_gradient(q, count, dimension) = only(Enzyme.gradient(
    Enzyme.Reverse, x -> static_power_cost(x, count, dimension), q))

function native_static_power_cost(q, count, ::Val{N}) where {N}
    B = SMatrix{N,N}(q)
    result = one(B)
    while count > 0
        if isodd(count)
            result *= B
        end
        count = div(count, 2)
        if count > 0
            B *= B
        end
    end
    return sum(result)
end

@testset "Immutable static power seed, value semantics and ordinary reverse" begin
    for T in (Float32, Float64), N in (2, 4)
        host_q = T.(vec(0.8Matrix(I, N, N) + 0.03reshape(sin.(1:N^2), N, N)))
        q = Reactant.to_rarray(host_q)
        count = Reactant.to_rarray(Int64(3); track_numbers=true)
        value = @compile static_power_and_input(q, count, Val(N))
        gradient = @compile static_power_gradient(q, count, Val(N))
        for shift in (T(0), T(0.01)), n in (0, 1, 2, 3, 5, 9, 17)
            host_input = host_q .+ shift
            input = Reactant.to_rarray(host_input)
            runtime_n = Reactant.to_rarray(Int64(n); track_numbers=true)
            actual, after = value(input, runtime_n, Val(N))
            expected = SMatrix{N,N}(host_input)^n
            @test Array(actual) ≈ Array(expected)
            @test Array(after) ≈ reshape(host_input, N, N)
            @test Array(gradient(input, runtime_n, Val(N))) ≈ only(Enzyme.gradient(
                Enzyme.Reverse, x -> native_static_power_cost(x, n, Val(N)), host_input))
            @test Array(input) == host_input
        end
    end
end
