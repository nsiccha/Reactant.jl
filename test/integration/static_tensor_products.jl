using Reactant, StaticArrays, Enzyme, Test

function static_products(q)
    @allowscalar begin
        A = SMatrix{2,3}(q[1:6])
        B = SMatrix{3,2}(q[7:12])
        v = SVector{3}(q[13:15])
    end
    fixed = SMatrix{3,2}(cos.(1:6))
    return (A * B, A * v, A * fixed, transpose(fixed) * B)
end

static_product_cost(q) = sum(sum, static_products(q))
static_product_gradient(q) = only(Enzyme.gradient(Enzyme.Reverse, static_product_cost, q))

function native_static_product_cost(q)
    A = SMatrix{2,3}(q[1:6])
    B = SMatrix{3,2}(q[7:12])
    v = SVector{3}(q[13:15])
    fixed = SMatrix{3,2}(cos.(1:6))
    return sum(A * B) + sum(A * v) + sum(A * fixed) + sum(transpose(fixed) * B)
end

@testset "Static contractions preserve wrappers, promotion and ordinary reverse" begin
    for T in (Float32, Float64)
        host_q = T.(sin.(1:15))
        q = Reactant.to_rarray(host_q)
        values = @compile static_products(q)
        gradient = @compile static_product_gradient(q)
        for shift in (T(0), T(0.2))
            input = host_q .+ shift
            actual = values(Reactant.to_rarray(input))
            expected = static_products(input)
            for (a, e) in zip(actual, expected)
                @test a isa StaticArray
                @test Reactant.unwrapped_eltype(a) == eltype(e)
                @test Array(a) ≈ Array(e)
            end
            @test Array(gradient(Reactant.to_rarray(input))) ≈
                only(Enzyme.gradient(Enzyme.Reverse, native_static_product_cost, input))
            @test Array(q) == host_q
        end
    end
end
