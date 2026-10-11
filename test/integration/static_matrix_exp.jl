using Reactant, StaticArrays, LinearAlgebra, Enzyme, Test

function static_matrix_exp(A, ::Val{N}) where {N}
    @allowscalar S = SMatrix{N,N}(A)
    return exp(S)
end

static_exp_cost(A, n) = sum(static_matrix_exp(A, n))
static_exp_gradient(A, n) = only(Enzyme.gradient(Enzyme.Reverse, x -> static_exp_cost(x, n), A))
native_static_exp_cost(A) = sum(exp(SMatrix{size(A,1),size(A,2)}(A)))

# Independent dense block-exponential oracle for the Fréchet adjoint of sum(exp(A)).
# This is a test oracle, not a differentiation rule for the implementation.
function dense_exp_gradient(A)
    n = size(A, 1)
    Z = zeros(eltype(A), 2n, 2n)
    Z[1:n, 1:n] .= transpose(A)
    Z[(n+1):end, (n+1):end] .= transpose(A)
    Z[1:n, (n+1):end] .= one(eltype(A))
    return exp(Z)[1:n, (n+1):end]
end

@testset "Real static exponential values and ordinary reverse" begin
    for T in (Float32, Float64), n in (1, 2, 3, 4, 6)
        println("Compiling real static exponential and default reverse: T=", T, " N=", n)
        flush(stdout)
        base = T.(reshape(sin.(1:(n*n)), n, n))
        base ./= opnorm(base, 1)
        initial = Reactant.to_rarray(base)
        value_exe = @compile static_matrix_exp(initial, Val(n))
        grad_exe = @compile static_exp_gradient(initial, Val(n))
        @testset "T=$T N=$n, reuse across approximation branches" begin
            for magnitude in (0.005, 0.1, 0.5, 1.5, 3.0, 20.0)
                A = base .* T(magnitude)
                actual = T.(value_exe(Reactant.to_rarray(A), Val(n)))
                expected_grad = dense_exp_gradient(A)
                actual_grad = Array(grad_exe(Reactant.to_rarray(A), Val(n)))
                rtol = T === Float32 ? 2e-4 : 1e-9
                atol = T === Float32 ? 2e-5 : 1e-11
                @test actual ≈ exp(A) rtol=rtol atol=atol
                @test actual_grad ≈ expected_grad rtol=rtol atol=atol
            end
        end
    end
end

@testset "All real 2×2 discriminant branches" begin
    initial = Reactant.to_rarray([0.3 1.2; 0.0 0.3])
    value_exe = @compile static_matrix_exp(initial, Val(2))
    grad_exe = @compile static_exp_gradient(initial, Val(2))
    for A in ([0.3 1.2; 0.0 0.3], [0.3 0.0; 0.0 0.3],
              [0.3 1.2; -0.7 0.3], [0.3 1.2; 0.7 0.3],
              [-100.0 1.0; 0.0 -101.0])
        @test Float64.(value_exe(Reactant.to_rarray(A), Val(2))) ≈ exp(A) atol=1e-12
        actual = Array(grad_exe(Reactant.to_rarray(A), Val(2)))
        @test actual ≈ dense_exp_gradient(A) rtol=1e-9 atol=1e-11
    end
    A = [0.3 1.2; 0.0 0.3]
    native = only(Enzyme.gradient(Enzyme.Reverse, native_static_exp_cost, A))
    @test native ≈ dense_exp_gradient(A) rtol=1e-9
    @test native[2,1] ≈ exp(0.3)*(1 + 1.2 + 1.2^2/6)
end

function integer_static_exp2(A)
    @allowscalar S = SMatrix{2,2}(A)
    return exp(S)
end

@testset "Real integer 2x2 exponential promotion" begin
    for I in (Int32, Int64)
        A = I[-100000 1; 0 0]
        a = Reactant.to_rarray(A)
        f = @compile integer_static_exp2(a)
        @test Float64.(f(a)) ≈ [0.0 1e-5; 0.0 1.0] rtol=1e-12
    end
    A = Int64[typemin(Int64) 0; 0 0]
    a = Reactant.to_rarray(A)
    @test Float64.(@jit integer_static_exp2(a)) == [0.0 0.0; 0.0 1.0]
end
