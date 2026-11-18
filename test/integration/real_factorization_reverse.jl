using Reactant, LinearAlgebra, Enzyme, Test

real_lu_parts(A) = Reactant.Ops.lu(A)
real_solve_cost(A, B) = sum(Reactant.TracedLinearAlgebra._lu_solve_math(A, B))
real_solve_grad(A, B) = Enzyme.gradient(Enzyme.Reverse, real_solve_cost, A, B)

@testset "Real pivoted LU value, pivots and singular info" begin
    for (m, n) in ((3, 3), (4, 3), (3, 4))
        A = reshape(sin.(1:(m*n)), m, n)
        A[1,1] = 0.01
        factors, pivots, permutation, info = Reactant.TracedLinearAlgebra._lu_math(A, Int32)
        F = Array(factors)
        p = Int.(Array(permutation))
        k = min(m, n)
        L = tril(F[:,1:k], -1)
        L[diagind(L)] .= 1
        U = triu(F[1:k,:])
        @test L*U ≈ A[p,:]
        @test Int.(Array(pivots)) == lu(A; check=false).ipiv
        @test Int32(info) == 0
    end
    A = [0.0 1.0; 0.0 2.0]
    _, _, _, info = Reactant.TracedLinearAlgebra._lu_math(A, Int32)
    @test Int32(info) == 1
end

@testset "Ordinary real solve reverse through row pivot changes" begin
    A = [0.1 2.0 0.3; 3.0 0.5 0.2; 0.7 0.4 4.0]
    B = reshape(sin.(1:6), 3, 2)
    initial_A, initial_B = Reactant.to_rarray(A), Reactant.to_rarray(B)
    exe = @compile real_solve_grad(initial_A, initial_B)
    for scale in (0.5, 1.0, 2.0)
        input_A = A + scale*I
        input_B = B .* scale
        dA, dB = exe(Reactant.to_rarray(input_A), Reactant.to_rarray(input_B))
        X = input_A \ input_B
        expected_B = transpose(input_A) \ ones(size(X))
        @test Array(dB) ≈ expected_B rtol=1e-9
        @test Array(dA) ≈ -expected_B*transpose(X) rtol=1e-9
    end
end

function triangular_options(A, B, options)
    left, lower, transposed, unit = options
    return Reactant.Ops.triangular_solve(
        A, B; left_side=left, lower, transpose_a=transposed ? 'T' : 'N',
        unit_diagonal=unit,
    )
end

@testset "Triangular solve orientations and unit diagonal" begin
    A = [2.0 0.3 0.5; 0.2 3.0 0.7; 0.4 0.1 4.0]
    for left in (false, true), lower in (false, true), transposed in (false, true),
        unit in (false, true)
        B = left ? reshape(sin.(1:6), 3, 2) : reshape(sin.(1:6), 2, 3)
        triangle = lower ? (unit ? UnitLowerTriangular(A) : LowerTriangular(A)) :
                           (unit ? UnitUpperTriangular(A) : UpperTriangular(A))
        matrix = transposed ? transpose(triangle) : triangle
        expected = left ? matrix \ B : B / matrix
        options = (left, lower, transposed, unit)
        actual = @jit triangular_options(Reactant.to_rarray(A), Reactant.to_rarray(B), options)
        @test Array(actual) ≈ expected rtol=1e-10
    end
end

@testset "Batched real mathematical solve and default reverse" begin
    A = reshape(cos.(1:18), 3, 3, 2)
    A[:,:,1] += 4I
    A[:,:,2] += 5I
    B = reshape(sin.(1:12), 3, 2, 2)
    function cost(A, B)
        permutation = [3, 1, 2]
        a = Reactant.Ops.transpose(A, permutation)
        b = Reactant.Ops.transpose(B, permutation)
        values = only(Reactant.Ops.batch(
            Reactant.TracedLinearAlgebra._lu_solve_math, [a,b], [size(A,3)]
        ))
        return sum(values)
    end
    gradient(A, B) = Enzyme.gradient(Enzyme.Reverse, cost, A, B)
    dA, dB = @jit gradient(Reactant.to_rarray(A), Reactant.to_rarray(B))
    for i in 1:2
        X = A[:,:,i] \ B[:,:,i]
        expected_B = transpose(A[:,:,i]) \ ones(size(X))
        @test Array(dB)[:,:,i] ≈ expected_B rtol=1e-9
        @test Array(dA)[:,:,i] ≈ -expected_B*transpose(X) rtol=1e-9
    end
end
