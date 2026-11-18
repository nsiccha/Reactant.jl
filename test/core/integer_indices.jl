using Reactant, Test

function integer_width_indices(A, i)
    @allowscalar return A[i], A[i, :], A[i, i]
end

@testset "Scalar index widths preserve scalar and sliced dimensions" begin
    A = reshape(collect(1.0:9.0), 3, 3)
    for T in (Int32, Int64, UInt32, UInt64)
        i = Reactant.to_rarray(T(2))
        exe = @compile integer_width_indices(Reactant.to_rarray(A), i)
        scalar, row, diagonal = exe(Reactant.to_rarray(A), i)
        @test Float64(scalar) == A[2]
        @test Array(row) == A[2, :]
        @test Float64(diagonal) == A[2, 2]
    end
end
