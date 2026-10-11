using Reactant, Test

integer_shifts(x, n) = (x << n, x >> n, x >>> n)
integer_shifts_one(x) = integer_shifts(x, 1)
integer_shifts_literal(n) = integer_shifts(Int32(-37), n)

@testset "Integer scalar shifts" begin
    for T in (Int32, Int64, UInt32, UInt64), C in (Int64, UInt64)
        first_value = T <: Signed ? T(-37) : typemax(T) - T(36)
        initial = Reactant.to_rarray(first_value; track_numbers=true)
        count = Reactant.to_rarray(C(1); track_numbers=true)
        exe = @compile integer_shifts(initial, count)
        counts = C <: Signed ?
            (typemin(C), -70, -32, -1, 0, 1, 31, 32, 63, 64, 70, typemax(C)) :
            (C(0), C(1), C(31), C(32), C(63), C(64), C(70), typemax(C))
        for value in (first_value, T(37)), n in counts
            actual = exe(Reactant.to_rarray(value; track_numbers=true),
                         Reactant.to_rarray(n; track_numbers=true))
            @test T.(actual) == integer_shifts(value, n)
        end
        fixed_exe = @compile integer_shifts_one(initial)
        @test T.(fixed_exe(initial)) == integer_shifts(first_value, 1)
    end
    count = Reactant.to_rarray(Int64(1); track_numbers=true)
    literal_exe = @compile integer_shifts_literal(count)
    for n in (-70, -1, 0, 1, 32, 70)
        @test Int32.(literal_exe(Reactant.to_rarray(n; track_numbers=true))) ==
              integer_shifts(Int32(-37), n)
    end
end
