using Reactant, Enzyme, Test

function nested_trace_scopes(p)
    @allowscalar begin
        y = p[1]
        n = ceil(Int, p[2])
        condition = p[3] > 0
    end
    @trace if condition
        @trace for i in 1:n
            @trace for j in 1:i
                y = y*y
            end
        end
    else
        y = -y
    end
    return y
end

function traced_macro_hygiene(p)
    @allowscalar begin
        y = p[1]
        n = ceil(Int, p[2])
    end
    @trace for i in 1:n
        @allowscalar y = y + p[1]
    end
    return y
end

@testset "Nested macros keep defining-module hygiene" begin
    initial = Reactant.to_rarray([2.0, 3.0])
    exe = @compile traced_macro_hygiene(initial)
    for n in (0, 1, 3, 8)
        @test Float64(exe(Reactant.to_rarray([2.0, Float64(n)]))) == 2.0*(n+1)
    end
end

function native_nested_scopes(p)
    y = p[1]
    if p[3] > 0
        for i in 1:ceil(Int, p[2]), j in 1:i
            y = y*y
        end
    else
        y = -y
    end
    return y
end

nested_trace_gradient(p) = only(Enzyme.gradient(Enzyme.Reverse, nested_trace_scopes, p))

@testset "Nested trace scopes and default dynamic reverse" begin
    initial = Reactant.to_rarray([1.001, 2.0, 1.0])
    value_exe = @compile nested_trace_scopes(initial)
    grad_exe = @compile nested_trace_gradient(initial)
    for count in (0, 1, 2, 3), condition in (-1.0, 1.0)
        p = [1.001, Float64(count), condition]
        @test Float64(value_exe(Reactant.to_rarray(p))) ≈ native_nested_scopes(p)
        expected = only(Enzyme.gradient(Enzyme.Reverse, native_nested_scopes, p))
        @test Array(grad_exe(Reactant.to_rarray(p))) ≈ expected
    end
end

function nested_function_return(p)
    @allowscalar x = p[1]
    @trace result = if x > 0
        (y -> begin
            return y*y
        end)(x)
    else
        -x
    end
    return result
end

@testset "Control flow belongs to its lexical scope" begin
    x = Reactant.to_rarray([2.0])
    exe = @compile nested_function_return(x)
    @test Float64(exe(x)) == 4.0
    @test Float64(exe(Reactant.to_rarray([-2.0]))) == 2.0
    @test_throws LoadError @eval @trace if false
        return 1
    end
    @test_throws LoadError @eval @trace for i in 1:2
        break
    end
    @test_throws LoadError @eval @trace while false
        continue
    end
end
