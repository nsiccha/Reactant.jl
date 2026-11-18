using Reactant, Enzyme, StaticArrays, Test

# The argument named `args` also checks hygiene of the generated branch formal.
function apply_captured_branch(pred, yes, no, args)
    Reactant.@trace result = if pred
        yes(args...)
    else
        no(args...)
    end
    return result
end

function single_captured_cost(q)
    Reactant.@allowscalar begin
        x = q[1]
        pred = q[2] > 0
    end
    return apply_captured_branch(pred, () -> 2x, () -> x, ())
end

function nested_captured_cost(q)
    Reactant.@allowscalar begin
        x = q[1]
        pred = q[2] > 0
        inner = q[3] > 0
    end
    return apply_captured_branch(
        pred,
        () -> apply_captured_branch(inner, () -> 2x, () -> x, ()),
        () -> x,
        (),
    )
end

function nested_static_cost(q)
    Reactant.@allowscalar begin
        A = SMatrix{3,3}(-q[1], 0.0, 0.0, 0.1, -q[2], 0.0, 0.0, 0.1, -q[3])
        carry = (state=SVector(q[1], q[2], q[3]), extra=q[4])
        pred = q[5] > 0
        t = q[6]
    end
    result = apply_captured_branch(
        pred,
        () -> apply_captured_branch(
            t > 0,
            () -> (state=exp(A * t) * carry.state, extra=carry.extra),
            () -> carry,
            (),
        ),
        () -> carry,
        (),
    )
    return sum(result.state) + result.extra
end

# Native Enzyme must see pure Julia math rather than @allowscalar's task-local
# state. These references use ordinary branches and the same exp primitive.
function reference_captured_cost(q, nested)
    if q[2] > 0 && (!nested || q[3] > 0)
        return 2q[1]
    end
    return q[1]
end

function reference_static_cost(q)
    state = SVector(q[1], q[2], q[3])
    if q[5] > 0 && q[6] > 0
        A = SMatrix{3,3}(-q[1], 0.0, 0.0, 0.1, -q[2], 0.0, 0.0, 0.1, -q[3])
        return sum(exp(A * q[6]) * state) + q[4]
    end
    return sum(state) + q[4]
end

@testset "Converted box contents stay isolated" begin
    input = Reactant.TracedRNumber{Float64}(((:input,),), nothing)
    original = Core.Box(input)
    seen = Reactant.OrderedIdDict()
    traced = Reactant.make_tracer(seen, original, (:box,), Reactant.TracedSetPath)
    @test traced !== original
    @test traced.contents !== input
    @test traced.contents === seen[input]
    @test original.contents === input
    @test Reactant.TracedUtils.get_paths(traced.contents) == ((:box, :contents),)
end

@testset "Nested callable branches preserve captures and default reverse" begin
    for f in (single_captured_cost, nested_captured_cost)
        reference = x -> reference_captured_cost(x, f === nested_captured_cost)
        q = Reactant.to_rarray([1.2, 1.0, 1.0])
        value = Reactant.@compile f(q)
        gradient = x -> only(Enzyme.gradient(Enzyme.Reverse, f, x))
        reverse = Reactant.@compile gradient(q)
        for x in (0.4, 1.2), pred in (-1.0, 1.0), inner in (-1.0, 1.0)
            host = [x, pred, inner]
            arg = Reactant.to_rarray(host)
            @test Float64(value(arg)) ≈ reference(host)
            @test Array(reverse(arg)) ≈ only(Enzyme.gradient(Enzyme.Reverse, reference, host))
            @test Array(arg) == host
        end
    end
end

@testset "Nested callable static matrix branches" begin
    q = Reactant.to_rarray([0.1, 0.2, 0.3, 1.0, 1.0, 1.0])
    value = Reactant.@compile nested_static_cost(q)
    gradient = x -> only(Enzyme.gradient(Enzyme.Reverse, nested_static_cost, x))
    reverse = Reactant.@compile gradient(q)
    for pred in (-1.0, 1.0), t in (-1.0, 0.1, 20.0)
        host = [0.1, 0.2, 0.3, 1.0, pred, t]
        arg = Reactant.to_rarray(host)
        @test Float64(value(arg)) ≈ reference_static_cost(host)
        @test Array(reverse(arg)) ≈ only(Enzyme.gradient(Enzyme.Reverse, reference_static_cost, host))
        @test Array(arg) == host
    end
end
