using Reactant, StaticArrays, Enzyme, Test

function lazy_integral_branch(pred, yes, no, args)
    @trace result = if pred
        yes(args...)
    else
        no(args...)
    end
    return result
end

function static_integral_term(q, t)
    @allowscalar begin
        A = SMatrix{3,3}(-q[1], 0.0, 0.0, 0.1, -q[2], 0.0, 0.0, 0.1, -q[3])
        state = SVector(q[1], q[2], q[3])
    end
    return lazy_integral_branch(t > 0,
        () -> sum(StaticArrays._exp_solve(A, exp(A * t) - one(A)) * state),
        () -> 0.0, ())
end

function retained_static_integral(q, times)
    result = 0.0
    @trace for i in eachindex(times)
        @allowscalar t = times[i]
        term_result = static_integral_term(q, t)
        result = result + term_result
    end
    return result
end

retained_integral_gradient(q, times) =
    only(Enzyme.gradient(Enzyme.Reverse, x -> retained_static_integral(x, times), q))

function native_static_integral(q, times)
    A = SMatrix{3,3}(-q[1], 0.0, 0.0, 0.1, -q[2], 0.0, 0.0, 0.1, -q[3])
    state = SVector(q[1], q[2], q[3])
    return sum(t > 0 ? sum((A \ (exp(A * t) - one(A))) * state) : 0.0 for t in times)
end

@testset "Static exponential and solve inside a retained lazy recurrence" begin
    q = Reactant.to_rarray([0.1, 0.2, 0.3])
    times = Reactant.to_rarray([0.1, 0.0, 20.0])
    value_exe = @compile retained_static_integral(q, times)
    gradient_exe = @compile retained_integral_gradient(q, times)
    for host_q in ([0.1, 0.2, 0.3], [0.17, 0.4, 0.27]),
        host_times in ([0.1, 0.0, 20.0], [0.0, 0.5, 3.0])
        runtime_q = Reactant.to_rarray(host_q)
        runtime_times = Reactant.to_rarray(host_times)
        @test Float64(value_exe(runtime_q, runtime_times)) ≈
            native_static_integral(host_q, host_times)
        native_gradient = only(Enzyme.gradient(
            Enzyme.Reverse, x -> native_static_integral(x, host_times), host_q))
        @test Array(gradient_exe(runtime_q, runtime_times)) ≈ native_gradient
        @test Array(runtime_q) == host_q
        @test Array(runtime_times) == host_times
    end
end
