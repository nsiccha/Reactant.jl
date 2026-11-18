using Reactant, Enzyme, StaticArrays, Test, LinearAlgebra

function condition_scalar_cost(q,n)
    Reactant.@allowscalar b=q[1]
    r=1.0; e=copy(n)
    Reactant.@trace while e>0
        r=r*b
        e=e>>1
    end
    return r
end
condition_scalar_gradient(q,n)=only(Enzyme.gradient(Enzyme.Reverse,
    x->condition_scalar_cost(x,n),q))

function condition_power_cost(A,n)
    R=Reactant.@allowscalar one(A)
    B=copy(A); e=copy(n)
    Reactant.@trace while e>0
        Reactant.@trace R = if isodd(e)
            R*B
        else
            R
        end
        e=e>>1
        Reactant.@trace B = if e>0
            B*B
        else
            B
        end
    end
    return sum(R)
end
condition_power_gradient(A,n)=only(Enzyme.gradient(Enzyme.Reverse,
    x->condition_power_cost(x,n),A))

# Both the stopping condition and the lazy branch read active values. The
# logarithm arm is invalid for positive b, and must never execute in primal/AD.
function condition_active_cost(q,limit)
    Reactant.@allowscalar b=q[1]
    r=1.0
    Reactant.@trace while r<limit
        Reactant.@trace r = if b>0
            r*b
        else
            log(b)
        end
    end
    return r
end
condition_active_gradient(q,limit)=only(Enzyme.gradient(Enzyme.Reverse,
    x->condition_active_cost(x,limit),q))

@testset "Runtime condition loop ordinary reverse" begin
    q=Reactant.to_rarray([1.1]); n=Reactant.to_rarray(5;track_numbers=true)
    f=Reactant.@compile condition_scalar_cost(q,n)
    g=Reactant.@compile condition_scalar_gradient(q,n)
    for k in (0,1,2,5,17,257)
        t=Reactant.to_rarray(k;track_numbers=true)
        bits=k==0 ? 0 : ndigits(k;base=2)
        @test Float64(f(q,t))≈1.1^bits
        @test only(Array(g(q,t)))≈bits*1.1^(bits-1)
        @test Int(t)==k
    end
    A=[0.9 0.0;0.03 0.9]; a=Reactant.to_rarray(A)
    fp=Reactant.@compile condition_power_cost(a,n)
    gp=Reactant.@compile condition_power_gradient(a,n)
    for k in (0,1,2,5,17,257)
        t=Reactant.to_rarray(k;track_numbers=true)
        @test Float64(fp(a,t))≈sum(A^k)
        block=zeros(4,4)
        block[1:2,1:2].=transpose(A)
        block[3:4,3:4].=transpose(A)
        block[1:2,3:4].=1
        expected=(block^k)[1:2,3:4]
        @test Array(gp(a,t))≈expected rtol=1e-9
        @test Int(t)==k
    end
    b=Reactant.to_rarray([1.5]); limit=Reactant.to_rarray(2.0;track_numbers=true)
    fa=Reactant.@compile condition_active_cost(b,limit)
    ga=Reactant.@compile condition_active_gradient(b,limit)
    for bound in (0.5,1.0,2.0,3.0,8.0)
        l=Reactant.to_rarray(bound;track_numbers=true)
        r=1.0; count=0
        while r<bound
            r*=1.5; count+=1
        end
        @test Float64(fa(b,l))≈r
        @test only(Array(ga(b,l)))≈count*1.5^(count-1)
    end
end

function static_power_cost(q,n)
    A=Reactant.@allowscalar SMatrix{1,1}(q)
    R=one(A); B=copy(A); e=copy(n)
    Reactant.@trace while e>0
        Reactant.@trace R = if isodd(e)
            R*B
        else
            R
        end
        e=e>>1
        Reactant.@trace B = if e>0
            B*B
        else
            B
        end
    end
    return sum(R)
end
static_condition_gradient(A,n)=only(Enzyme.gradient(Enzyme.Reverse,x->static_power_cost(x,n),A))
@testset "Ordinary immutable static seed and wide integer reverse" begin
    for I in (Int32,Int64)
        a=Reactant.to_rarray(reshape([-1.0],1,1));n=Reactant.to_rarray(I(3);track_numbers=true)
        f=Reactant.@compile static_power_cost(a,n)
        g=Reactant.@compile static_condition_gradient(a,n)
        counts=I===Int32 ? I.((0,1,3,9,17)) : I.((0,1,3,9,17,2^54,2^54+1))
        for k in counts
            t=Reactant.to_rarray(k;track_numbers=true)
            @test Float64(f(a,t))==(-1.0)^k
            expected=k==0 ? 0.0 : Float64(k)*(-1.0)^(k-1)
            @test only(Array(g(a,t)))≈expected
            @test I(t)==k
        end
    end
end
