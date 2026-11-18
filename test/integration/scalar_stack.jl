using Reactant, Enzyme, Test
scalar_stack_cost(x)=sum(abs2,stack(x))
scalar_stack_gradient(x)=only(Enzyme.gradient(Enzyme.Reverse,scalar_stack_cost,x))
scalar_stack_flat(x)=stack(x;dims=1)
scalar_stack_bad_axis(x)=stack(x;dims=2)
@testset "Stack a scalar tensor collection" begin
    for shape in ((3,),(2,3),(2,3,2))
        x=reshape(collect(1.0:prod(shape)),shape)
        q=Reactant.to_rarray(x)
        f=Reactant.@compile stack(q)
        flat=Reactant.@compile scalar_stack_flat(q)
        g=Reactant.@compile scalar_stack_gradient(q)
        @test Array(f(q))==stack(x)
        @test Array(flat(q))==stack(x;dims=1)
        @test Array(g(q))==2x
        @test Array(q)==x
    end
    @test_throws ArgumentError Reactant.@compile stack(Reactant.to_rarray(Float64[]))
    @test_throws ArgumentError Reactant.@compile scalar_stack_bad_axis(Reactant.to_rarray([1.0]))
end
