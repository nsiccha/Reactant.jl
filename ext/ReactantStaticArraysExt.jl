module ReactantStaticArraysExt

using Reactant
using LinearAlgebra
import Reactant.TracedRArrayOverrides: overloaded_map, overloaded_mapreduce
import Reactant.TracedLinearAlgebra: overloaded_mul

using StaticArrays: StaticArrays, SArray, StaticArray

const SAReact{Sz,T} = StaticArray{Sz,T} where {Sz<:Tuple,T<:Reactant.TracedRNumber}
const SMatReact = StaticArrays.StaticMatrix{M,N,T} where {M,N,T<:Reactant.TracedRNumber}
const SVecReact = StaticArrays.StaticVector{N,T} where {N,T<:Reactant.TracedRNumber}
const SVecOrMat = Union{StaticArrays.StaticVector,StaticArrays.StaticMatrix}
const SVecOrMatReact = Union{SVecReact,SMatReact}

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(FA::Type{SArray{S,T,N,L}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {S,T,N,L}
    T_traced = Reactant.traced_type_inner(T, seen, mode, track_numbers, ndevices, runtime)
    return SArray{S,T_traced,N,L}
end

function Reactant.make_tracer(
    seen, x::SArray{S,T,N,0}, path, mode; kwargs...
) where {S,T,N}
    # There are no scalar leaves whose conversion can rebuild an empty array.
    # Remove the tracing-only element type when reconstructing a result.
    if mode == Reactant.TracedToConcrete && T <: Reactant.TracedRNumber
        return SArray{S,Reactant.unwrapped_eltype(T),N,0}()
    end
    return Reactant.make_tracer_unknown(seen, x, path, mode; kwargs...)
end

# Static containers have value semantics, although their tracing leaves are
# mutable handles. Prepare distinct carry slots before staging either region,
# without changing tracer identity during region mutation detection.
Reactant.Ops._loop_seed_value(x::SArray{S,T,N,L}) where
    {S,T<:Reactant.TracedRNumber,N,L} = map(copy, x)

function Reactant.materialize_traced_array(x::SAReact)
    # Array(x) reaches the traced copyto! overlay, which materializes x again.
    # Build from the scalar tuple so conversion has a non-recursive base case.
    return Reactant.aos_to_soa(reshape(collect(Tuple(x)), size(x)))
end

# Keep the authored static result while lowering its contraction as one tensor
# operation. Scalar expansion of every product inside nested checkpoint replay
# produces a large reverse graph even for a four-dimensional matrix.
function _tensor_static_product(A, B)
    tensor_A = Reactant.aos_to_soa(reshape(collect(Tuple(A)), size(A)))
    tensor_B = Reactant.aos_to_soa(reshape(collect(Tuple(B)), size(B)))
    result = Reactant.call_with_native(
        Reactant.TracedLinearAlgebra.overloaded_mul, tensor_A, tensor_B
    )
    result_type = StaticArrays.similar_type(A, eltype(result), StaticArrays.Size(size(result)))
    Reactant.@allowscalar return result_type(ntuple(i -> result[i], Val(length(result))))
end

Reactant.@reactant_overlay Base.:(*)(A::SMatReact, B::SVecOrMatReact) =
    _tensor_static_product(A, B)
Reactant.@reactant_overlay Base.:(*)(A::SMatReact, B::SVecOrMat) =
    _tensor_static_product(A, B)
Reactant.@reactant_overlay Base.:(*)(A::StaticArrays.StaticMatrix, B::SVecOrMatReact) =
    _tensor_static_product(A, B)

function Reactant.TracedUtils.set_mlir_data!(x::SAReact, data)
    # The container is immutable; its traced scalar leaves are mutable. Update
    # those leaves when a region argument/result replaces the tensor value.
    tensor = Reactant.TracedRArray{Reactant.unwrapped_eltype(x)}(data)
    Reactant.@allowscalar for i in eachindex(x)
        Reactant.TracedUtils.set_mlir_data!(
            x[i], Reactant.TracedUtils.get_mlir_data(tensor[i])
        )
    end
    return x
end

# We don't want to overload map on StaticArrays since it is likely better to just unroll things
overloaded_map(f, a::SAReact, rest::SAReact...) = f.(a, rest...)
overloaded_mapreduce(f, op, a::SAReact; kwargs...) = mapreduce(f, op, a, kwargs...)

function overloaded_mul(A::SAReact, B::SAReact, alpha::Number=true, beta::Number=false)
    # beta is not supported since it is zero by default in Reactant 
    # (similar is zero'd automatically for TracedRArrays)
    C = A * B
    if !(alpha isa Reactant.TracedRNumber) && isone(alpha)
        return C
    end
    return C .* alpha
end

# Older StaticArrays releases still load the extension; the shared lazy kernels
# are used when the coordinated StaticArrays capability is available.
if all(name -> isdefined(StaticArrays, name),
       (:_exp_choose, :_exp_solve, :_exp_square, :_exp_real_2x2))

function StaticArrays._exp_choose(
    condition::Reactant.TracedRNumber{Bool}, yes, no, args
)
    Reactant.@trace result = if condition
        yes(args...)
    else
        no(args...)
    end
    return result
end

function StaticArrays._exp(
    ::StaticArrays.Size{(2,2)},
    A::StaticArrays.StaticMatrix{2,2,Reactant.TracedRNumber{T}},
) where {T<:Real}
    return StaticArrays._exp_real_2x2(A)
end

function StaticArrays._exp_solve(A::SAReact, B::SAReact)
    # Static scalar leaves suit unrolled arithmetic, but a factorization needs
    # tensors. Restore the static result shape after the ordinary tensor solve.
    tensor_A = Reactant.materialize_traced_array(A)
    tensor_B = Reactant.materialize_traced_array(B)
    factors, pivots, permutation, info = Reactant.Ops.call(
        Reactant.TracedLinearAlgebra._lu_math, tensor_A, Int32
    )
    factorization = Reactant.TracedLinearAlgebra.BatchedLU(factors, pivots, permutation, info)
    result = factorization \ tensor_B
    Reactant.@allowscalar return StaticArrays.similar_type(B)(
        ntuple(i -> result[i], Val(length(B)))
    )
end

function StaticArrays._exp_square(A::SAReact, count::Reactant.TracedRNumber)
    Reactant.@trace for _ in 1:count
        A = A*A
    end
    return A
end

end

end
