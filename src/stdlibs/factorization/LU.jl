struct BatchedLU{T,S<:AbstractArray,P<:AbstractArray,I<:Union{AbstractArray,Number}} <:
       BatchedFactorization{T}
    factors::S
    ipiv::P
    perm::P
    info::I
end

_factorization_index(::Type{P}, i::Integer) where {P} = P(i)
_factorization_index(::Type{P}, i::TracedRNumber) where {P} =
    promote_to(TracedRNumber{P}, i)

# A primal decomposition of pivoted LU into scalar/tensor operations. Keeping
# this graph visible lets ordinary reverse differentiate the computation; no
# derivative rule or activity override is needed for the factorization.
function _lu_math(A::AbstractMatrix, ::Type{P}) where {P}
    pivot_type = P
    F = copy(A)
    m, n = size(F)
    permutation = similar(F, P, m)
    permutation .= collect(P, 1:m)
    pivots = similar(F, P, min(m,n))
    fill!(pivots, zero(P))
    info = zero(P)
    @allowscalar begin
        @trace for k in 1:min(m,n)
            pivot = _factorization_index(pivot_type, k)
            largest = abs(F[k,k])
            @trace for row in (k+1):m
                magnitude = abs(F[row,k])
                @trace if magnitude > largest
                    pivot = _factorization_index(pivot_type, row)
                    largest = magnitude
                end
            end
            pivots[k] = pivot
            old_row = copy(F[k,:])
            F[k,:] = F[pivot,:]
            F[pivot,:] = old_row
            old_permutation = permutation[k]
            permutation[k] = permutation[pivot]
            permutation[pivot] = old_permutation
            diagonal = F[k,k]
            @trace if !iszero(diagonal)
                @trace for row in (k+1):m
                    F[row,k] = F[row,k] / diagonal
                end
                @trace for column in (k+1):n
                    @trace for row in (k+1):m
                        F[row,column] = F[row,column] - F[row,k]*F[k,column]
                    end
                end
            else
                @trace if iszero(info)
                    info = _factorization_index(pivot_type, k)
                end
            end
        end
    end
    return F, pivots, permutation, info
end

function _lu_solve_math(A::AbstractMatrix, B::AbstractMatrix)
    factors, pivots, permutation, info = @opcall call(_lu_math, A, Int32)
    return BatchedLU(factors, pivots, permutation, info) \ B
end

_triangular_rows(::Val{true}, n) = 1:n
_triangular_rows(::Val{false}, n) = n:-1:1
_triangular_terms(::Val{true}, i, n) = 1:(i-1)
_triangular_terms(::Val{false}, i, n) = (i+1):n
_triangular_diagonal(::Val{true}, A, value, i) = value
_triangular_diagonal(::Val{false}, A, value, i) = value / A[i,i]

function _triangular_solve_math(A::AbstractMatrix, B::AbstractMatrix, lower, unit)
    X = copy(B)
    n = size(A,1)
    rows = _triangular_rows(lower, n)
    @allowscalar begin
        @trace for column in 1:size(B,2)
            @trace for row in rows
                value = X[row,column]
                @trace for j in _triangular_terms(lower, row, n)
                    value = value - A[row,j]*X[j,column]
                end
                X[row,column] = _triangular_diagonal(unit, A, value, row)
            end
        end
    end
    return X
end

Base.size(lu::BatchedLU) = size(lu.factors)
Base.size(lu::BatchedLU, i::Integer) = size(lu.factors, i)
Base.ndims(lu::BatchedLU) = ndims(lu.factors)
function Base.copy(lu::BatchedLU)
    return BatchedLU(copy(lu.factors), copy(lu.ipiv), copy(lu.perm), copy(lu.info))
end

function BatchedLU(factors::S, ipiv::P, perm::P, info::I) where {S,P,I}
    @assert ndims(ipiv) == ndims(perm) == ndims(factors) - 1
    @assert ndims(info) == ndims(factors) - 2
    return BatchedLU{eltype(factors),S,P,I}(factors, ipiv, perm, info)
end

function overloaded_lu(x::AbstractArray, args...; kwargs...)
    return overloaded_lu(Reactant.promote_to(TracedRArray, x), args...; kwargs...)
end

function overloaded_lu(
    A::AnyTracedRArray{T,N}, ::RowMaximum; check::Bool=false, allowsingular::Bool=false
) where {T,N}
    # TODO: don't ignore the check and allowsingular flags
    # Batching here is in the last dimensions. `Ops.lu` expects the last dimensions
    permdims = vcat(collect(Int64, 3:N), 1, 2)
    A = @opcall transpose(materialize_traced_array(A), permdims)
    factors, ipiv, perm, info = @opcall lu(A)

    # Permute back to the original dimensions
    perm_perm = vcat(N - 1, collect(Int64, 1:(N - 2)))
    factors = @opcall transpose(factors, invperm(permdims))
    ipiv = @opcall transpose(ipiv, perm_perm)
    perm = @opcall transpose(perm, perm_perm)
    return BatchedLU(factors, ipiv, perm, info)
end

function LinearAlgebra.ldiv!(
    lu::BatchedLU{T,<:AbstractArray{T,N},P,I}, B::AbstractArray{T,M}
) where {T,P,I,N,M}
    @assert N == M + 1
    ldiv!(lu, reshape(B, size(B, 1), 1, size(B)[2:end]...))
    return B
end

function LinearAlgebra.ldiv!(
    lu::BatchedLU{T,<:AbstractArray{T,2},P,I}, B::AbstractArray{T,2}
) where {T,P,I}
    B .= _lu_solve_core(lu.factors, B, lu.perm)
    return B
end

function LinearAlgebra.ldiv!(
    lu::BatchedLU{T,<:AbstractArray{T,N},P,I}, B::AbstractArray{T,N}
) where {T,P,I,N}
    batch_shape = size(lu.factors)[3:end]
    @assert batch_shape == size(B)[3:end]

    permutation = vcat(collect(Int64, 3:N), 1, 2)

    factors = @opcall transpose(materialize_traced_array(lu.factors), permutation)
    B_permuted = @opcall transpose(materialize_traced_array(B), permutation)
    perm = @opcall transpose(
        materialize_traced_array(lu.perm), vcat(collect(Int64, 2:(N - 1)), 1)
    )

    res = @opcall transpose(
        only(
            @opcall(
                batch(
                    _lu_solve_core, [factors, B_permuted, perm], collect(Int64, batch_shape)
                )
            ),
        ),
        invperm(permutation),
    )
    B .= res
    return B
end

function LinearAlgebra.det(lu::BatchedLU{T,<:AbstractMatrix}) where {T}
    n = LinearAlgebra.checksquare(lu)
    # TODO: check for non-singular matrices

    P = prod(LinearAlgebra.diag(lu.factors))
    return ifelse(isodd(sum(lu.ipiv[1:n] .!= (1:n))), -one(T), one(T)) * P
end

function LinearAlgebra.logabsdet(lu::BatchedLU{T,<:AbstractMatrix}) where {T}
    n = LinearAlgebra.checksquare(lu)
    Treal = real(T)
    # TODO: check for non-singular matrices

    d = LinearAlgebra.diag(lu.factors)
    absdet = sum(log ∘ abs, d)
    P = prod(sign, d)
    s = ifelse(isodd(sum(lu.ipiv[1:n] .!= (1:n))), -one(Treal), one(Treal)) * P
    return absdet, s
end

for f_wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    aType in (:AbstractVecOrMat, :AbstractArray)

    @eval function LinearAlgebra.ldiv!(lu::$(f_wrapper){<:Any,<:BatchedLU}, B::$aType)
        # TODO: implement this
        error("`$(f_wrapper)` is not supported yet for LU.")
        return nothing
    end
end

# currently we lower inverse to lu decomposition + triangular solve. we should
# instead emit getri and lower that to a fallback if the backend doesn't support
# it.
function LinearAlgebra.inv!(lu::BatchedLU)
    @assert ndims(lu) == 2 "Only implemented for 2D tensors"
    rhs = Reactant.promote_to(
        TracedRArray{Reactant.unwrapped_eltype(eltype(lu)),2}, LinearAlgebra.I(size(lu, 1))
    )
    ldiv!(lu, rhs)
    return rhs
end

function _lu_solve_core(factors::AbstractMatrix, B::AbstractMatrix, perm::AbstractVector)
    permuted_B = B[Int64.(perm), :]
    return UpperTriangular(factors) \ (UnitLowerTriangular(factors) \ permuted_B)
end
