module RegularizationModel

using LinearAlgebra, NLPModels

export RegNLP

"""
    RegNLP

Implements

    f(x) + ½ ρ ‖x - z‖²
"""
mutable struct RegNLP <: AbstractNLPModel
  meta::NLPModelMeta
  inner::AbstractNLPModel
  ρ
  z
end

show_header(io::IO, nlp::RegNLP) = println(io, "RegNLP - Regularized model")

function Base.show(io::IO, nlp::RegNLP)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.inner.counters)
end

function RegNLP(nlp::AbstractNLPModel, ρ, z)
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh + n
  return RegNLP(NLPModelMeta(n, x0 = nlp.meta.x0, nnzh = nnzh), nlp, ρ, z)
end

@default_counters RegNLP inner

function NLPModels.obj(nlp::RegNLP, x::AbstractVector)
  return obj(nlp.inner, x) + nlp.ρ * norm(x - nlp.z)^2 / 2
end

function NLPModels.grad!(nlp::RegNLP, x::AbstractVector, g::AbstractVector)
  grad!(nlp.inner, x, g)
  g .+= nlp.ρ * (x - nlp.z)
  return g
end

function NLPModels.hess_structure!(
  nlp::RegNLP,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nz = nlp.inner.meta.nnzh
  n = nlp.meta.nvar
  @views hess_structure!(nlp.inner, rows[1:nz], cols[1:nz])
  rows[(nz + 1):end] .= 1:n
  cols[(nz + 1):end] .= 1:n
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::RegNLP,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = 1.0,
)
  nz = nlp.inner.meta.nnzh
  n = nlp.meta.nvar
  @views hess_coord!(nlp.inner, x, vals[1:nz]; obj_weight = obj_weight)
  vals[(nz + 1):end] .= nlp.ρ * obj_weight
  return vals
end

function NLPModels.hprod!(
  nlp::RegNLP,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = 1.0,
)
  hprod!(nlp.inner, x, v, Hv, obj_weight = obj_weight)
  Hv .+= nlp.ρ * obj_weight * v
  return Hv
end

end # module
