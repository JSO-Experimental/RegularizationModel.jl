using LinearAlgebra, NLPModels, Printf, RegularizationModel, SparseArrays, Test

include(joinpath(dirname(pathof(NLPModels)), "..", "test", "consistency.jl"))

function tests()
  f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  ρ = rand()
  z = rand(2)
  rnlp = RegNLP(nlp, ρ, z)
  manual = ADNLPModel(x -> f(x) + ρ * norm(x - z)^2 / 2, [-1.2; 1.0])

  consistent_nlps([rnlp, manual])
end

tests()
