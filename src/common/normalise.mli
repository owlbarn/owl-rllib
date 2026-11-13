(** Observation and reward normalisation, as Gymnasium 0.29.1's [NormalizeObservation] and
    [NormalizeReward].

    Statistics are float64 running moments (starting from mean 0, variance 1, count 1e-4), updated
    with every value seen. Each wrapped environment keeps its own statistics. *)

val reward :
  ?epsilon:float ->
  ?bound:float ->
  gamma:float ->
  ('obs, 'action) Ndenv.t ->
  ('obs, 'action) Ndenv.t
(** Divide rewards by the running standard deviation of the discounted return, then clip them to
    [[-bound, bound]] (default: no clip). As in Gymnasium, the return restarts after a termination
    but not after a truncation. [epsilon] defaults to 1e-8. *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  val observation :
    ?epsilon:float -> ?bound:float -> (A.arr, 'action) Ndenv.t -> (A.arr, 'action) Ndenv.t
  (** [(obs - mean) / sqrt (var + epsilon)] after adding [obs] to the statistics, clipped to
      [[-bound, bound]] (default: no clip). The observation space becomes [[-bound, bound]]. Raises
      [Invalid_argument] unless observations are a Box. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
