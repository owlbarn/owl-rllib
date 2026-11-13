(** Policies: how a network's output becomes a distribution and an action.

    A Gaussian policy stores the {e unclipped} sample, so log-probabilities stay those of the
    Gaussian; only the action sent to the environment is clipped to the action bounds. *)

type ('tensor, 'array, 'action) t = {
  sample : rng:Random.State.t -> 'tensor -> 'array;  (** one sample per row of the output *)
  log_prob : 'tensor -> 'tensor -> 'tensor;  (** [log_prob output samples], shape [[batch; 1]] *)
  entropy : 'tensor -> 'tensor;
  actions : 'array -> 'action array;  (** environment actions from samples *)
}

module Make (AD : Owl_algodiff_generic_sig.Sig) : sig
  val row : AD.A.arr -> int -> AD.A.arr
  (** Row [i] of a matrix, as a vector. *)

  val categorical : (AD.t, AD.A.arr, int) t
  (** The output is logits; samples are one-hot. *)

  val gaussian : low:float array -> high:float array -> (AD.t, AD.A.arr, AD.A.arr) t
  (** The output is [[mean_1 .. mean_d; log_std_1 .. log_std_d]]; log_std is bounded to [[-5, 2]].
      Actions are clipped to [[low, high]]. Raises [Invalid_argument] unless [low < high]
      element-wise. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff)
module D : module type of Make (Owl.Neural.D.Algodiff)
