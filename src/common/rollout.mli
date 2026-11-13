(** On-policy rollouts and generalised advantage estimation.

    Data from [n_envs] parallel environments is stored flat and time-major: index
    [t * n_envs + env]. *)

val gae :
  n_envs:int ->
  gamma:float ->
  lambda:float ->
  rewards:float array ->
  values:float array ->
  next_values:float array ->
  terminated:bool array ->
  truncated:bool array ->
  float array * float array
(** Advantages and returns (advantages + values). [next_values] are the values of each transition's
    final observation, never of an automatic reset. A termination cuts the bootstrap; a truncation
    bootstraps from the next value but stops the recursion, so advantages never cross episodes.
    Raises [Invalid_argument] on inconsistent lengths. *)

val permutation : rng:Random.State.t -> int -> int array
(** A uniformly random permutation of [0 .. n - 1] (Fisher-Yates). *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  type transition = {
    obs : A.arr;
    action : A.arr;  (** one-hot for discrete actions, the raw sample for Gaussians *)
    log_prob : float;
    value : float;
    reward : float;
    next_value : float;
    terminated : bool;
    truncated : bool;
  }

  type t = { data : transition array; advantages : float array; returns : float array }

  val create : n_envs:int -> gamma:float -> lambda:float -> transition array -> t
  (** Compute advantages and returns with {!val-gae}. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
