(** Returns, TD targets and an experience replay buffer. *)

val returns : gamma:float -> ?last_value:float -> float array -> float array
(** Discounted returns of a reward sequence, accumulated in float64. [last_value] (default 0) is the
    value after the last reward, for an episode cut by a time limit. *)

val td_target : gamma:float -> reward:float -> terminated:bool -> next_value:float -> float
(** [reward + gamma * next_value], without the bootstrap after a termination. *)

(** A bounded replay buffer of discrete-action transitions. *)
module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  type transition = {
    obs : A.arr;
    action : int;
    reward : float;
    next_obs : A.arr;
    terminated : bool;
    truncated : bool;
  }

  type t

  val create : int -> t
  (** An empty buffer that keeps the most recent [capacity] transitions. *)

  val length : t -> int

  val add : t -> transition -> unit
  (** Observations are copied, so the caller may reuse its arrays. *)

  val sample : rng:Random.State.t -> batch_size:int -> t -> transition array
  (** Uniform sampling with replacement; the observations are fresh copies. *)

  val batch : A.arr array -> A.arr
  (** Stack observations along a new first (batch) axis. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
