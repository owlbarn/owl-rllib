(** MountainCar-v0, as in Gymnasium. Actions: 0 push left, 1 no push, 2 push right. Reward -1 per
    step until the car reaches the flag; truncated after [max_steps] (default 200) steps.

    Observations match Gymnasium's step by step; the simulator state is kept in float64 and only
    observations are cast to the ndarray precision. *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  val make :
    ?max_steps:int ->
    ?initial_state:float array ->
    rng:Random.State.t ->
    unit ->
    (A.arr, int) Ndenv.t
  (** Initial states are drawn from [rng] as in Gymnasium, unless [initial_state] (the internal
      physical state) is given. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
include module type of S
