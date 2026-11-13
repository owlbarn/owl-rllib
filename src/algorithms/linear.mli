(** Semi-gradient SARSA and Q-learning on sparse binary features (for example tile coding). The step
    size is divided by the number of active features. *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  val train :
    ?off_policy:bool ->
    features:(A.arr -> int array) ->
    n_features:int ->
    gamma:float ->
    alpha:float ->
    epsilon:float ->
    episodes:int ->
    rng:Random.State.t ->
    (A.arr, int) Ndenv.t ->
    float array array * float array
  (** [features obs] lists the active features. Returns the weights [w.(action).(feature)] and the
      return of every episode. [off_policy] (default false) selects Q-learning instead of SARSA. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
