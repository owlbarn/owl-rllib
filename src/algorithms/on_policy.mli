(** Rollout collection and updates shared by the actor-critic methods ({!Owl_rllib.A2c},
    {!Owl_rllib.Ppo}) and by language-model RL.

    Actor and critic are separate networks with separate optimisers. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  include module type of struct
    include Neural.Make (Graph)
  end

  module R : module type of struct
    include Rollout.Make (Algodiff.A)
  end

  module P : module type of struct
    include Policy.Make (Algodiff)
  end

  type state = {
    mutable obs : Algodiff.A.arr array;
    totals : float array;
    mutable completed : float list;
  }
  (** The vector environment's current observations, the running return of each environment, and the
      returns of completed episodes, newest first. *)

  val start : (Algodiff.A.arr, 'action) Vec_env.t -> state
  (** Reset every environment. *)

  val collect :
    horizon:int ->
    gamma:float ->
    lambda:float ->
    rng:Random.State.t ->
    policy:Graph.network ->
    critic:Graph.network ->
    distribution:(Algodiff.t, Algodiff.A.arr, 'action) Policy.t ->
    (Algodiff.A.arr, 'action) Vec_env.t ->
    state ->
    R.t
  (** Run [horizon] steps in every environment and compute GAE. The value of each transition's final
      observation (not of the automatic reset) is recorded for the bootstrap. *)

  val normalise : float array -> float array
  (** Zero mean, unit standard deviation (plus 1e-8). *)

  val update_clipped :
    ?step_scale:float -> max_grad_norm:float -> Graph.network -> Stepper.state -> Algodiff.t -> unit
  (** One optimiser step, with the gradient clipped to a global norm over all the network's
      parameters. [step_scale] (default 1) multiplies the step; for Adam that is the same as
      multiplying the learning rate. *)

  val select : R.t -> int array -> Algodiff.t * Algodiff.t * Algodiff.t * float array * Algodiff.t
  (** Observations, actions, old log-probabilities, advantages and returns of the given transitions.
  *)

  val old_values : R.t -> int array -> Algodiff.t
  (** The critic's values recorded during the rollout. *)
end
