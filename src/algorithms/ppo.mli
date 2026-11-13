(** Proximal policy optimisation with a clipped surrogate (Schulman et al., 2017), GAE, several
    epochs of minibatches per rollout, and optional details from CleanRL (value clipping,
    learning-rate annealing). Works with discrete and continuous actions through
    {!Owl_rllib.Policy}. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  type config = {
    horizon : int;  (** steps per environment per rollout *)
    epochs : int;
    batch_size : int;  (** minibatch size *)
    gamma : float;
    lambda : float;  (** GAE *)
    learning_rate : float;  (** Adam, for actor and critic *)
    clip : float;  (** ratio clip, also the value clip *)
    entropy_coef : float;
    value_coef : float;
    max_grad_norm : float;
    normalise_advantage : bool;  (** per minibatch *)
    clip_value : bool;  (** clip the value update to [clip] around the rollout's values *)
    anneal_lr : bool;  (** the learning rate falls linearly to 0 over the updates *)
  }

  val default : config
  (** horizon 128, 4 epochs, batch 64, gamma 0.99, lambda 0.95, learning rate 3e-4, clip 0.2,
      entropy 0, value 0.5, gradient norm 0.5, advantages normalised, no value clipping, no
      annealing. *)

  val surrogate :
    clip:float ->
    log_prob:Graph.Neuron.Optimise.Algodiff.t ->
    old_log_prob:Graph.Neuron.Optimise.Algodiff.t ->
    advantages:Graph.Neuron.Optimise.Algodiff.t ->
    Graph.Neuron.Optimise.Algodiff.t
  (** The clipped surrogate loss, negated for minimisation. *)

  val clipped_value_error :
    clip:float ->
    old:Graph.Neuron.Optimise.Algodiff.t ->
    predicted:Graph.Neuron.Optimise.Algodiff.t ->
    returns:Graph.Neuron.Optimise.Algodiff.t ->
    Graph.Neuron.Optimise.Algodiff.t
  (** Per-sample [max ((v - R)^2, (clip v - R)^2)], where [clip v] stays within [clip] of the old
      value. *)

  val learning_rate_fraction : updates:int -> int -> float
  (** Update [u] of [updates] runs at [1 - (u - 1) / updates] of the learning rate when annealing.
  *)

  val train :
    ?config:config ->
    updates:int ->
    rngs:Rng.t ->
    policy:Graph.network ->
    critic:Graph.network ->
    distribution:
      (Graph.Neuron.Optimise.Algodiff.t, Graph.Neuron.Optimise.Algodiff.A.arr, 'action) Policy.t ->
    (Graph.Neuron.Optimise.Algodiff.A.arr, 'action) Vec_env.t ->
    float array
  (** Train [policy] and [critic] in place for [updates] rollouts; returns the return of every
      completed episode. *)
end

module S : module type of Make (Owl.Neural.S.Graph)
module D : module type of Make (Owl.Neural.D.Graph)
