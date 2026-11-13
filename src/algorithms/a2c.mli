(** Synchronous advantage actor-critic: one update per fresh rollout, no replay. Works with discrete
    and continuous actions through {!Owl_rllib.Policy}. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  type config = {
    horizon : int;  (** steps per environment per update *)
    gamma : float;
    lambda : float;  (** GAE *)
    learning_rate : float;  (** Adam, for actor and critic *)
    entropy_coef : float;
    value_coef : float;
    max_grad_norm : float;
  }

  val default : config
  (** horizon 5, gamma 0.99, lambda 1, learning rate 7e-4, entropy 0.01, value 0.5, gradient norm
      0.5. *)

  val actor_loss :
    log_prob:Graph.Neuron.Optimise.Algodiff.t ->
    advantages:Graph.Neuron.Optimise.Algodiff.t ->
    entropy:Graph.Neuron.Optimise.Algodiff.t ->
    entropy_coef:float ->
    Graph.Neuron.Optimise.Algodiff.t
  (** [-mean (log_prob * advantages) - entropy_coef * mean entropy]. *)

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
