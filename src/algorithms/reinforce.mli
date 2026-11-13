(** Episodic REINFORCE with discrete actions, optionally with a learned state-value baseline.

    Each update uses a batch of complete episodes. Without a baseline, a time limit defines a
    finite-horizon Monte Carlo objective; with one, the return at a truncation bootstraps from the
    baseline's value. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  val act : rng:Random.State.t -> Graph.network -> Graph.Neuron.Optimise.Algodiff.A.arr -> int
  (** Sample an action from the policy's logits. *)

  val train :
    ?baseline:Graph.network ->
    ?gamma:float ->
    ?learning_rate:float ->
    ?batch_episodes:int ->
    ?normalise_advantage:bool ->
    episodes:int ->
    rng:Random.State.t ->
    policy:Graph.network ->
    (Graph.Neuron.Optimise.Algodiff.A.arr, int) Ndenv.t ->
    float array
  (** Train [policy] (and [baseline], if given) in place; returns every episode's return. Defaults:
      gamma 0.99, Adam with learning rate 1e-3, 8 episodes per batch, advantages not normalised. *)
end

module S : module type of Make (Owl.Neural.S.Graph)
module D : module type of Make (Owl.Neural.D.Graph)
