(** Deep Q-networks (Mnih et al., 2015): experience replay, a periodically copied target network, an
    MSE loss and epsilon-greedy exploration.

    Below, [network] is [Graph.network] and environments are
    [(Graph.Neuron.Optimise.Algodiff.A.arr, int) Ndenv.t]. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  type config = {
    gamma : float;
    learning_rate : float;  (** Adam *)
    capacity : int;  (** replay buffer size *)
    batch_size : int;
    learning_starts : int;  (** steps before the first update *)
    train_every : int;  (** steps between updates *)
    target_every : int;  (** steps between target-network copies *)
    epsilon_start : float;
    epsilon_end : float;
    exploration_steps : int;  (** epsilon falls linearly over these steps *)
    target_network : bool;  (** [false] bootstraps from the online network (an ablation) *)
  }

  val default : config
  (** gamma 0.99, learning rate 1e-3, capacity 10,000, batch 64, learning starts after 1,000 steps,
      an update every 4 steps, a target copy every 500, epsilon 1 to 0.05 over 10,000 steps. *)

  val train :
    ?config:config ->
    steps:int ->
    rngs:Rng.t ->
    network:Graph.network ->
    (Graph.Neuron.Optimise.Algodiff.A.arr, int) Ndenv.t ->
    float array
  (** Train [network] in place for [steps] environment steps; returns the return of every completed
      episode. *)
end

module S : module type of Make (Owl.Neural.S.Graph)
module D : module type of Make (Owl.Neural.D.Graph)
