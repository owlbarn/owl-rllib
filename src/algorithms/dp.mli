(** Dynamic programming on a known MDP (Sutton and Barto, chapter 4).

    Values are float arrays indexed by state. A deterministic policy is an [int array] of actions; a
    stochastic one gives [policy.(s).(a)], the probability of [a] in [s]. Sweeps update values in
    place and stop when the largest change falls below [theta] (default 1e-10). *)

val backup : gamma:float -> Mdp.t -> float array -> int -> int -> float
(** [backup ~gamma mdp v s a]: the expected return of [a] in [s], then [v]. *)

val q_from_v : gamma:float -> Mdp.t -> float array -> float array array
(** Action values [q.(s).(a)] from state values. *)

val argmax : float array -> int
(** The first index of the largest element. *)

val greedy : float array array -> int array

val one_hot : Mdp.t -> int array -> float array array
(** A deterministic policy as action probabilities. *)

val policy_evaluation : ?theta:float -> gamma:float -> Mdp.t -> float array array -> float array
(** The value of a stochastic policy (iterative policy evaluation). *)

val policy_iteration : ?theta:float -> gamma:float -> Mdp.t -> int array * float array
(** An optimal policy and its value. Ties keep the current action, so the iteration always
    terminates. *)

val value_iteration : ?theta:float -> gamma:float -> Mdp.t -> int array * float array
(** The optimal value and a policy greedy with respect to it. *)
