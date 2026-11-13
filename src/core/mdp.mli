(** Finite Markov decision processes with an explicit model.

    States and actions are integers. The discount factor is not part of the model; algorithms take
    it as an argument. *)

type transition = {
  prob : float;
  next : int;
  reward : float;
  terminal : bool;  (** the episode ends on this transition: do not bootstrap *)
}

type t = {
  n_states : int;
  n_actions : int;
  transitions : transition list array array;  (** [transitions.(s).(a)] *)
  initial : float array;  (** distribution of the first state *)
}

val sample_from : Random.State.t -> ('a * float) list -> 'a
(** Draw from a discrete distribution given as (value, probability) pairs. Raises [Invalid_argument]
    on an empty list. *)

val sample_initial : Random.State.t -> t -> int
(** Draw a first state from [initial]. *)

val sample_transition : Random.State.t -> t -> int -> int -> transition
(** [sample_transition rng mdp s a] draws one of [transitions.(s).(a)]. *)
