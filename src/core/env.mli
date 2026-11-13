(** Tabular environments: integer states and actions, with the model hidden behind [reset] and
    [step]. *)

type step = {
  state : int;
  reward : float;
  terminated : bool;  (** reached a terminal state: do not bootstrap *)
  truncated : bool;  (** hit the step limit: the episode ends, but bootstrap *)
}

type t = { n_states : int; n_actions : int; reset : unit -> int; step : int -> step }

val of_mdp : ?max_steps:int -> rng:Random.State.t -> Mdp.t -> t
(** Sample an MDP's transitions with [rng]. Episodes that last [max_steps] steps (default:
    unlimited) are truncated. *)
