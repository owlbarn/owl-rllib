(** Deterministic grid worlds described by a text map.

    ['S'] start, ['G'] goal (terminal), ['#'] wall, anything else a free cell. States are numbered
    row by row; actions are 0 up, 1 right, 2 down, 3 left. Moving into a wall or off the grid leaves
    the agent in place. Without any ['S'], episodes start in a random free cell. *)

type t = { rows : int; cols : int; cells : string array; mdp : Mdp.t }

val make : ?step_reward:float -> ?goal_reward:float -> string array -> t
(** Every move earns [step_reward] (default -1); entering a goal adds [goal_reward] (default 0). *)

val sutton_4x4 : unit -> t
(** The 4x4 grid of Sutton and Barto's example 4.1, goals in two corners. *)

val arrows : string array
(** One character per action, for printing policies. *)

val render : t -> (int -> string) -> unit
(** Print the grid with a label for each state. *)

val print_values : t -> float array -> unit
val print_policy : t -> int array -> unit
