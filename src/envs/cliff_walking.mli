(** CliffWalking, with the same transition table as Gymnasium's CliffWalking-v0 (Sutton and Barto,
    example 6.6).

    A 4x12 grid; the agent starts in the bottom-left corner and must reach the bottom-right one.
    Every step costs 1; stepping onto the cliff between them costs 100 and sends the agent back to
    the start. Actions: 0 up, 1 right, 2 down, 3 left. *)

val rows : int
val cols : int
val start : int
val goal : int
val is_cliff : int -> bool
val make : unit -> Mdp.t

val print_policy : int array -> unit
(** Print one arrow per cell. *)
