(** FrozenLake, with the same transition table as Gymnasium's FrozenLake-v1.

    ['S'] start, ['F'] frozen, ['H'] hole (terminal), ['G'] goal (terminal, reward 1). Actions: 0
    left, 1 down, 2 right, 3 up. On slippery ice the agent moves in the intended direction or one of
    the two perpendicular ones, each with probability 1/3. *)

val map_4x4 : string array

val make : ?slippery:bool -> ?map:string array -> unit -> Mdp.t
(** Slippery by default, on [map_4x4] by default. *)
