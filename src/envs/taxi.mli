(** Taxi, with the same transition table as Gymnasium's Taxi-v3.

    A taxi on a 5x5 grid picks up a passenger at one of four locations and drops them at another.
    Every step costs 1, a successful drop-off earns 20, an illegal pick-up or drop-off costs 10.
    Actions: 0 south, 1 north, 2 east, 3 west, 4 pick up, 5 drop off. *)

val encode : int -> int -> int -> int -> int
(** [encode row col passenger destination] is the state number, as in Gymnasium; [passenger] 4 means
    in the taxi. *)

val decode : int -> int * int * int * int
val make : unit -> Mdp.t
