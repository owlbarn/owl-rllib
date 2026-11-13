(** Observation and action spaces. *)

type t =
  | Discrete of int  (** actions or states [0 .. n - 1] *)
  | Box of { low : float array; high : float array; shape : int array }
      (** an array of the given [shape], without a batch axis; [low] and [high] are flattened in
          row-major order and may be infinite *)

val box : low:float array -> high:float array -> shape:int array -> t
(** A validated [Box]: the bounds have one entry per element and [low <= high]. Raises
    [Invalid_argument] otherwise. *)
