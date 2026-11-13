(** Synchronous vector environments with automatic reset. *)

type ('obs, 'action) t

type 'obs step = {
  transitions : 'obs Ndenv.step array;
      (** each environment's step, with its true final observation *)
  obs : 'obs array;
      (** the observations for the next action: after a reset where an episode ended *)
}

val create : ('obs, 'action) Ndenv.t array -> ('obs, 'action) t
(** Raises [Invalid_argument] if the array is empty or the spaces differ. *)

val length : ('obs, 'action) t -> int
val reset : ('obs, 'action) t -> 'obs array
val step : ('obs, 'action) t -> 'action array -> 'obs step
