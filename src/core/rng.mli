(** Explicit random streams for one experiment.

    Each use gets its own stream derived from the seed, so that, for example, evaluating a policy
    does not change the randomness of training. The streams are reproducible; they are not claimed
    to be statistically independent. *)

type t = {
  env : Random.State.t;  (** environment dynamics and initial states *)
  action : Random.State.t;  (** exploration and action sampling *)
  sample : Random.State.t;  (** replay sampling and minibatch shuffling *)
  eval : Random.State.t;  (** evaluation *)
  network_seed : int;  (** seed for Owl's generator before initialising a network *)
}

val create : int -> t

val init_network : t -> unit
(** Seed Owl's global generator with [network_seed]. *)

val normal : Random.State.t -> float
(** A standard normal sample (Box-Muller). *)
