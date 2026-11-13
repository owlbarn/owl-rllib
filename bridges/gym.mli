(** Gymnasium environments, run in a Python child process.

    Python runs in its own process, so the native libraries it loads never mix with Owl's. Requests
    and replies are lines over a pipe, with floats in hexadecimal, so values cross bit-exact. The
    interpreter is [$OWL_RLLIB_PYTHON] (default [python3]), run with [-I]; the child exits when the
    program does.

    Errors raised in Python are reported as [Failure] with Python's message, and the environment
    remains usable. *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  val discrete : seed:int -> string -> (A.arr, int) Owl_rllib.Ndenv.t
  (** [discrete ~seed id] is [gymnasium.make id] with discrete actions. [seed] seeds the first
      reset, as [env.reset(seed=seed)]; later resets continue from the environment's own generator,
      as in Gymnasium. Observations must be a Box; terminated and truncated are Gymnasium's. Raises
      [Invalid_argument] if the actions are not discrete. Raises [Failure] if Python cannot create
      the environment. *)

  val continuous : seed:int -> string -> (A.arr, A.arr) Owl_rllib.Ndenv.t
  (** The same with Box actions. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
