(** Environments with array observations, as in Gymnasium.

    The action type is a parameter: [int] for discrete control, an ndarray for continuous control.
    An episode ends when [terminated] (do not bootstrap) or [truncated] (time limit: bootstrap) is
    set; both can be set at once. After that, [step] raises [Invalid_argument] until the next
    [reset]. *)

type 'obs step = { obs : 'obs; reward : float; terminated : bool; truncated : bool }

type ('obs, 'action) t = {
  observation_space : Space.t;
  action_space : Space.t;
  reset : unit -> 'obs;
  step : 'action -> 'obs step;
}

(** Simulators whose observations are ndarrays of one precision. *)
module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  type nonrec step = A.arr step
  type nonrec 'action t = (A.arr, 'action) t

  val array : float array -> A.arr
  (** A one-dimensional observation. *)

  val simulator :
    observation_space:Space.t ->
    action_space:Space.t ->
    max_steps:int ->
    reset_state:(unit -> 's) ->
    observe:('s -> float array) ->
    advance:('s -> 'action -> 's * float * bool) ->
    'action t
  (** An environment from a pure simulator: [advance state action] returns the next state, the
      reward and whether the episode terminated. Episodes are truncated after [max_steps] steps.
      Observations are fresh arrays owned by the caller. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
