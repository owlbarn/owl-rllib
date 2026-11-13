(** VizDoom scenarios, run in a Python child process (see {!Owl_rllib_bridges.Gym} for how the
    bridge works).

    Action [i] presses button [i] alone; in [basic], [defend_the_center] and [health_gathering] that
    is move or turn left, move or turn right, and attack or move forward. *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  val make :
    ?frame_skip:int -> ?downsample:int -> seed:int -> string -> (A.arr, int) Owl_rllib.Ndenv.t
  (** [make ~seed scenario] loads [scenario.cfg] from VizDoom's scenarios.

      One step repeats the action for [frame_skip] tics (default 4) and returns their summed reward.
      The 160x120 grayscale screen is averaged over [downsample] x [downsample] blocks (default 2)
      and scaled to [[0, 1]]: observations have shape [[60; 80; 1]] by default, ready for
      {!Owl_rllib.Frame_stack}. An episode that reaches the scenario's time limit is truncated; one
      that ends otherwise (a kill, a death) is terminated. VizDoom renders no screen after the end,
      so the final observation repeats the last one. Raises [Invalid_argument] unless [downsample]
      divides 120 and 160. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
