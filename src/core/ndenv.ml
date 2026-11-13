(* Array observations, with the action type kept explicit: int for discrete
   control, ndarray for continuous control. The tabular Env is separate. *)
type 'obs step =
  { obs : 'obs; reward : float; terminated : bool; truncated : bool }

type ('obs, 'action) t =
  { observation_space : Space.t
  ; action_space : Space.t
  ; reset : unit -> 'obs
  ; step : 'action -> 'obs step
  }

module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  type nonrec step = A.arr step
  type nonrec 'action t = (A.arr, 'action) t

  let array xs = A.of_array (Array.map A.float_to_elt xs) [| Array.length xs |]

  (* Internal simulator state is float64; only observations are cast to S / D.
     Each observation is owned by the caller and remains valid after step. *)
  let simulator ~observation_space ~action_space ~max_steps ~reset_state ~observe ~advance =
    if max_steps <= 0 then invalid_arg "Ndenv.simulator: max_steps must be positive";
    let state = ref None and steps = ref 0 and ended = ref true in
    let reset () =
      let s = reset_state () in
      state := Some s;
      steps := 0;
      ended := false;
      array (observe s)
    in
    let step action =
      if !ended then invalid_arg "Ndenv.step: reset before stepping a new episode";
      let s, reward, terminated = advance (Option.get !state) action in
      state := Some s;
      incr steps;
      (* Gymnasium TimeLimit can report both flags on the last step. *)
      let truncated = !steps >= max_steps in
      ended := terminated || truncated;
      { obs = array (observe s); reward; terminated; truncated }
    in
    { observation_space; action_space; reset; step }
end

module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
