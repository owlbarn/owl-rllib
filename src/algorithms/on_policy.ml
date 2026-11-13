(* Shared rollout collection and tensor boundaries for A2C / PPO. *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include Neural.Make (Graph)
  open Algodiff
  module R = Rollout.Make (A)
  module P = Policy.Make (Algodiff)
  type state = { mutable obs : A.arr array; totals : float array; mutable completed : float list }
  let start env = { obs = Vec_env.reset env; totals = Array.make (Vec_env.length env) 0.; completed = [] }

  let collect ~horizon ~gamma ~lambda ~rng ~policy ~critic ~distribution env state =
    let n = Vec_env.length env in
    let data = Array.init horizon (fun _ ->
      let input = batch state.obs in
      let output = Graph.run input policy in
      let v = Graph.run input critic |> unpack_arr |> values in
      let samples = distribution.Policy.sample ~rng output in
      let logp = distribution.log_prob output (Arr samples) |> unpack_arr |> values in
      let old_obs = state.obs in
      let result = Vec_env.step env (distribution.actions samples) in
      let final_obs = Array.map (fun st -> st.Ndenv.obs) result.transitions in
      let next_v = Graph.run (batch final_obs) critic |> unpack_arr |> values in
      state.obs <- result.obs;
      Array.mapi (fun i st ->
        state.totals.(i) <- state.totals.(i) +. st.Ndenv.reward;
        if st.terminated || st.truncated then (
          state.completed <- state.totals.(i) :: state.completed;
          state.totals.(i) <- 0.);
        { R.obs = A.copy old_obs.(i); action = P.row samples i; log_prob = logp.(i)
        ; value = v.(i); reward = st.reward; next_value = next_v.(i)
        ; terminated = st.terminated; truncated = st.truncated }) result.transitions)
      |> Array.to_list |> Array.concat in
    R.create ~n_envs:n ~gamma ~lambda data

  let normalise xs =
    let n = float_of_int (Array.length xs) in
    let mean = Array.fold_left ( +. ) 0. xs /. n in
    let var = Array.fold_left (fun sum x -> sum +. (x -. mean) ** 2.) 0. xs /. n in
    Array.map (fun x -> (x -. mean) /. (sqrt var +. 1e-8)) xs

  (* One global norm across a network's parameters, not per-tensor clipping.
     Actor and critic are independent networks and are clipped independently. *)
  (* [step_scale] multiplies the optimiser's step; for Adam this is the same
     as multiplying the learning rate, which is how PPO anneals it. *)
  let update_clipped ?(step_scale = 1.) ~max_grad_norm nn stepper loss =
    let ws, gs = Graph.backward nn loss in
    let norm2 = Array.fold_left (fun total layer -> Array.fold_left (fun total g ->
      total +. (Maths.sum' (Maths.sqr g) |> unpack_flt)) total layer) 0. gs in
    let scale = min 1. (max_grad_norm /. (sqrt norm2 +. 1e-8)) in
    let gs = Array.map (Array.map (fun g -> Maths.(g * scalar scale))) gs in
    let ws' = Stepper.step stepper ws gs in
    let ws' = if step_scale = 1. then ws'
      else Array.map2 (Array.map2 (fun w w' -> Maths.(primal' w + scalar step_scale * (w' - primal' w)))) ws ws' in
    Graph.update nn ws';
    Gc.minor ()

  let select rollout indices =
    let data = Array.map (Array.get rollout.R.data) indices in
    batch (Array.map (fun x -> x.R.obs) data),
    batch (Array.map (fun x -> x.R.action) data),
    column (Array.map (fun x -> x.R.log_prob) data),
    Array.map (Array.get rollout.advantages) indices,
    column (Array.map (Array.get rollout.returns) indices)

  let old_values rollout indices = column (Array.map (fun i -> rollout.R.data.(i).R.value) indices)
end
