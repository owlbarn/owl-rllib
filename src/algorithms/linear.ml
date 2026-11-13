(* Semi-gradient SARSA / Q-learning with sparse binary features (tile coding).
   Step size is divided by the number of active features. *)
module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  module E = Ndenv.Make (A)
  let values weights active =
    Array.map (fun w -> Array.fold_left (fun sum i -> sum +. w.(i)) 0. active) weights

  let train ?(off_policy = false) ~features ~n_features ~gamma ~alpha ~epsilon
      ~episodes ~rng (env : int E.t) =
    let n_actions = match env.action_space with Space.Discrete n -> n | _ -> invalid_arg "Linear: discrete actions required" in
    let weights = Array.make_matrix n_actions n_features 0. in
    let act active = Tabular.epsilon_greedy rng ~epsilon (values weights active) in
    let curve = Array.init episodes (fun _ ->
      let rec go active action total =
        let st = env.step action in
        let next = features st.obs in
        let next_action = act next in
        let qs = values weights next in
        let next_value = if off_policy then Array.fold_left Float.max neg_infinity qs else qs.(next_action) in
        let target = Buffer.td_target ~gamma ~reward:st.reward ~terminated:st.terminated ~next_value in
        let delta = target -. (values weights active).(action) in
        let step = alpha /. float_of_int (Array.length active) *. delta in
        Array.iter (fun i -> weights.(action).(i) <- weights.(action).(i) +. step) active;
        let total = total +. st.reward in
        if st.terminated || st.truncated then total else go next next_action total
      in
      let active = features (env.reset ()) in
      go active (act active) 0.)
    in
    weights, curve
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
