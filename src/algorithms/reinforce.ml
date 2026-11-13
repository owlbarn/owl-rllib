(* Episodic REINFORCE, optionally with a learned state-value baseline.
   A batch contains complete episodes. Without a critic, a time limit defines
   a finite-horizon Monte Carlo objective; with one, truncation bootstraps V. *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include Neural.Make (Graph)
  open Algodiff

  let act ~rng policy obs =
    Graph.run (batch [| obs |]) policy
    |> Distribution.Categorical.of_output
    |> Distribution.Categorical.sample ~rng
    |> values |> Dp.argmax

  let train ?baseline ?(gamma = 0.99) ?(learning_rate = 0.001)
      ?(batch_episodes = 8) ?(normalise_advantage = false)
      ~episodes ~rng ~policy (env : int E.t) =
    if episodes <= 0 || batch_episodes <= 0 then invalid_arg "Reinforce.train: episodes";
    let n = n_actions env in
    let optimiser = Stepper.init (adam learning_rate) (Graph.mkpri policy) in
    let critic_stepper = Option.map (fun nn -> Stepper.init (adam learning_rate) (Graph.mkpri nn)) baseline in
    let curve = Array.make episodes 0. in
    let observations = ref [] and actions = ref [] and targets = ref [] and discounts = ref [] in
    for episode = 0 to episodes - 1 do
      let rec rollout obs os acts rs =
        let action = act ~rng policy obs in
        let st = env.step action in
        let os = A.copy obs :: os and acts = action :: acts and rs = st.reward :: rs in
        if st.terminated || st.truncated then os, acts, rs, st
        else rollout st.obs os acts rs
      in
      let os, acts, rs, last = rollout (env.reset ()) [] [] [] in
      let rewards = Array.of_list (List.rev rs) in
      curve.(episode) <- Array.fold_left ( +. ) 0. rewards;
      let last_value = match baseline with
        | Some nn when not last.terminated -> (predict nn last.obs).(0)
        | _ -> 0.
      in
      let returns = Buffer.returns ~gamma ~last_value rewards in
      observations := !observations @ List.rev os;
      actions := !actions @ List.rev acts;
      targets := !targets @ Array.to_list returns;
      discounts := !discounts @ List.init (Array.length rewards) (fun t -> gamma ** float_of_int t);
      if (episode + 1) mod batch_episodes = 0 || episode + 1 = episodes then (
        let obs = batch (Array.of_list !observations) in
        let returns = Array.of_list !targets in
        let advantages = match baseline with
          | None -> Array.copy returns
          | Some nn ->
            let v = Graph.run obs nn |> unpack_arr |> values in
            Array.map2 ( -. ) returns v
        in
        (* Optional teaching experiment: batch normalization changes the estimator. *)
        if normalise_advantage then (
          let count = float_of_int (Array.length advantages) in
          let mean = Array.fold_left ( +. ) 0. advantages /. count in
          let variance = Array.fold_left (fun acc x -> acc +. (x -. mean) ** 2.) 0. advantages /. count in
          Array.iteri (fun i x -> advantages.(i) <- (x -. mean) /. (sqrt variance +. 1e-8)) advantages);
        let weights = Array.map2 ( *. ) advantages (Array.of_list !discounts) |> column in
        let logits, _ = Graph.forward policy obs in
        let logp = Distribution.Categorical.log_prob (Distribution.Categorical.of_output logits)
                     (one_hot n (Array.of_list !actions)) in
        let count = float_of_int (if (episode + 1) mod batch_episodes = 0 then batch_episodes else (episode + 1) mod batch_episodes) in
        let loss = Maths.(neg (sum' (logp * weights)) / scalar count) in
        update policy optimiser loss;
        (match baseline, critic_stepper with
         | Some nn, Some st ->
           let v, _ = Graph.forward nn obs in
           update nn st Maths.(mean (sqr (v - column returns)))
         | _ -> ());
        observations := []; actions := []; targets := []; discounts := [])
    done;
    curve
end
module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
