(* Model-free prediction and control with tables (Sutton & Barto, chapters 5-6).

   A policy is a function from state to action; it may sample internally. *)

(* argmax with ties broken at random *)
let argmax rng xs =
  let best = Array.fold_left Float.max neg_infinity xs in
  let ties = List.filter (fun i -> xs.(i) = best) (List.init (Array.length xs) Fun.id) in
  List.nth ties (Random.State.int rng (List.length ties))


let epsilon_greedy rng ~epsilon qs =
  if Random.State.float rng 1. < epsilon
  then Random.State.int rng (Array.length qs)
  else argmax rng qs


(* first-visit Monte Carlo prediction *)
let mc_prediction ~gamma ~episodes ~policy (env : Env.t) =
  let v = Array.make env.n_states 0. in
  let visits = Array.make env.n_states 0 in
  for _ = 1 to episodes do
    (* trajectory as (state, reward received after leaving it), latest first *)
    let rec rollout s acc =
      let st = env.step (policy s) in
      let acc = (s, st.reward) :: acc in
      if st.terminated || st.truncated then acc else rollout st.state acc
    in
    let trajectory = rollout (env.reset ()) [] in
    (* walk backwards; an earlier visit of the same state overwrites a later one *)
    let first = Hashtbl.create 16 in
    let g = ref 0. in
    List.iter
      (fun (s, r) ->
        g := r +. (gamma *. !g);
        Hashtbl.replace first s !g)
      trajectory;
    Hashtbl.iter
      (fun s g ->
        visits.(s) <- visits.(s) + 1;
        v.(s) <- v.(s) +. ((g -. v.(s)) /. float_of_int visits.(s)))
      first
  done;
  v


(* TD(0) prediction *)
let td0_prediction ~gamma ~alpha ~episodes ~policy (env : Env.t) =
  let v = Array.make env.n_states 0. in
  for _ = 1 to episodes do
    let rec go s =
      let st = env.step (policy s) in
      let target = st.reward +. if st.terminated then 0. else gamma *. v.(st.state) in
      v.(s) <- v.(s) +. (alpha *. (target -. v.(s)));
      if not (st.terminated || st.truncated) then go st.state
    in
    go (env.reset ())
  done;
  v


(* Shared loop of SARSA and Q-learning. [bootstrap q s' a'] is the value used
   for the next state, where [a'] is the action the agent will actually take.
   Returns the action values and the undiscounted return of every episode. *)
let control ~bootstrap rng ~gamma ~alpha ~epsilon ~episodes (env : Env.t) =
  let q = Array.make_matrix env.n_states env.n_actions 0. in
  let returns =
    Array.init episodes (fun _ ->
        let rec go s a total =
          let st = env.step a in
          let a' = epsilon_greedy rng ~epsilon q.(st.state) in
          let target =
            st.reward +. if st.terminated then 0. else gamma *. bootstrap q st.state a'
          in
          q.(s).(a) <- q.(s).(a) +. (alpha *. (target -. q.(s).(a)));
          let total = total +. st.reward in
          if st.terminated || st.truncated then total else go st.state a' total
        in
        let s = env.reset () in
        go s (epsilon_greedy rng ~epsilon q.(s)) 0.)
  in
  q, returns


(* on-policy: bootstrap from the action that will be taken *)
let sarsa rng = control ~bootstrap:(fun q s a -> q.(s).(a)) rng

(* off-policy: bootstrap from the best action *)
let q_learning rng =
  control ~bootstrap:(fun q s _ -> Array.fold_left Float.max neg_infinity q.(s)) rng
