(* Dynamic programming on a known MDP (Sutton & Barto, chapter 4). *)

(* expected return of taking action [a] in state [s], then following [v] *)
let backup ~gamma (mdp : Mdp.t) v s a =
  List.fold_left
    (fun acc (tr : Mdp.transition) ->
      let future = if tr.terminal then 0. else gamma *. v.(tr.next) in
      acc +. (tr.prob *. (tr.reward +. future)))
    0.
    mdp.transitions.(s).(a)


let q_from_v ~gamma (mdp : Mdp.t) v =
  Array.init mdp.n_states (fun s -> Array.init mdp.n_actions (backup ~gamma mdp v s))


let argmax xs =
  let best = ref 0 in
  Array.iteri (fun i x -> if x > xs.(!best) then best := i) xs;
  !best


let greedy q = Array.map argmax q

(* a deterministic policy as action probabilities *)
let one_hot (mdp : Mdp.t) policy =
  Array.map (fun a -> Array.init mdp.n_actions (fun b -> if a = b then 1. else 0.)) policy


(* sweep the states in place until the largest change is below [theta] *)
let sweep_until ~theta (mdp : Mdp.t) v update =
  let delta = ref infinity in
  while !delta >= theta do
    delta := 0.;
    for s = 0 to mdp.n_states - 1 do
      let v' = update s in
      delta := Float.max !delta (Float.abs (v' -. v.(s)));
      v.(s) <- v'
    done
  done


(* value of a stochastic policy, [policy.(s).(a)] = probability of [a] in [s] *)
let policy_evaluation ?(theta = 1e-10) ~gamma (mdp : Mdp.t) policy =
  let v = Array.make mdp.n_states 0. in
  sweep_until ~theta mdp v (fun s ->
      let total = ref 0. in
      Array.iteri (fun a p -> if p > 0. then total := !total +. (p *. backup ~gamma mdp v s a)) policy.(s);
      !total);
  v


let policy_iteration ?theta ~gamma (mdp : Mdp.t) =
  let rec improve policy =
    let v = policy_evaluation ?theta ~gamma mdp (one_hot mdp policy) in
    let q = q_from_v ~gamma mdp v in
    (* keep the current action on ties so that the loop terminates *)
    let policy' =
      Array.mapi (fun s a -> if q.(s).(argmax q.(s)) > q.(s).(a) +. 1e-12 then argmax q.(s) else a) policy
    in
    if policy' = policy then policy, v else improve policy'
  in
  improve (Array.make mdp.n_states 0)


let value_iteration ?(theta = 1e-10) ~gamma (mdp : Mdp.t) =
  let v = Array.make mdp.n_states 0. in
  sweep_until ~theta mdp v (fun s ->
      Array.init mdp.n_actions (backup ~gamma mdp v s) |> Array.fold_left Float.max neg_infinity);
  greedy (q_from_v ~gamma mdp v), v
