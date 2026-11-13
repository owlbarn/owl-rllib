open Owl_rllib

let close ?(eps = 1e-6) a b = Float.abs (a -. b) < eps

let max_diff a b =
  let d = ref 0. in
  Array.iteri (fun i x -> d := Float.max !d (Float.abs (x -. b.(i)))) a;
  !d


(* --- environments against Gymnasium's transition tables --- *)

(* merge duplicate outcomes and sort, so that the order of entries is irrelevant *)
let normalise transitions =
  let tbl = Hashtbl.create 4 in
  List.iter
    (fun (tr : Mdp.transition) ->
      let key = tr.next, tr.reward, tr.terminal in
      Hashtbl.replace tbl key (tr.prob +. Option.value ~default:0. (Hashtbl.find_opt tbl key)))
    transitions;
  Hashtbl.fold (fun key prob acc -> (key, prob) :: acc) tbl [] |> List.sort compare


let load_fixture name : Mdp.t =
  let ic = Scanf.Scanning.open_in ("fixtures/" ^ name ^ ".txt") in
  let n_states, n_actions = Scanf.bscanf ic "N %d %d\n" (fun s a -> s, a) in
  let transitions = Array.make_matrix n_states n_actions [] in
  let initial = Array.make n_states 0. in
  (try
     while true do
       Scanf.bscanf ic "%c " (function
           | 'T' ->
             Scanf.bscanf ic "%d %d %f %d %f %d\n" (fun s a prob next reward terminal ->
                 transitions.(s).(a)
                   <- { Mdp.prob; next; reward; terminal = terminal = 1 } :: transitions.(s).(a))
           | 'I' -> Scanf.bscanf ic "%d %f\n" (fun s p -> initial.(s) <- p)
           | c -> failwith (Printf.sprintf "unexpected record %c" c))
     done
   with
   | End_of_file -> ());
  { n_states; n_actions; transitions; initial }


let same_as_gym name (ours : Mdp.t) () =
  let gym = load_fixture name in
  Alcotest.(check int) "states" gym.n_states ours.n_states;
  Alcotest.(check int) "actions" gym.n_actions ours.n_actions;
  Alcotest.(check bool) "initial distribution" true (max_diff gym.initial ours.initial < 1e-12);
  for s = 0 to gym.n_states - 1 do
    for a = 0 to gym.n_actions - 1 do
      let g = normalise gym.transitions.(s).(a)
      and o = normalise ours.transitions.(s).(a) in
      let same =
        List.length g = List.length o
        && List.for_all2 (fun (k, p) (k', p') -> k = k' && close ~eps:1e-12 p p') g o
      in
      if not same then Alcotest.failf "transitions differ at state %d, action %d" s a
    done
  done


(* --- dynamic programming --- *)

(* Sutton & Barto, figure 4.1: value of the random policy on the 4x4 grid *)
let sutton_values =
  [| 0.; -14.; -20.; -22.; -14.; -18.; -20.; -20.; -20.; -20.; -18.; -14.; -22.; -20.; -14.; 0. |]


let policy_evaluation () =
  let mdp = (Gridworld.sutton_4x4 ()).mdp in
  let random = Array.make_matrix mdp.n_states 4 0.25 in
  let v = Dp.policy_evaluation ~gamma:1. mdp random in
  Alcotest.(check bool) "matches the book" true (max_diff v sutton_values < 1e-6)


let iteration_methods_agree mdp () =
  let gamma = 0.95 in
  let pi_policy, pi_v = Dp.policy_iteration ~gamma mdp in
  let _, vi_v = Dp.value_iteration ~gamma mdp in
  Alcotest.(check bool) "same optimal values" true (max_diff pi_v vi_v < 1e-6);
  (* the policy found by policy iteration achieves those values *)
  let v = Dp.policy_evaluation ~gamma mdp (Dp.one_hot mdp pi_policy) in
  Alcotest.(check bool) "policy is optimal" true (max_diff v vi_v < 1e-6)


(* the optimal route along the cliff takes 13 steps *)
let cliff_optimum () =
  let mdp = Cliff_walking.make () in
  let _, v = Dp.value_iteration ~gamma:1. mdp in
  Alcotest.(check bool) "13 steps" true (close v.(Cliff_walking.start) (-13.))


(* --- model-free methods --- *)

let random_policy rng n_actions _ = Random.State.int rng n_actions

let prediction estimate eps () =
  let mdp = Frozen_lake.make () in
  let gamma = 0.95 in
  let exact = Dp.policy_evaluation ~gamma mdp (Array.make_matrix mdp.n_states 4 0.25) in
  let rng = Random.State.make [| 1 |] in
  let v = estimate ~gamma ~policy:(random_policy rng 4) (Env.of_mdp ~rng mdp) in
  let err = max_diff v exact in
  if err >= eps then Alcotest.failf "largest error %.4f, expected below %.4f" err eps


let mc = Tabular.mc_prediction ~episodes:50_000

let td0 = Tabular.td0_prediction ~alpha:0.01 ~episodes:50_000

(* in a deterministic world Q-learning converges to the optimal action values *)
let q_learning_optimal () =
  let mdp = (Gridworld.make [| "...."; ".#.."; "...G" |]).mdp in
  let gamma = 0.9 in
  let _, v_star = Dp.value_iteration ~gamma mdp in
  let q_star = Dp.q_from_v ~gamma mdp v_star in
  let rng = Random.State.make [| 2 |] in
  let env = Env.of_mdp ~max_steps:100 ~rng mdp in
  let q, _ = Tabular.q_learning rng ~gamma ~alpha:0.5 ~epsilon:0.3 ~episodes:3000 env in
  let err = Array.fold_left Float.max 0. (Array.map2 max_diff q q_star) in
  if err >= 1e-3 then Alcotest.failf "largest error %.5f" err


(* Sutton & Barto, example 6.6 *)
let cliff ~seed algorithm =
  let mdp = Cliff_walking.make () in
  let rng = Random.State.make [| seed |] in
  let env = Env.of_mdp ~rng mdp in
  let q, returns = algorithm rng ~gamma:1. ~alpha:0.5 ~epsilon:0.1 ~episodes:500 env in
  let last = Array.sub returns 400 100 in
  mdp, q, Array.fold_left ( +. ) 0. last /. 100.


let greedy_return (mdp : Mdp.t) q =
  let policy = Dp.greedy q in
  let rec go s total steps =
    let tr = List.hd mdp.transitions.(s).(policy.(s)) in
    let total = total +. tr.reward in
    if tr.terminal || steps > 100 then total else go tr.next total (steps + 1)
  in
  go Cliff_walking.start 0. 0


let cliff_example () =
  let mdp, q, q_online = cliff ~seed:3 Tabular.q_learning in
  let _, _, sarsa_online = cliff ~seed:3 Tabular.sarsa in
  Alcotest.(check bool) "Q-learning finds the shortest path" true (close (greedy_return mdp q) (-13.));
  if sarsa_online <= q_online
  then Alcotest.failf "online return: sarsa %.1f, q-learning %.1f" sarsa_online q_online


let deterministic () =
  let run () =
    let _, q, online = cliff ~seed:4 Tabular.sarsa in
    q, online
  in
  Alcotest.(check bool) "same seed, same result" true (run () = run ())


let truncation () =
  let mdp = (Gridworld.make [| "S.G" |]).mdp in
  let env = Env.of_mdp ~max_steps:3 ~rng:(Random.State.make [| 0 |]) mdp in
  ignore (env.reset ());
  let steps = List.init 3 (fun _ -> env.step 3) in
  let last = List.nth steps 2 in
  Alcotest.(check bool) "truncated, not terminated" true (last.truncated && not last.terminated);
  ignore (env.reset ());
  ignore (env.step 1);
  let st = env.step 1 in
  Alcotest.(check bool) "terminated, not truncated" true (st.terminated && not st.truncated)


let () =
  Alcotest.run
    "tabular"
    [ ( "environments"
      , [ "frozen lake", `Quick, same_as_gym "frozen_lake" (Frozen_lake.make ())
        ; "frozen lake, not slippery", `Quick, same_as_gym "frozen_lake_dry" (Frozen_lake.make ~slippery:false ())
        ; "cliff walking", `Quick, same_as_gym "cliff_walking" (Cliff_walking.make ())
        ; "taxi", `Quick, same_as_gym "taxi" (Taxi.make ())
        ; "truncation", `Quick, truncation
        ] )
    ; ( "dynamic programming"
      , [ "policy evaluation", `Quick, policy_evaluation
        ; "frozen lake", `Quick, iteration_methods_agree (Frozen_lake.make ())
        ; "taxi", `Quick, iteration_methods_agree (Taxi.make ())
        ; "cliff optimum", `Quick, cliff_optimum
        ] )
    ; ( "model-free"
      , [ "monte carlo prediction", `Quick, prediction mc 0.01
        ; "td(0) prediction", `Quick, prediction td0 0.02
        ; "q-learning is optimal", `Quick, q_learning_optimal
        ; "cliff example", `Quick, cliff_example
        ; "deterministic", `Quick, deterministic
        ] )
    ]
