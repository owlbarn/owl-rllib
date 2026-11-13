open Owl_rllib
module A = Owl.Neural.S.Algodiff.A
module AD = Owl.Neural.D.Algodiff
module N = Neural.Make (Owl.Neural.S.Graph)

let check label eps x y = Alcotest.(check (float eps)) label x y
let rng () = Random.State.make [| 17 |]

let trajectory name make () =
  let ic = open_in ("fixtures/" ^ name ^ "_steps.txt") in
  let current = ref None in
  (try while true do
    let fields = input_line ic |> String.split_on_char ' ' in
    match fields with
    | "C" :: limit :: state ->
      let env = make (int_of_string limit) (Array.of_list (List.map float_of_string state)) in
      ignore (env.Ndenv.reset ()); current := Some env
    | "T" :: action :: reward :: terminated :: truncated :: obs ->
      let st = (Option.get !current).Ndenv.step (float_of_string action) in
      check "reward" 1e-10 (float_of_string reward) st.reward;
      Alcotest.(check bool) "terminated" (terminated = "1") st.terminated;
      Alcotest.(check bool) "truncated" (truncated = "1") st.truncated;
      List.iteri (fun i value -> check "observation" 3e-6 (float_of_string value) (A.get st.obs [| i |])) obs
    | _ -> failwith "invalid fixture"
  done with End_of_file -> close_in ic)

let discrete make limit state =
  let env : int Ndenv.S.t = make ?max_steps:(Some limit) ?initial_state:(Some state) ~rng:(rng ()) () in
  { env with step = (fun a -> env.step (int_of_float a)) }

let pendulum limit state =
  let env = Pendulum.make ?max_steps:(Some limit) ?initial_state:(Some state) ~rng:(rng ()) () in
  { env with step = (fun a -> env.step (A.of_array [| a |] [| 1 |])) }

let ownership () =
  let env = Cartpole.make ~rng:(rng ()) () in
  let obs = env.reset () in
  let before = A.copy obs in
  ignore (env.step 1);
  Alcotest.(check bool) "old observation survives step" true (Owl.Dense.Ndarray.S.to_array obs = Owl.Dense.Ndarray.S.to_array before);
  let boundary = Cartpole.make ~max_steps:1 ~initial_state:[| 2.39; 1.; 0.; 0. |] ~rng:(rng ()) () in
  ignore (boundary.reset ());
  let last = boundary.step 1 in
  Alcotest.(check bool) "terminal at time limit has both flags" true (last.terminated && last.truncated);
  Alcotest.check_raises "must reset after episode" (Invalid_argument "Ndenv.step: reset before stepping a new episode")
    (fun () -> ignore (boundary.step 0));
  let streams = Rng.create 42 in
  let expected = Random.State.copy streams.env in
  ignore (Random.State.bits streams.eval);
  Rng.init_network streams;
  check "network/evaluation do not consume env RNG" 0.
    (Random.State.float expected 1.) (Random.State.float streams.env 1.)

let replay () =
  let b = Buffer.S.create 2 in
  let obs = A.of_array [| 1.; 2. |] [| 2 |] in
  let add reward = Buffer.S.add b { obs; next_obs = obs; action = 1; reward; terminated = false; truncated = true } in
  add 1.; add 2.; add 3.;
  A.set obs [| 0 |] 99.;
  let samples = Buffer.S.sample ~rng:(rng ()) ~batch_size:100 b in
  Array.iter (fun (tr : Buffer.S.transition) ->
    check "owned observation" 0. 1. (A.get tr.obs [| 0 |]);
    Alcotest.(check bool) "oldest overwritten" true (tr.reward = 2. || tr.reward = 3.)) samples;
  Alcotest.(check int) "capacity" 2 (Buffer.S.length b);
  check "terminal does not bootstrap" 0. 2. (Buffer.td_target ~gamma:0.9 ~reward:2. ~terminated:true ~next_value:10.);
  check "truncation bootstraps" 1e-12 11. (Buffer.td_target ~gamma:0.9 ~reward:2. ~terminated:false ~next_value:10.);
  let g = Buffer.returns ~gamma:0.5 ~last_value:4. [| 1.; 2. |] in
  Alcotest.(check (array (float 1e-12))) "returns" [| 3.; 4. |] g;
  Alcotest.(check (array int)) "multidimensional batch"
    [| 2; 2; 3 |] (A.shape (Buffer.S.batch [| A.zeros [| 2; 3 |]; A.ones [| 2; 3 |] |]))

module Check = Owl_algodiff_check.Make (AD)
let tensor rows = AD.Arr (AD.A.of_arrays rows)
let scalar x = AD.F x
let gradient name f x =
  let shape = AD.shape x in
  let directions = Array.init (Array.fold_left ( * ) 1 shape) (fun j ->
    AD.Arr (AD.A.init shape (fun i -> if i = j then 1. else 0.))) in
  let ok, _ = Check.Reverse.check ~threshold:1e-6 ~eps:1e-4 ~order:`fourth
      ~f ~directions [| x |] in
  Alcotest.(check bool) name true ok

let distributions () =
  let module C = Dist.D.Categorical in
  let module G = Dist.D.Gaussian in
  let x = tensor [| [| 0.2; -0.3; 0.7 |]; [| -0.1; 0.5; 0.2 |] |] in
  let y = tensor [| [| 0.; 1.; 0. |]; [| 1.; 0.; 0. |] |] in
  let q = C.of_output (tensor [| [| 0.1; 0.4; 0.2 |]; [| 0.3; 0.2; 0.1 |] |]) in
  gradient "categorical log_prob" (fun x -> AD.Maths.sum' (C.log_prob (C.of_output x) y)) x;
  gradient "categorical entropy" (fun x -> AD.Maths.sum' (C.entropy (C.of_output x))) x;
  gradient "categorical KL" (fun x -> AD.Maths.sum' (C.kl (C.of_output x) q)) x;
  let uniform = C.of_output (tensor [| [| 1000.; 1000.; 1000. |] |]) in
  check "stable entropy" 1e-10 (log 3.) (AD.unpack_arr (C.entropy uniform) |> fun a -> AD.A.get a [| 0; 0 |]);
  let mu = tensor [| [| 0.2; -0.1 |]; [| 0.7; 0.5 |] |] in
  let ls = tensor [| [| -0.3; 0.2 |] |] in
  let action = tensor [| [| 0.1; 0.4 |]; [| 0.3; -0.2 |] |] in
  let reference = G.create ~mean:(AD.Maths.(mu + scalar 0.3)) ~log_std:(AD.Maths.(ls + scalar 0.1)) in
  List.iter (fun (name, loss) ->
    gradient (name ^ " mean") (fun mean -> loss (G.create ~mean ~log_std:ls)) mu;
    gradient (name ^ " scale") (fun log_std -> loss (G.create ~mean:mu ~log_std)) ls)
    [ "gaussian log_prob", (fun p -> AD.Maths.sum' (G.log_prob p action))
    ; "gaussian entropy", (fun p -> AD.Maths.sum' (G.entropy p))
    ; "gaussian KL", (fun p -> AD.Maths.sum' (G.kl p reference)) ];
  let p = G.create ~mean:mu ~log_std:ls in
  Alcotest.(check bool) "samples reproducible" true
    (Owl.Dense.Ndarray.D.to_array (G.sample ~rng:(rng ()) p) = Owl.Dense.Ndarray.D.to_array (G.sample ~rng:(rng ()) p));
  check "identical KL" 1e-12 0. (AD.Maths.sum' (G.kl p p) |> AD.unpack_flt)

let reference_values () =
  let expected = Hashtbl.create 20 in
  let ic = open_in "fixtures/function_approximation_numerics.txt" in
  (try while true do
    match input_line ic |> String.split_on_char ' ' with
    | name :: values -> Hashtbl.add expected name (Array.of_list (List.map float_of_string values))
    | _ -> assert false
  done with End_of_file -> close_in ic);
  let compare name value =
    let actual = AD.unpack_arr value |> Owl.Dense.Ndarray.D.to_array in
    Alcotest.(check (array (float 1e-10))) name (Hashtbl.find expected name) actual
  in
  let c = Dist.D.Categorical.of_output in
  let x = tensor [| [| 0.2; -0.3; 0.7 |]; [| -0.1; 0.5; 0.2 |] |] in
  let action = tensor [| [| 0.; 1.; 0. |]; [| 1.; 0.; 0. |] |] in
  let q = c (tensor [| [| 0.1; 0.4; 0.2 |]; [| 0.3; 0.2; 0.1 |] |]) in
  List.iter (fun (name, f) ->
    compare name (f x);
    compare (name ^ "_grad0") (AD.grad (fun x -> AD.Maths.sum' (f x)) x))
    [ "categorical_log_prob", (fun x -> Dist.D.Categorical.log_prob (c x) action)
    ; "categorical_entropy", (fun x -> Dist.D.Categorical.entropy (c x))
    ; "categorical_kl", (fun x -> Dist.D.Categorical.kl (c x) q) ];
  let mu = tensor [| [| 0.2; -0.1 |]; [| 0.7; 0.5 |] |] in
  let ls = tensor [| [| -0.3; 0.2 |] |] in
  let a = tensor [| [| 0.1; 0.4 |]; [| 0.3; -0.2 |] |] in
  let module G = Dist.D.Gaussian in
  let q = G.create ~mean:AD.Maths.(mu + scalar 0.3) ~log_std:AD.Maths.(ls + scalar 0.1) in
  List.iter (fun (name, f) ->
    compare name (f (G.create ~mean:mu ~log_std:ls));
    compare (name ^ "_grad0") (AD.grad (fun mean -> AD.Maths.sum' (f (G.create ~mean ~log_std:ls))) mu);
    compare (name ^ "_grad1") (AD.grad (fun log_std -> AD.Maths.sum' (f (G.create ~mean:mu ~log_std))) ls))
    [ "gaussian_log_prob", (fun p -> G.log_prob p a); "gaussian_kl", (fun p -> G.kl p q) ];
  compare "gaussian_entropy" (G.entropy (G.create ~mean:mu ~log_std:ls));
  compare "gaussian_entropy_grad0" (AD.grad (fun log_std -> AD.Maths.sum' (G.entropy (G.create ~mean:mu ~log_std))) ls)

let bandit () : int Ndenv.S.t =
  { observation_space = Space.box ~low:[| 1. |] ~high:[| 1. |] ~shape:[| 1 |]
  ; action_space = Space.Discrete 2
  ; reset = (fun () -> A.ones [| 1 |])
  ; step = (fun a -> { obs = A.ones [| 1 |]; reward = (if a = 1 then 1. else -1.); terminated = true; truncated = false }) }

let weights nn = Owl.Neural.S.Graph.mkpri nn |> Array.map (Array.map (fun w -> Owl.Neural.S.Algodiff.unpack_arr w |> Owl.Dense.Ndarray.S.to_array))

let reinforce () =
  let run () =
    let streams = Rng.create 7 in
    let policy = N.mlp ~seed:streams.network_seed ~shape:[| 1 |] ~hidden:[] ~outputs:2 in
    let critic = N.mlp ~seed:(streams.network_seed + 1) ~shape:[| 1 |] ~hidden:[] ~outputs:1 in
    let curve = Reinforce.S.train ~baseline:critic ~learning_rate:0.02 ~episodes:200
        ~batch_episodes:4 ~rng:streams.action ~policy (bandit ()) in
    let logits = N.predict policy (A.ones [| 1 |]) in
    Alcotest.(check bool) "policy learns rewarding action" true (logits.(1) -. logits.(0) > 2.);
    curve, weights policy, weights critic
  in
  Alcotest.(check bool) "REINFORCE deterministic" true (run () = run ())

let dqn () =
  let run () =
    let streams = Rng.create 8 in
    let network = N.mlp ~seed:streams.network_seed ~shape:[| 1 |] ~hidden:[] ~outputs:2 in
    let config = { Dqn.S.default with learning_rate = 0.02; learning_starts = 8;
      train_every = 1; batch_size = 8; target_every = 20; exploration_steps = 100 } in
    let curve = Dqn.S.train ~config ~steps:250 ~rngs:streams ~network (bandit ()) in
    let q = N.predict network (A.ones [| 1 |]) in
    check "Q(bad)" 0.05 (-1.) q.(0); check "Q(good)" 0.05 1. q.(1);
    curve, weights network
  in
  Alcotest.(check bool) "DQN deterministic" true (run () = run ())

let time_limit_learning () =
  (* Same observation after each timeout: Q must include future episodes' value.
     Treating truncation as terminal instead would converge to 1, not 2. *)
  let module ND = Neural.Make (Owl.Neural.D.Graph) in
  let streams = Rng.create 31 in
  let network = ND.mlp ~seed:streams.network_seed ~shape:[| 1 |] ~hidden:[] ~outputs:1 in
  let env : int Ndenv.D.t =
    { observation_space = Space.box ~low:[| 1. |] ~high:[| 1. |] ~shape:[| 1 |]
    ; action_space = Space.Discrete 1
    ; reset = (fun () -> AD.A.ones [| 1 |])
    ; step = (fun _ -> { obs = AD.A.ones [| 1 |]; reward = 1.; terminated = false; truncated = true }) }
  in
  let config = { Dqn.D.default with gamma = 0.5; learning_rate = 0.02;
      learning_starts = 1; train_every = 1; batch_size = 4; target_every = 10 } in
  ignore (Dqn.D.train ~config ~steps:400 ~rngs:streams ~network env);
  check "float64 DQN bootstraps time limits" 0.02 2. (ND.predict network (env.reset ())).(0);
  let config = { config with target_network = false } in
  ignore (Dqn.D.train ~config ~steps:100 ~rngs:streams ~network env);
  check "no-target ablation also bootstraps" 0.02 2. (ND.predict network (env.reset ())).(0)

let linear_learning () =
  let weights, _ = Linear.S.train ~features:(fun _ -> [| 0 |]) ~n_features:1
      ~gamma:0.9 ~alpha:0.1 ~epsilon:0.5 ~episodes:200 ~rng:(rng ()) (bandit ()) in
  check "linear good action" 0.02 1. weights.(1).(0);
  check "linear bad action" 0.02 (-1.) weights.(0).(0)

let sampling () =
  let module C = Dist.D.Categorical in
  let logp = C.of_output (tensor [| [| log 0.2; log 0.8 |] |]) in
  let rng = rng () in
  let hits = ref 0. in
  for _ = 1 to 10_000 do
    hits := !hits +. AD.A.get (C.sample ~rng logp) [| 0; 1 |]
  done;
  check "categorical frequency" 0.02 0.8 (!hits /. 10_000.);
  let p = Dist.D.Gaussian.create ~mean:(tensor [| [| 2. |] |]) ~log_std:(tensor [| [| log 3. |] |]) in
  let sum = ref 0. and sum2 = ref 0. in
  for _ = 1 to 10_000 do
    let x = AD.A.get (Dist.D.Gaussian.sample ~rng p) [| 0; 0 |] in
    sum := !sum +. x; sum2 := !sum2 +. x *. x
  done;
  let mean = !sum /. 10_000. in
  check "gaussian mean" 0.1 2. mean;
  check "gaussian variance" 0.3 9. (!sum2 /. 10_000. -. mean *. mean)

let () = Alcotest.run "function approximation"
  [ "control", [ "cartpole", `Quick, trajectory "cartpole" (discrete Cartpole.make)
               ; "mountain car", `Quick, trajectory "mountain_car" (discrete Mountain_car.make)
               ; "acrobot", `Quick, trajectory "acrobot" (discrete Acrobot.make)
               ; "pendulum", `Quick, trajectory "pendulum" pendulum
               ; "ownership and RNG", `Quick, ownership ]
  ; "numerics", [ "replay and returns", `Quick, replay; "distributions", `Quick, distributions; "PyTorch reference", `Quick, reference_values; "sampling", `Quick, sampling ]
  ; "learning", [ "reinforce", `Quick, reinforce; "dqn", `Quick, dqn; "float64 time limits", `Quick, time_limit_learning; "linear", `Quick, linear_learning ] ]
