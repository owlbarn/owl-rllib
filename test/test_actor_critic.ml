open Owl_rllib
module AD = Owl.Neural.D.Algodiff
module A = Owl.Dense.Ndarray.S
module N = Neural.Make (Owl.Neural.S.Graph)
let tensor xs = AD.Arr (AD.A.of_array xs [| Array.length xs; 1 |])
let reference =
  let tbl = Hashtbl.create 10 in
  let ic = open_in "fixtures/actor_critic_numerics.txt" in
  (try while true do
    match input_line ic |> String.split_on_char ' ' with
    | key :: xs -> Hashtbl.add tbl key (Array.of_list (List.map float_of_string xs))
    | _ -> assert false
  done with End_of_file -> close_in ic); tbl
let compare name xs = Alcotest.(check (array (float 1e-9))) name (Hashtbl.find reference name) xs
let flat x = AD.unpack_arr x |> Owl.Dense.Ndarray.D.to_array

let gae () =
  let advantages, returns = Rollout.gae ~n_envs:2 ~gamma:0.9 ~lambda:0.8
    ~rewards:[| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    ~values:[| 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9 |]
    ~next_values:[| 0.4; 0.5; 100.; 2.; 0.8; 0.9; 3.; 100. |]
    ~terminated:[| false; false; true; false; false; false; false; true |]
    ~truncated:[| false; false; false; true; false; false; false; true |] in
  compare "gae" advantages; compare "returns" returns

let losses () =
  let logp = tensor (Array.map log [| 0.5; 0.9; 1.1; 1.5; 0.5; 0.9; 1.1; 1.5 |]) in
  let f log_prob = Ppo.D.surrogate ~clip:0.2 ~log_prob ~old_log_prob:(tensor (Array.make 8 0.))
      ~advantages:(tensor [| 1.; 2.; 3.; 4.; -1.; -2.; -3.; -4. |]) in
  compare "ppo_loss" [| AD.unpack_flt (f logp) |];
  compare "ppo_grad" (AD.grad f logp |> flat);
  let logits = AD.Arr (AD.A.of_arrays [| [| 0.2; -0.1 |]; [| -0.3; 0.4 |] |]) in
  let f output =
    let distribution = Dist.D.Categorical.of_output output in
    A2c.D.actor_loss ~entropy_coef:0.01 ~entropy:(Dist.D.Categorical.entropy distribution)
      ~log_prob:(Dist.D.Categorical.log_prob distribution (AD.Arr (AD.A.of_arrays [| [| 1.; 0. |]; [| 0.; 1. |] |])))
      ~advantages:(tensor [| 2.; -1. |]) in
  compare "a2c_loss" [| AD.unpack_flt (f logits) |];
  compare "a2c_grad" (AD.grad f logits |> flat);
  let distribution = Policy.D.gaussian ~low:[| -2. |] ~high:[| 2. |] in
  let output = AD.Arr (AD.A.of_arrays [| [| 0.1; -0.3 |]; [| 0.2; 0.4 |] |]) in
  let f output = AD.Maths.sum' (distribution.log_prob output (tensor [| 0.7; -0.2 |])) in
  compare "gaussian_grad" (AD.grad f output |> flat);
  let raw = AD.A.of_arrays [| [| 4. |]; [| -3. |] |] in
  let clipped = distribution.actions raw in
  Alcotest.(check (float 0.)) "clip executed action" 2. (AD.A.get clipped.(0) [| 0 |]);
  Alcotest.(check (float 0.)) "preserve latent action" 4. (AD.A.get raw [| 0; 0 |])

let counter terminal () : int Ndenv.S.t =
  let resets = ref 0 in
  { observation_space = Space.box ~low:[| 0. |] ~high:[| 100. |] ~shape:[| 1 |]
  ; action_space = Space.Discrete 2
  ; reset = (fun () -> incr resets; A.of_array [| float_of_int !resets |] [| 1 |])
  ; step = (fun a -> { obs = A.of_array [| 99. |] [| 1 |]; reward = float_of_int a;
                      terminated = terminal; truncated = not terminal }) }

let vector_and_collector () =
  let env = Vec_env.create [| counter true (); counter false () |] in
  ignore (Vec_env.reset env);
  let st = Vec_env.step env [| 0; 1 |] in
  Alcotest.(check (float 0.)) "final observation" 99. (A.get st.transitions.(1).obs [| 0 |]);
  Alcotest.(check (float 0.)) "reset observation" 2. (A.get st.obs.(1) [| 0 |]);
  let policy = N.mlp ~seed:1 ~shape:[| 1 |] ~hidden:[] ~outputs:2 in
  let critic = N.mlp ~seed:2 ~shape:[| 1 |] ~hidden:[] ~outputs:1 in
  let module O = On_policy.Make (Owl.Neural.S.Graph) in
  let state = O.start env in
  let final_value = (N.predict critic (A.of_array [| 99. |] [| 1 |])).(0) in
  let rollout = O.collect ~horizon:2 ~gamma:0.9 ~lambda:0.95 ~rng:(Random.State.make [| 0 |])
      ~policy ~critic ~distribution:Policy.S.categorical env state in
  Alcotest.(check (float 1e-5)) "collector final value" final_value rollout.data.(1).next_value;
  Alcotest.(check (float 1e-5)) "terminal return"
    rollout.data.(0).reward rollout.returns.(0);
  Alcotest.(check (float 1e-5)) "truncated return"
    (rollout.data.(1).reward +. 0.9 *. final_value) rollout.returns.(1)

let bandit () : int Ndenv.S.t =
  { observation_space = Space.box ~low:[| 1. |] ~high:[| 1. |] ~shape:[| 1 |]
  ; action_space = Space.Discrete 2
  ; reset = (fun () -> A.ones [| 1 |])
  ; step = (fun a -> { obs = A.ones [| 1 |]; reward = (if a = 1 then 1. else -1.); terminated = true; truncated = false }) }
let weights nn = Owl.Neural.S.Graph.mkpri nn |> Array.map (Array.map (fun x -> Owl.Neural.S.Algodiff.unpack_arr x |> A.to_array))
let learning a2c () =
  let run () =
    let rngs = Rng.create 123 in
    let policy = N.mlp ~seed:rngs.network_seed ~shape:[| 1 |] ~hidden:[] ~outputs:2 in
    let critic = N.mlp ~seed:(rngs.network_seed + 1) ~shape:[| 1 |] ~hidden:[] ~outputs:1 in
    let env = Vec_env.create (Array.init 3 (fun _ -> bandit ())) in
    let curve = if a2c then
      A2c.S.train ~config:{ A2c.S.default with learning_rate = 0.02 } ~updates:100
        ~rngs ~policy ~critic ~distribution:Policy.S.categorical env
    else Ppo.S.train ~config:{ Ppo.S.default with horizon = 5; batch_size = 8; learning_rate = 0.01 }
        ~updates:30 ~rngs ~policy ~critic ~distribution:Policy.S.categorical env in
    let logits = N.predict policy (A.ones [| 1 |]) in
    Alcotest.(check bool) "learns good action" true (logits.(1) -. logits.(0) > 2.);
    curve, weights policy, weights critic
  in
  Alcotest.(check bool) "same seed, same curve/weights" true (run () = run ())

let continuous () =
  let module ND = Neural.Make (Owl.Neural.D.Graph) in
  let rngs = Rng.create 8 in
  let policy = ND.mlp ~seed:rngs.network_seed ~shape:[| 1 |] ~hidden:[] ~outputs:2 in
  let critic = ND.mlp ~seed:(rngs.network_seed + 1) ~shape:[| 1 |] ~hidden:[] ~outputs:1 in
  let make () : AD.A.arr Ndenv.D.t =
    { observation_space = Space.box ~low:[| 1. |] ~high:[| 1. |] ~shape:[| 1 |]
    ; action_space = Space.box ~low:[| -2. |] ~high:[| 2. |] ~shape:[| 1 |]
    ; reset = (fun () -> AD.A.ones [| 1 |])
    ; step = (fun a -> { obs = AD.A.ones [| 1 |]; reward = -.((AD.A.get a [| 0 |] -. 0.7) ** 2.); terminated = true; truncated = false }) }
  in
  let distribution = Policy.D.gaussian ~low:[| -2. |] ~high:[| 2. |] in
  let env = Vec_env.create (Array.init 4 (fun _ -> make ())) in
  ignore (Ppo.D.train ~config:{ Ppo.D.default with horizon = 16; batch_size = 32; learning_rate = 0.003 }
    ~updates:100 ~rngs ~policy ~critic ~distribution env);
  let result = ND.predict policy (AD.A.ones [| 1 |]) in
  Alcotest.(check (float 0.15)) "continuous optimum" 0.7 result.(0);
  Alcotest.(check bool) "learned exploration scale" true (result.(1) < -0.5)

let gradient_clipping () =
  let module O = On_policy.Make (Owl.Neural.D.Graph) in
  let nn = O.mlp ~seed:9 ~shape:[| 1 |] ~hidden:[] ~outputs:1 in
  let before = O.Graph.mkpri nn |> Array.map (Array.map (fun x -> AD.unpack_arr x |> AD.A.copy)) in
  let params = O.Params.config ~learning_rate:(O.Learning_Rate.Const 0.1) 1. in
  let stepper = O.Stepper.init params (O.Graph.mkpri nn) in
  let output, _ = O.Graph.forward nn (tensor [| 3. |]) in
  O.update_clipped ~max_grad_norm:0.5 nn stepper AD.Maths.(sum' output * AD.F 100.);
  let after = O.Graph.mkpri nn in
  let norm2 = ref 0. in
  Array.iteri (fun i layer -> Array.iteri (fun j x ->
    let old = Owl.Dense.Ndarray.D.to_array before.(i).(j) in
    let current = AD.unpack_arr x |> Owl.Dense.Ndarray.D.to_array in
    Array.iteri (fun k v -> norm2 := !norm2 +. (v -. old.(k)) ** 2.) current) layer) after;
  Alcotest.(check (float 1e-10)) "global SGD step norm" 0.05 (sqrt !norm2)

let minibatches () =
  let run () = Rollout.permutation ~rng:(Random.State.make [| 2 |]) 15 in
  let indices = run () in
  Alcotest.(check (array int)) "repeatable order" indices (run ());
  Array.sort Int.compare indices;
  Alcotest.(check (array int)) "every item once" (Array.init 15 Fun.id) indices

let () = Alcotest.run "actor-critic"
  [ "numerics", [ "GAE", `Quick, gae; "PyTorch losses and gradients", `Quick, losses; "global gradient norm", `Quick, gradient_clipping ]
  ; "rollout", [ "auto reset and final observations", `Quick, vector_and_collector; "minibatch order", `Quick, minibatches ]
  ; "learning", [ "A2C", `Quick, learning true; "PPO", `Quick, learning false; "continuous float64", `Quick, continuous ] ]
