(* PPO with CleanRL ppo.py's defaults, for comparison with its published
   classic-control table: 4 environments, 128 steps, 4 epochs of 4
   minibatches, entropy 0.01, learning rate 2.5e-4. Not reproduced: CleanRL
   anneals the learning rate, clips the value loss, uses tanh layers with
   orthogonal initialisation and Adam eps 1e-5.
   stdout: training episode returns (CSV); stderr: summary.
   OPENBLAS_NUM_THREADS=1 dune exec bench/ppo_cleanrl.exe -- --env cartpole --seed 0 *)
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 0 and steps = ref 500_000 and env = ref "cartpole" in
  Arg.parse [ "--seed", Arg.Set_int seed, "Experiment seed"
            ; "--steps", Arg.Set_int steps, "Environment steps"
            ; "--env", Arg.Set_string env, "cartpole, acrobot or mountain_car" ] (fun _ -> ()) "PPO (CleanRL)";
  let make, shape, n = match !env with
    | "cartpole" -> (fun rng -> Cartpole.make ~rng ()), 4, 2
    | "acrobot" -> (fun rng -> Acrobot.make ~rng ()), 6, 3
    | "mountain_car" -> (fun rng -> Mountain_car.make ~rng ()), 2, 3
    | e -> invalid_arg ("unknown env " ^ e) in
  let rngs = Rng.create !seed in
  let policy = N.mlp ~seed:rngs.network_seed ~shape:[| shape |] ~hidden:[ 64; 64 ] ~outputs:n in
  let critic = N.mlp ~seed:(rngs.network_seed + 1) ~shape:[| shape |] ~hidden:[ 64; 64 ] ~outputs:1 in
  let envs = Vec_env.create (Array.init 4 (fun _ -> make rngs.env)) in
  let config = { Ppo.S.default with horizon = 128; batch_size = 128; learning_rate = 2.5e-4; entropy_coef = 0.01 } in
  let updates = !steps / (4 * config.horizon) in
  let start = Unix.gettimeofday () in
  let curve = Ppo.S.train ~config ~updates ~rngs ~policy ~critic ~distribution:Policy.S.categorical envs in
  let seconds = Unix.gettimeofday () -. start in
  print_endline "episode,return";
  Array.iteri (fun i r -> Printf.printf "%d,%.0f\n" (i + 1) r) curve;
  let eval = make rngs.eval in
  let scores = Array.init 20 (fun _ ->
    let rec go obs total =
      let st = eval.step (N.predict policy obs |> Dp.argmax) in
      let total = total +. st.reward in
      if st.terminated || st.truncated then total else go st.obs total
    in go (eval.reset ()) 0.) in
  let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs) in
  let last = Array.sub curve (max 0 (Array.length curve - 100)) (min 100 (Array.length curve)) in
  Printf.eprintf "env=%s seed=%d steps=%d last100=%.2f eval_mean=%.2f seconds=%.0f\n%!"
    !env !seed (updates * 4 * config.horizon) (mean last) (mean scores) seconds
