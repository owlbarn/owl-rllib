(* A2C on CartPole with the RL Baselines3 Zoo budget: 8 environments,
   500k steps, entropy 0 (--entropy to change). The Zoo uses RMSprop; Algo.A2c
   uses Adam, otherwise the defaults agree (5 steps, lambda 1, learning rate
   7e-4, value 0.5, gradient norm 0.5).
   stdout: training episode returns (CSV); stderr: summary.
   OPENBLAS_NUM_THREADS=1 dune exec bench/a2c_zoo.exe -- --seed 0 *)
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 0 and steps = ref 500_000 and entropy = ref 0. in
  Arg.parse [ "--seed", Arg.Set_int seed, "Experiment seed"
            ; "--steps", Arg.Set_int steps, "Environment steps"
            ; "--entropy", Arg.Set_float entropy, "Entropy coefficient" ] (fun _ -> ()) "A2C (Zoo)";
  let rngs = Rng.create !seed in
  let policy = N.mlp ~seed:rngs.network_seed ~shape:[| 4 |] ~hidden:[ 64; 64 ] ~outputs:2 in
  let critic = N.mlp ~seed:(rngs.network_seed + 1) ~shape:[| 4 |] ~hidden:[ 64; 64 ] ~outputs:1 in
  let envs = Vec_env.create (Array.init 8 (fun _ -> Cartpole.make ~rng:rngs.env ())) in
  let config = { A2c.S.default with entropy_coef = !entropy } in
  let updates = !steps / (8 * config.horizon) in
  let start = Unix.gettimeofday () in
  let curve = A2c.S.train ~config ~updates ~rngs ~policy ~critic ~distribution:Policy.S.categorical envs in
  let seconds = Unix.gettimeofday () -. start in
  print_endline "episode,return";
  Array.iteri (fun i r -> Printf.printf "%d,%.0f\n" (i + 1) r) curve;
  let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs) in
  let last = Array.sub curve (max 0 (Array.length curve - 100)) (min 100 (Array.length curve)) in
  Printf.eprintf "env=cartpole entropy=%g seed=%d steps=%d last100=%.2f eval_mean=nan seconds=%.0f\n%!"
    !entropy !seed (updates * 8 * config.horizon) (mean last) seconds
