(* DQN with CleanRL dqn.py's defaults, for comparison with its published
   classic-control table. CleanRL's DQN is the same algorithm as Algo.Dqn:
   MSE loss, Adam, hard target copies, bootstrap cut only on termination.
   stdout: training episode returns (CSV); stderr: summary.
   OPENBLAS_NUM_THREADS=1 dune exec bench/dqn_cleanrl.exe -- --env cartpole --seed 0 *)
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)

let config steps =
  { Dqn.S.gamma = 0.99; learning_rate = 2.5e-4; capacity = 10_000; batch_size = 128
  ; learning_starts = 10_000; train_every = 10; target_every = 500
  ; epsilon_start = 1.; epsilon_end = 0.05; exploration_steps = steps / 2
  ; target_network = true }

let () =
  let seed = ref 0 and steps = ref 500_000 and env = ref "cartpole" in
  Arg.parse [ "--seed", Arg.Set_int seed, "Experiment seed"
            ; "--steps", Arg.Set_int steps, "Environment steps"
            ; "--env", Arg.Set_string env, "cartpole, acrobot or mountain_car" ] (fun _ -> ()) "DQN (CleanRL)";
  let make, shape, n = match !env with
    | "cartpole" -> (fun rng -> Cartpole.make ~rng ()), 4, 2
    | "acrobot" -> (fun rng -> Acrobot.make ~rng ()), 6, 3
    | "mountain_car" -> (fun rng -> Mountain_car.make ~rng ()), 2, 3
    | e -> invalid_arg ("unknown env " ^ e) in
  let rngs = Rng.create !seed in
  let network = N.mlp ~seed:rngs.network_seed ~shape:[| shape |] ~hidden:[ 120; 84 ] ~outputs:n in
  let start = Unix.gettimeofday () in
  let curve = Dqn.S.train ~config:(config !steps) ~steps:!steps ~rngs ~network (make rngs.env) in
  let seconds = Unix.gettimeofday () -. start in
  print_endline "episode,return";
  Array.iteri (fun i r -> Printf.printf "%d,%.0f\n" (i + 1) r) curve;
  let eval = make rngs.eval in
  let scores = Array.init 20 (fun _ ->
    let rec go obs total =
      let st = eval.step (N.predict network obs |> Dp.argmax) in
      let total = total +. st.reward in
      if st.terminated || st.truncated then total else go st.obs total
    in go (eval.reset ()) 0.) in
  let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs) in
  let last = Array.sub curve (max 0 (Array.length curve - 100)) (min 100 (Array.length curve)) in
  Printf.eprintf "env=%s seed=%d steps=%d last100=%.2f eval_mean=%.2f seconds=%.0f\n%!"
    !env !seed !steps (mean last) (mean scores) seconds
