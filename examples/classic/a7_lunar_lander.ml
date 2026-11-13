(* A7, continued. PPO on Gymnasium's LunarLander-v2, run through the Python
   bridge, with the RL Baselines3 Zoo hyperparameters: 16 environments,
   horizon 1024, gamma 0.999, lambda 0.98, entropy 0.01.
   OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a7_lunar_lander.exe
   Needs Python with gymnasium and Box2D. *)
open Owl_rllib
open Owl_rllib_bridges
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 0 and steps = ref 1_000_000 in
  Arg.parse [ "--seed", Arg.Set_int seed, "Experiment seed"
            ; "--steps", Arg.Set_int steps, "Total steps (rounded down to complete rollouts)" ] (fun _ -> ()) "A7 LunarLander";
  let make rng = Gym.S.discrete ~seed:(Random.State.bits rng) "LunarLander-v2" in
  let rngs = Rng.create !seed in
  let eval_env = make rngs.eval in
  let policy = N.mlp ~seed:rngs.network_seed ~shape:[| 8 |] ~hidden:[ 64; 64 ] ~outputs:4 in
  let critic = N.mlp ~seed:(rngs.network_seed + 1) ~shape:[| 8 |] ~hidden:[ 64; 64 ] ~outputs:1 in
  let evaluate () =
    Array.init 20 (fun _ ->
      let rec go obs total =
        let st = eval_env.step (Dp.argmax (N.predict policy obs)) in
        let total = total +. st.reward in
        if st.terminated || st.truncated then total else go st.obs total
      in go (eval_env.reset ()) 0.)
    |> Array.fold_left ( +. ) 0. |> fun total -> total /. 20.
  in
  let initial = evaluate () in
  let env = Vec_env.create (Array.init 16 (fun _ -> make rngs.env)) in
  let config = { Ppo.S.default with horizon = 1024; gamma = 0.999; lambda = 0.98; entropy_coef = 0.01 } in
  let updates = !steps / (16 * config.horizon) in
  if updates = 0 then invalid_arg "A7: budget smaller than one rollout";
  let start = Unix.gettimeofday () in
  let curve = Ppo.S.train ~config ~updates ~rngs ~policy ~critic ~distribution:Policy.S.categorical env in
  print_endline "episode,return";
  Array.iteri (fun i score -> Printf.printf "%d,%.3f\n" (i + 1) score) curve;
  Printf.eprintf "seed=%d steps=%d initial_mean=%.2f eval_mean=%.2f seconds=%.0f\n%!"
    !seed (updates * 16 * config.horizon) initial (evaluate ()) (Unix.gettimeofday () -. start)
