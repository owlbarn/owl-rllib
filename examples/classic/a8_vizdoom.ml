(* A8. Learning from pixels: PPO with a small CNN on VizDoom's basic scenario.
   The agent sees 60x80 grayscale frames, repeats each action for 4 tics
   (frame skip) and stacks the last 4 frames. Rewards are scaled by 0.01 for
   training (a kill is worth 100); returns are reported unscaled.
   OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a8_vizdoom.exe
   Needs Python with vizdoom. --frames 1 removes frame stacking. *)
open Owl_rllib
open Owl_rllib_bridges
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 0 and steps = ref 50_000 and frames = ref 4 in
  Arg.parse [ "--seed", Arg.Set_int seed, "Experiment seed"
            ; "--steps", Arg.Set_int steps, "Total steps (rounded down to complete rollouts)"
            ; "--frames", Arg.Set_int frames, "Frames stacked per observation" ] (fun _ -> ()) "A8";
  let make seed = Doom.S.make ~seed "basic" |> Frame_stack.S.wrap ~frames:!frames in
  let rngs = Rng.create !seed in
  let eval_env = make (Random.State.bits rngs.eval) in
  let shape = match eval_env.observation_space with Space.Box b -> b.shape | Discrete _ -> assert false in
  let n_actions = match eval_env.action_space with Space.Discrete n -> n | Box _ -> assert false in
  let network seed outputs = N.cnn ~seed ~shape ~convs:[ 8, 4, 16; 4, 2, 32 ] ~hidden:[ 128 ] ~outputs in
  let policy = network rngs.network_seed n_actions and critic = network (rngs.network_seed + 1) 1 in
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
  let reward_scale = 0.01 in
  let env = Array.init 8 (fun _ ->
    let env = make (Random.State.bits rngs.env) in
    { env with Ndenv.step = (fun action -> let st = env.step action in { st with reward = st.reward *. reward_scale }) })
    |> Vec_env.create in
  let config = { Ppo.S.default with entropy_coef = 0.01 } in
  let updates = !steps / (8 * config.horizon) in
  if updates = 0 then invalid_arg "A8: budget smaller than one rollout";
  let start = Unix.gettimeofday () in
  let curve = Ppo.S.train ~config ~updates ~rngs ~policy ~critic ~distribution:Policy.S.categorical env in
  print_endline "episode,return";
  Array.iteri (fun i score -> Printf.printf "%d,%.3f\n" (i + 1) (score /. reward_scale)) curve;
  Printf.eprintf "seed=%d steps=%d frames=%d initial_mean=%.2f eval_mean=%.2f seconds=%.0f\n%!"
    !seed (updates * 8 * config.horizon) !frames initial (evaluate ()) (Unix.gettimeofday () -. start)
