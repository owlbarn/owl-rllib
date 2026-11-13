(* Continuous-action PPO, with switches for CleanRL's implementation details.

   --preset cleanrl: CleanRL ppo_continuous_action.py's defaults, for its
     published MuJoCo table: 1 environment, 2048 steps, 10 epochs of 32
     minibatches, learning rate 3e-4 annealed, value clipping, tanh layers
     with orthogonal initialisation, state-independent log_std, observations
     and rewards normalised and clipped to [-10, 10], 1M steps.
   --preset a7: the Pendulum configuration of examples/classic/a7_actor_critic.ml
     (8 environments, 128 steps, 4 epochs, ReLU, state-dependent log_std,
     rewards x 0.1, 100k steps), where SB3 with the same settings reached
     -226. The switches add CleanRL's details to it one at a time.

   stdout: unscaled training episode returns (CSV); stderr: summary. The
   deterministic evaluation uses the mean action and is skipped when
   observations are normalised.
   OPENBLAS_NUM_THREADS=1 dune exec bench/ppo_continuous.exe -- --env Hopper-v4 --seed 0 *)
open Owl_rllib
open Owl_rllib_bridges
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 0 and env_id = ref "pendulum" and preset = ref "cleanrl" and steps = ref 0 in
  let independent = ref false and orthogonal = ref false and tanh = ref false in
  let normalise = ref false and clip_value = ref false and anneal = ref false and no_bootstrap = ref false in
  Arg.parse
    [ "--seed", Arg.Set_int seed, "Experiment seed"
    ; "--env", Arg.Set_string env_id, "pendulum (native) or a Gymnasium id"
    ; "--preset", Arg.Set_string preset, "cleanrl or a7"
    ; "--steps", Arg.Set_int steps, "Environment steps (default: the preset's)"
    ; "--state-independent-std", Arg.Set independent, "a7: log_std as one learned vector"
    ; "--orthogonal", Arg.Set orthogonal, "a7: orthogonal initialisation"
    ; "--tanh", Arg.Set tanh, "a7: tanh hidden layers"
    ; "--normalise", Arg.Set normalise, "a7: normalise observations and rewards"
    ; "--clip-value", Arg.Set clip_value, "a7: clip the value update"
    ; "--anneal", Arg.Set anneal, "a7: anneal the learning rate"
    ; "--truncation-as-termination", Arg.Set no_bootstrap,
      "PPO treats time limits as terminal, as CleanRL's GAE does (the wrappers still see the true flags)" ]
    (fun _ -> ()) "PPO (continuous)";
  let cleanrl = match !preset with "cleanrl" -> true | "a7" -> false | p -> invalid_arg ("unknown preset " ^ p) in
  if cleanrl then (independent := true; orthogonal := true; tanh := true; normalise := true; clip_value := true; anneal := true);
  let steps = if !steps > 0 then !steps else if cleanrl then 1_000_000 else 100_000 in
  let rngs = Rng.create !seed in
  let make rng =
    if !env_id = "pendulum" then Pendulum.make ~rng () else Gym.S.continuous ~seed:(Random.State.bits rng) !env_id in
  let probe = make rngs.eval in
  let obs_dim, low, high = match probe.observation_space, probe.action_space with
    | Space.Box o, Space.Box a -> Array.fold_left ( * ) 1 o.shape, a.low, a.high
    | _ -> invalid_arg "continuous observations and actions required" in
  let actions = Array.length low in
  let n_envs = if cleanrl then 1 else 8 in
  let reward_scale = if cleanrl then 1. else 0.1 in
  let gamma = 0.99 in
  (* raw returns are recorded before any reward transformation *)
  let returns = ref [] in
  let envs = Array.init n_envs (fun _ ->
    let env = make rngs.env and total = ref 0. in
    let env = { env with Ndenv.step = (fun a ->
      let st = env.step a in
      total := !total +. st.reward;
      if st.terminated || st.truncated then (returns := !total :: !returns; total := 0.);
      st) } in
    let env = if !normalise then Normalise.reward ~bound:10. ~gamma (Normalise.S.observation ~bound:10. env)
      else { env with step = (fun a -> let st = env.step a in { st with reward = st.reward *. reward_scale }) } in
    if not !no_bootstrap then env
    else { env with step = (fun a -> let st = env.step a in
                             { st with terminated = st.terminated || st.truncated; truncated = false }) })
    |> Vec_env.create in
  let activation = if !tanh then N.Activation.Tanh else N.Activation.Relu in
  let init gain = if !orthogonal then N.Orthogonal gain else N.Default in
  let shape = [| obs_dim |] and hidden = [ 64; 64 ] in
  let policy =
    if !independent then N.gaussian_mlp ~activation ~init:(init 0.01) ~seed:rngs.network_seed ~shape ~hidden ~actions
    else N.mlp_with ~activation ~init:(init 0.01) ~seed:rngs.network_seed ~shape ~hidden ~outputs:(2 * actions) in
  let critic = N.mlp_with ~activation ~init:(init 1.) ~seed:(rngs.network_seed + 1) ~shape ~hidden ~outputs:1 in
  let config =
    if cleanrl then { Ppo.S.default with horizon = 2048; epochs = 10; batch_size = 64; clip_value = true; anneal_lr = true }
    else { Ppo.S.default with clip_value = !clip_value; anneal_lr = !anneal } in
  let updates = steps / (n_envs * config.horizon) in
  let start = Unix.gettimeofday () in
  ignore (Ppo.S.train ~config ~updates ~rngs ~policy ~critic ~distribution:(Policy.S.gaussian ~low ~high) envs);
  let seconds = Unix.gettimeofday () -. start in
  let curve = Array.of_list (List.rev !returns) in
  print_endline "episode,return";
  Array.iteri (fun i r -> Printf.printf "%d,%.3f\n" (i + 1) r) curve;
  let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs) in
  let last = Array.sub curve (max 0 (Array.length curve - 100)) (min 100 (Array.length curve)) in
  let eval =
    if !normalise then nan
    else
      mean (Array.init 20 (fun _ ->
        let rec go obs total =
          let output = N.predict policy obs in
          let action = Owl.Dense.Ndarray.S.of_array (Array.init actions (fun j ->
            Float.max low.(j) (Float.min high.(j) output.(j)))) [| actions |] in
          let st = probe.step action in
          let total = total +. st.reward in
          if st.terminated || st.truncated then total else go st.obs total
        in go (probe.reset ()) 0.)) in
  Printf.eprintf "env=%s preset=%s truncation_as_termination=%b seed=%d steps=%d last100=%.2f eval_mean=%.2f seconds=%.0f\n%!"
    !env_id !preset !no_bootstrap !seed (updates * n_envs * config.horizon) (mean last) eval seconds
