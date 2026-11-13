(* A7. Synchronous A2C and PPO, with discrete or continuous actions.
   OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a7_actor_critic.exe
   Add --pendulum for Gaussian actions, --a2c for single-pass updates. *)
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)
module A = Owl.Dense.Ndarray.S

let run ~seed ~steps ~a2c ~shape ~outputs ~distribution ~make ~deterministic ~reward_scale =
  let rngs = Rng.create seed in
  let policy = N.mlp ~seed:rngs.network_seed ~shape ~hidden:[ 64; 64 ] ~outputs in
  let critic = N.mlp ~seed:(rngs.network_seed + 1) ~shape ~hidden:[ 64; 64 ] ~outputs:1 in
  let evaluate () =
    let env = make (Random.State.make [| seed; 99 |]) in
    Array.init 20 (fun _ ->
      let rec go obs total =
        let action = deterministic (N.predict policy obs) in
        let st = env.Ndenv.step action in
        let total = total +. st.reward in
        if st.terminated || st.truncated then total else go st.obs total
      in go (env.reset ()) 0.)
    |> Array.fold_left ( +. ) 0. |> fun total -> total /. 20.
  in
  let initial = evaluate () in
  let env = Array.init 8 (fun i ->
    let env = make (Random.State.make [| seed; 0; i |]) in
    { env with Ndenv.step = (fun action -> let st = env.step action in { st with reward = st.reward *. reward_scale }) })
    |> Vec_env.create in
  let horizon = if a2c then A2c.S.default.horizon else Ppo.S.default.horizon in
  if steps < 8 * horizon then invalid_arg "A7: budget smaller than one rollout";
  let updates = steps / (8 * horizon) in
  let curve = if a2c then A2c.S.train ~updates ~rngs ~policy ~critic ~distribution env
    else Ppo.S.train ~updates ~rngs ~policy ~critic ~distribution env in
  print_endline "episode,return";
  Array.iteri (fun i score -> Printf.printf "%d,%.3f\n" (i + 1) (score /. reward_scale)) curve;
  Printf.eprintf "seed=%d steps=%d initial_mean=%.2f eval_mean=%.2f\n%!"
    seed (updates * 8 * horizon) initial (evaluate ())

let () =
  let seed = ref 0 and steps = ref 100_000 and a2c = ref false and pendulum = ref false in
  Arg.parse [ "--seed", Arg.Set_int seed, "Experiment seed"
            ; "--steps", Arg.Set_int steps, "Total steps (rounded down to complete rollouts)"
            ; "--a2c", Arg.Set a2c, "Use A2C instead of PPO"
            ; "--pendulum", Arg.Set pendulum, "Use continuous Pendulum" ] (fun _ -> ()) "A7";
  if !steps <= 0 then invalid_arg "A7: positive steps required";
  if !pendulum then
    run ~seed:!seed ~steps:!steps ~a2c:!a2c ~shape:[| 3 |] ~outputs:2
      ~distribution:(Policy.S.gaussian ~low:[| -2. |] ~high:[| 2. |])
      ~make:(fun rng -> Pendulum.make ~rng ())
      ~deterministic:(fun output -> A.of_array [| Float.max (-2.) (Float.min 2. output.(0)) |] [| 1 |])
      ~reward_scale:0.1
  else
    run ~seed:!seed ~steps:!steps ~a2c:!a2c ~shape:[| 4 |] ~outputs:2
      ~distribution:Policy.S.categorical ~make:(fun rng -> Cartpole.make ~rng ())
      ~deterministic:Dp.argmax ~reward_scale:1.
