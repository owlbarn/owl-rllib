(* A5. Replay and target networks, with an ablation switch.
   OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a5_dqn.exe -- --steps 50000 *)
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 42 and steps = ref 50_000 and target = ref true and acrobot = ref false in
  Arg.parse
    [ "--seed", Arg.Set_int seed, "Experiment seed"
    ; "--steps", Arg.Set_int steps, "Environment steps"
    ; "--no-target", Arg.Clear target, "Bootstrap from online network"
    ; "--acrobot", Arg.Set acrobot, "Use Acrobot instead of CartPole" ]
    (fun _ -> ()) "A5";
  let rngs = Rng.create !seed in
  let make rng = if !acrobot then Acrobot.make ~rng () else Cartpole.make ~rng () in
  let shape = [| if !acrobot then 6 else 4 |] in
  let network = N.mlp ~seed:rngs.network_seed ~shape ~hidden:[ 64; 64 ] ~outputs:(if !acrobot then 3 else 2) in
  let config = { Dqn.S.default with target_network = !target } in
  let curve = Dqn.S.train ~config ~steps:!steps ~rngs ~network (make rngs.env) in
  print_endline "episode,return";
  Array.iteri (fun i r -> Printf.printf "%d,%.0f\n" (i + 1) r) curve;
  let eval = make rngs.eval in
  let scores = Array.init 20 (fun _ ->
    let rec go obs total =
      let action = N.predict network obs |> Dp.argmax in
      let st = eval.step action in
      let total = total +. st.reward in
      if st.terminated || st.truncated then total else go st.obs total
    in go (eval.reset ()) 0.) in
  Printf.eprintf "seed=%d eval_mean=%.1f\n%!" !seed (Array.fold_left ( +. ) 0. scores /. 20.)
