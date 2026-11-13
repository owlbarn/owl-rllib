(* A6. Policy gradients, optionally with a learned state-value baseline.
   OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a6_reinforce.exe -- --baseline *)
open Owl_rllib
module N = Neural.Make (Owl.Neural.S.Graph)

let () =
  let seed = ref 42 and episodes = ref 1000 and baseline = ref false and normalise = ref false in
  Arg.parse
    [ "--seed", Arg.Set_int seed, "Experiment seed"
    ; "--episodes", Arg.Set_int episodes, "Training episodes"
    ; "--baseline", Arg.Set baseline, "Learn a state-value baseline"
    ; "--normalise", Arg.Set normalise, "Normalize advantages (changes the estimator)" ]
    (fun _ -> ()) "A6";
  let rngs = Rng.create !seed in
  let policy = N.mlp ~seed:rngs.network_seed ~shape:[| 4 |] ~hidden:[ 64 ] ~outputs:2 in
  let critic = if !baseline then Some (N.mlp ~seed:(rngs.network_seed + 1) ~shape:[| 4 |] ~hidden:[ 64 ] ~outputs:1) else None in
  let curve = Reinforce.S.train ?baseline:critic ~normalise_advantage:!normalise
      ~episodes:!episodes ~rng:rngs.action ~policy (Cartpole.make ~rng:rngs.env ()) in
  print_endline "episode,return";
  Array.iteri (fun i r -> Printf.printf "%d,%.0f\n" (i + 1) r) curve;
  let eval = Cartpole.make ~rng:rngs.eval () in
  let scores = Array.init 20 (fun _ ->
    let rec go obs total =
      let action = Reinforce.S.act ~rng:rngs.eval policy obs in
      let st = eval.step action in
      let total = total +. st.reward in
      if st.terminated || st.truncated then total else go st.obs total
    in go (eval.reset ()) 0.) in
  Printf.eprintf "seed=%d eval_mean=%.1f\n%!" !seed (Array.fold_left ( +. ) 0. scores /. 20.)
