(* B5. PPO at the token level.

   Every token is an action. A critic estimates the value of each state, the
   reward is the outcome at the last token minus beta * k1 at every token (the
   KL penalty), and GAE turns rewards and values into per-token advantages.
   The clipped objective lets each batch be used for several gradient steps
   without the policy moving too far from the one that sampled it.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b5_ppo.exe -- --kl 0 *)
open Owl_rllib
open Common

let () =
  let iterations = ref 250 and kl_coef = ref R.ppo_default.kl_coef in
  Arg.parse
    [ "--iterations", Arg.Set_int iterations, "Training iterations"
    ; "--kl", Arg.Set_float kl_coef, "KL penalty coefficient in the reward" ]
    (fun _ -> ()) "B5";
  let config = { R.ppo_default with kl_coef = !kl_coef } in
  let policy = fine_tuned () in
  let reference = R.Graph.copy policy in
  let critic = R.create ~seed:2 ~outputs:1 () in

  (* 1. Token-level advantages for a few completions, before any training *)
  let batch = R.rollout ~rngs:(Rng.create 1) ~prompts:3 ~group_size:1 ~problems:train policy in
  let values = R.Graph.run (R.features batch.contexts) critic |> R.Algodiff.unpack_arr |> R.values in
  let advantages, _ = R.token_advantages ~config ~ref_log_probs:(R.log_probs_of reference batch) ~values batch in
  print_endline "Token-level advantages with an untrained critic (policy = reference, so the KL penalty is 0):";
  Array.iteri
    (fun i s ->
      if i = 0 || batch.sequence.(i - 1) <> s then
        Printf.printf "  %s%s reward %.0f:" (Lm_task.to_string batch.problems.(s).prompt)
          (Lm_task.to_string batch.samples.(s).tokens) batch.rewards.(s);
      Printf.printf "  %s %+.3f" (Lm_task.to_string [| batch.actions.(i) |]) advantages.(i);
      if i = Array.length batch.sequence - 1 || batch.sequence.(i + 1) <> s then print_newline ())
    batch.sequence;

  (* 2. Train: 4 epochs per batch, one actor and one critic step per epoch *)
  Printf.printf "\nPPO, KL coefficient %g, %d iterations of %d completions:\n" config.kl_coef !iterations
    (config.prompts * config.group_size);
  print_eval "  start" policy;
  let t0 = Unix.gettimeofday () in
  ignore (R.train_ppo ~config ~reference ~critic ~iterations:!iterations ~rngs:(Rng.create 7) ~problems:train policy
            ~report:(report ~every:50 policy));
  Printf.printf "  %.0f s, three networks: policy, critic, frozen reference\n" (Unix.gettimeofday () -. t0)
