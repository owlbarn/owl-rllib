(* B7d. CISPO: clip the weight, not the gradient.

   In PPO's clipped objective, a token whose ratio has left the range on the
   side of its advantage contributes nothing to the gradient for the rest of
   the batch's updates. CISPO multiplies log pi by the clipped ratio, held
   constant (Algodiff.primal'), so every token keeps a gradient and the clip
   only caps its weight. It has no lower bound and averages over all tokens.

   1. 16 updates on one batch: the fraction of tokens that the clipped
      objective has stopped updating, epoch by epoch.
   2. Training with 16 updates per batch.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b7d_cispo.exe  (about 4 minutes) *)
open Owl_rllib
open Owl_rllib_llm
open Common

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations" ] (fun _ -> ()) "B7d";
  let start = fine_tuned () in
  let config = { grpo_base with epochs = 16 } in

  (* 1. One batch, 16 updates of each objective from the same start *)
  let batch = R.rollout ~rngs:(Rng.create 3) ~prompts:config.prompts ~group_size:config.group_size ~problems:train start in
  let advantages = R.per_row (Lm_rl.grpo ~group_size:config.group_size batch.rewards) batch in
  let x = R.features batch.contexts and old_log_prob = R.column batch.old_log_probs in
  let run loss =
    let policy = R.Graph.copy start in
    let step = R.adam_clipped ~max_grad_norm:config.max_grad_norm ~learning_rate:config.learning_rate policy in
    Array.init config.epochs (fun _ ->
        let logits, _ = R.Graph.forward policy x in
        let log_prob = R.token_log_probs logits batch.actions in
        let fraction =
          Lm_rl.clip_fraction ~clip_low:0.2 ~clip_high:0.2 ~log_ratios:(R.log_ratios ~log_prob batch) ~advantages
        in
        step (loss log_prob);
        fraction, Lm_rl.mean (R.exact_kl ~reference:start policy batch))
  in
  let weights agg = Lm_rl.row_weights agg ~lengths:batch.lengths batch.sequence in
  let grpo =
    run (fun log_prob ->
        R.clipped_loss ~clip:0.2 ~clip_high:0.2 ~log_prob ~old_log_prob ~advantages ~weights:(weights Lm_rl.Sequence_mean))
  in
  let cispo =
    run (fun log_prob ->
        R.cispo_loss ~clip:1. ~clip_high:0.2 ~log_prob ~old_log_prob ~advantages ~weights:(weights Lm_rl.Token_mean))
  in
  print_endline "Tokens outside [0.8, 1.2] on the side of their advantage, before each update, and KL to the start after it:";
  Printf.printf "  %5s %26s %9s %26s %9s\n" "epoch" "GRPO: no gradient" "KL" "CISPO: still updated" "KL";
  Array.iteri
    (fun e ((g, gk), (c, ck)) -> Printf.printf "  %5d %26.3f %9.4f %26.3f %9.4f\n" (e + 1) g gk c ck)
    (Array.map2 (fun a b -> a, b) grpo cispo);
  print_newline ();

  (* 2. Training, 16 updates per batch. The middle run isolates CISPO's
     objective: the same token-level mean, PPO's clip. *)
  List.iter
    (fun (label, config) -> ignore (train_variant ~iterations:!iterations label config start))
    [ "GRPO, 16 epochs", config
    ; "GRPO with a token-level mean, 16 epochs", { config with aggregation = Lm_rl.Token_mean }
    ; "CISPO, 16 epochs", R.cispo config ]
