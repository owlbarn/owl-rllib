(* B7b. DAPO: three changes to GRPO.

   - Clip-higher: the ratio may rise to 1 + 0.28 but fall only to 1 - 0.2,
     so that unlikely tokens with a positive advantage can grow; the paper
     uses it against the collapse of the policy's entropy.
   - Dynamic sampling: a group whose rewards are all equal has zero advantage
     and no gradient. Keep sampling and keep only the other groups until the
     batch is full.
   - Token-level loss: the mean over all tokens of the batch (see B7a).

   DAPO's fourth change, overlong reward shaping, is not implemented: here a
   completion that reaches the length limit is simply wrong.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b7b_dapo.exe  (about 4 minutes) *)
open Owl_rllib
open Owl_rllib_llm
open Common

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations" ] (fun _ -> ()) "B7b";
  let start = fine_tuned () in
  let config = grpo_base in

  (* 1. Dynamic sampling on one batch *)
  let batch, first, completions =
    R.rollout_mixed ~task:config.task ~rngs:(Rng.create 2) ~prompts:config.prompts ~group_size:config.group_size
      ~problems:train start
  in
  Printf.printf "A batch of %d prompts x %d: %.0f%% of the groups have zero advantage.\n" config.prompts config.group_size
    (100. *. Lm_rl.zero_groups ~group_size:config.group_size first.rewards);
  Printf.printf "Dynamic sampling generated %d completions to fill %d groups with differing rewards.\n\n" completions
    (Array.length batch.rewards / config.group_size);

  (* 2. Each change alone, then all of them. The last run generates as many
     completions as dynamic sampling does, with twice the gradient steps. *)
  List.iter
    (fun (label, config, iterations) -> ignore (train_variant ~iterations label config start))
    [ "GRPO (no KL)", config, !iterations
    ; "clip-higher", { config with clip_high = 0.28 }, !iterations
    ; "dynamic sampling", { config with dynamic_sampling = true }, !iterations
    ; "DAPO: all three", R.dapo config, !iterations
    ; "GRPO, twice the iterations", config, 2 * !iterations ]
