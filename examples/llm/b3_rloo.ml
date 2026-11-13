(* B3. RLOO: a baseline from the other completions of the same prompt.

   Sample k completions per prompt. Each one's baseline is the mean reward of
   the other k - 1, which does not depend on it, so the estimate stays
   unbiased; and it is specific to the prompt, so easy and hard prompts no
   longer share one baseline.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b3_rloo.exe *)
open Owl_rllib
open Owl_rllib_llm
open Common

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations" ] (fun _ -> ()) "B3";
  let policy = fine_tuned () in
  (* 1. One group of eight completions for one prompt. *)
  let batch = R.rollout ~rngs:(Rng.create 1) ~prompts:1 ~group_size:8 ~problems:train policy in
  let rloo = Lm_rl.rloo ~group_size:8 batch.rewards in
  Printf.printf "Eight completions of %s (answer %s):\n" (Lm_task.to_string batch.problems.(0).prompt)
    (Lm_task.to_string batch.problems.(0).answer);
  Array.iteri
    (fun i (s : R.sample) ->
      Printf.printf "  %-5s reward %.0f   advantage: RLOO %+.3f\n" (Lm_task.to_string s.tokens) batch.rewards.(i) rloo.(i))
    batch.samples;
  (* 2. Variance, against the estimators of B2, at two success rates. *)
  List.iter
    (fun steps ->
      Printf.printf "\nGradient variance, model fine-tuned for %d steps:\n" steps;
      compare_variance
        [ "no baseline", Fun.id; "batch-mean baseline", Lm_rl.batch_mean_baseline
        ; "leave-one-out (RLOO)", Lm_rl.rloo ~group_size:R.pg_default.group_size ]
        (fine_tuned ~steps ()))
    [ sft_steps; 1500 ];
  (* 3. Train with RLOO. *)
  Printf.printf "\nRLOO, %d iterations:\n" !iterations;
  print_eval "  start" policy;
  ignore
    (R.train_pg ~estimator:R.Leave_one_out ~iterations:!iterations ~rngs:(Rng.create 7) ~problems:train policy
       ~report:(report ~every:50 policy))
