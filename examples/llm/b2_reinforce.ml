(* B2. REINFORCE on whole completions, with and without a baseline.

   The policy gradient of the expected reward is
     E[ R(y) * sum_t grad log pi(y_t | x, y_<t) ],
   and Lm_rl.pg_loss is a loss whose gradient is minus that estimate. Any
   baseline b that does not depend on the sampled completion can replace R by
   R - b without changing the expectation. Whether it reduces the variance
   depends on the task: here we measure it.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b2_reinforce.exe *)
open Owl_rllib
open Common

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations per estimator" ] (fun _ -> ()) "B2";
  (* 1. Same batches, same frozen policy: how noisy is each gradient estimate?
     Measured for the B starting point and for a model fine-tuned longer. *)
  List.iter
    (fun steps ->
      Printf.printf "Gradient variance, model fine-tuned for %d steps:\n" steps;
      let policy = fine_tuned ~steps () in
      let b = optimal_baseline policy in
      compare_variance
        [ "no baseline", Fun.id; "batch-mean baseline", Lm_rl.batch_mean_baseline
        ; Printf.sprintf "constant b* = %.3f" b, Array.map (fun r -> r -. b) ]
        policy;
      print_newline ())
    [ sft_steps; 1500 ];
  (* 2. Train. One gradient step per batch of 64 prompts x 8 completions. *)
  List.iter
    (fun (name, estimator) ->
      Printf.printf "REINFORCE, %s, %d iterations:\n" name !iterations;
      let policy = fine_tuned () in
      print_eval "  start" policy;
      ignore
        (R.train_pg ~estimator ~iterations:!iterations ~rngs:(Rng.create 7) ~problems:train policy
           ~report:(report ~every:50 policy));
      print_newline ())
    [ "no baseline", R.No_baseline; "batch-mean baseline", R.Batch_mean ]
