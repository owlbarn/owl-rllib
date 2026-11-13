(* B6. GRPO: PPO's clipped objective without a critic.

   Sample a group of completions for every prompt; standardise their rewards
   within the group to get one advantage per completion, shared by all its
   tokens. The KL to the reference is a separate term of the loss (k3, with
   gradients) instead of a penalty in the reward. Three policies take part:
   pi_old sampled the batch, pi_theta is being optimised, pi_ref is frozen.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b6_grpo.exe *)
open Owl_rllib
open Owl_rllib_llm
open Common

let () =
  let iterations = ref 250 and kl_coef = ref R.grpo_default.kl_coef in
  Arg.parse
    [ "--iterations", Arg.Set_int iterations, "Training iterations"
    ; "--kl", Arg.Set_float kl_coef, "KL coefficient in the loss" ]
    (fun _ -> ()) "B6";
  let config = { R.grpo_default with kl_coef = !kl_coef } in
  let policy = fine_tuned () in
  let reference = R.Graph.copy policy in

  (* 1. Group-relative advantages for three prompts *)
  let batch = R.rollout ~rngs:(Rng.create 1) ~prompts:3 ~group_size:config.group_size ~problems:train policy in
  let advantages = Lm_rl.grpo ~group_size:config.group_size batch.rewards in
  Array.iteri
    (fun g _ ->
      let first = g * config.group_size in
      Printf.printf "%s (answer %s):\n" (Lm_task.to_string batch.problems.(first).prompt)
        (Lm_task.to_string batch.problems.(first).answer);
      for i = first to first + config.group_size - 1 do
        Printf.printf "  %-5s reward %.0f  advantage %+.3f\n" (Lm_task.to_string batch.samples.(i).tokens) batch.rewards.(i)
          advantages.(i)
      done)
    (Array.make 3 ());
  let zero = Array.fold_left (fun a g -> if Array.for_all (( = ) g.(0)) g then a + 1 else a) 0
      (Lm_rl.groups ~group_size:config.group_size (sample_batch ~rngs:(Rng.create 2) policy).rewards) in
  Printf.printf "In a batch of %d prompts, %d groups have the same reward everywhere: zero advantage, no gradient.\n"
    config.prompts zero;

  (* 2. Train *)
  Printf.printf "\nGRPO, KL coefficient %g, %d iterations of %d prompts x %d completions:\n" config.kl_coef !iterations
    config.prompts config.group_size;
  print_eval "  start" policy;
  let t0 = Unix.gettimeofday () in
  ignore (R.train_grpo ~config ~reference ~iterations:!iterations ~rngs:(Rng.create 7) ~problems:train policy
            ~report:(report ~every:50 policy));
  Printf.printf "  %.0f s, two networks: policy, frozen reference\n" (Unix.gettimeofday () -. t0)
